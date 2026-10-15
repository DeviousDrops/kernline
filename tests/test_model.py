"""The performance model (model/): no GPU needed."""
import csv

import pytest

from model import data, kernels, l2, ncu_check, params, spec, tiers
from model.kernels import gemm
from model.occupancy import occupancy


def test_occupancy_matches_ncu_for_every_committed_report():
    with ncu_check.FIXTURE.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 40
    assert ncu_check.check(rows) == []


def test_gemm_v3_is_register_limited_to_two_blocks():
    occ = occupancy(spec.load("rtx4070_laptop"), threads=256, regs=121, smem=8320)
    assert (occ.blocks_per_sm, occ.limiter) == (2, "registers")


def test_frozen_descriptor_interface():
    g = kernels.frozen(kernels.get("gemm"))
    assert g.flops("1024x1024x1024", "v1") == 2 * 1024**3
    assert g.launch_cfg("4096x4096x4096", "v3") == [("gemm_v3_kernel", 1024, 256, 0)]
    assert g.resources("v3")["gemm_v3_kernel"]["regs"] > 0
    assert g.bytes("4096x4096x4096", "v1") > 20 * g.bytes("4096x4096x4096", "v3")


@pytest.mark.parametrize("variant,dims,measured_gb", [
    ("v1", (4096, 4096, 4096), 35.0),   # ncu, results/ncu/rtx4070_laptop/metrics.csv
    ("v2", (4096, 4096, 4096), 9.2),
    ("v3", (4096, 4096, 4096), 1.21),
])
def test_panel_model_dram_bytes_near_ncu(variant, dims, measured_gb):
    gpu, p = spec.load("rtx4070_laptop"), params.load()
    L = gemm.launches(dims, variant, gpu, p)[0]
    assert (L.dram + L.dram_narrow) / 1e9 == pytest.approx(measured_gb, rel=0.15)


def test_l2_resident_gemm_moves_no_dram():
    a, bc = l2.gemm_panels(512, 512, 512, 128, 128, 4, 72, "n", 24e6)
    assert a == bc == 0


def test_tiers_only_add_time():
    gpu, p = spec.load("rtx4070_laptop"), params.load()
    for r in data.rows("rtx4070_laptop"):
        ls = kernels.get(r.kernel).launches(r.dims, r.variant, gpu, p)
        t = [tiers.predict(ls, gpu, p, tier, r.sm_mhz, r.mem_mhz)[0] for tier in tiers.TIERS]
        assert t[0] <= t[1] * (1 + 1e-9) <= t[2] * (1 + 1e-9), r


def test_params_were_tuned_on_the_4070_only():
    assert params.load()["tuned_on"].startswith("rtx4070_laptop")


def test_block_total_is_the_sum_of_its_steps():
    from model.kernels import block
    gpu, p = spec.load("rtx4070_laptop"), params.load()
    total = len(block.launches((2, 1024, 768, 12), "total", gpu, p))
    steps = sum(len(block.launches((2, 1024, 768, 12), s, gpu, p)) for s in block.STEPS)
    assert total == steps


def test_committed_params_are_what_tune_finds():
    from model import report
    fitted, _ = report.tune(data.rows("rtx4070_laptop"), spec.load("rtx4070_laptop"))
    committed = params.load()
    assert {k: committed[k] for k in fitted} == fitted
