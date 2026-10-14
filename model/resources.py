"""Registers and static shared memory per kernel, from the compiled library (DESIGN.md: resources
from nvcc). model/resources.json is committed so the model runs without a build;
`python -m model resources` regenerates it from build/libkernels.so after a kernel changes.

Keys are demangled kernel names without parameter lists, e.g. "gemm_v3_kernel" or
"softmax_v2_kernel<16>".
"""
import json
import re
import subprocess
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = Path(__file__).resolve().parent / "resources.json"
LIB = ROOT / "build" / "libkernels.so"


def _short(demangled: str) -> str:
    name = demangled.split("(")[0].strip()
    name = re.sub(r"^void ", "", name)
    return name.replace("(bool)1", "true").replace("(bool)0", "false")


def refresh(lib: Path = LIB) -> dict:
    out = subprocess.run(["cuobjdump", "--dump-resource-usage", str(lib)], check=True,
                         capture_output=True, text=True).stdout
    table = {}
    for mangled, regs, shared in re.findall(r"Function (\S+):\s+REG:(\d+).*?SHARED:(\d+)", out):
        name = subprocess.run(["c++filt", mangled], check=True, capture_output=True,
                              text=True).stdout.strip()
        table[_short(name)] = {"regs": int(regs), "smem": int(shared)}
    PATH.write_text(json.dumps(dict(sorted(table.items())), indent=1) + "\n")
    load.cache_clear()
    return table


@lru_cache(maxsize=None)
def load() -> dict:
    return json.loads(PATH.read_text())


def get(kernel: str) -> dict:
    table = load()
    if kernel not in table:
        raise KeyError(f"{kernel} not in model/resources.json; run `python -m model resources`")
    return table[kernel]
