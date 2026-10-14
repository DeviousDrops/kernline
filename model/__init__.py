"""KernLine analytical performance model (Part A, docs/DESIGN.md D1-D3).

Predicts kernel runtimes from measured ceilings (specs/<gpu>.json), each kernel's launch shape
and resource use (model/resources.json), and per-kernel descriptors (model/kernels/) that count
FLOPs and traffic at each level of the memory hierarchy from the kernel's index math. Three tiers
(D2), each adding effects to the previous one:

  T0  roofline: the slowest of compute, DRAM, L2, shared memory and the load/store unit
  T1  + occupancy and wave quantization (blocks per SM, partial last waves)
  T2  + latency hiding (Little's law, synchronous tile loads) and launch overhead

`python -m model --help` lists the commands. Parameters are tuned on the 4070 Laptop only
(model/params.json); the 4060 Laptop is predicted with no refitting (D1).
"""
