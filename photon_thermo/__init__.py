"""
`photon_thermo` (`import photon_thermo as ptt`) v4.0.0 — Full Sovereign Ecosystem
=================================================================================
A PyTorch-grade Photonic-Thermodynamic AI Framework powered by Bare-Metal x86_64
AVX-512 Assembly (`thermo_asm_avx512.s`) and Stage-4 1-Quadrillion Kronecker-Holographic
Phase Manifolds (`stage4_quadrillion_avx512.c`).

Ecosystem Architecture:
  - `ptt.backend`    : Bare-Metal Assembly Linkage (`ctypes_loader`, `hardware_audit`, `vector_gates`)
  - `ptt.core`       : Multi-Paradigm Tensor Engine (`ptt.Tensor`, `.to_ising_spins()`, `.helmholtz_free_energy()`, `.mmap_freeze()`)
  - `ptt.nn`         : Neural Structural Blocks (`Module`, `Parameter`, `IsingSpinLinear`, `Embedding`, `RMSNorm`, `SwiGLUDarkFFN`, `KroneckerHolographicRouter`)
  - `ptt.functional` : Physical Wave Operations (`softmax_boltzmann`, `gabor_diffraction`, `sparsity_gate`, `silu_wave`, `rmsnorm`)
  - `ptt.thermal`    : Silicon Thermal Diode Noise Bath (`randn_rdseed`, `langevin_bath`, `boltzmann_schedule`)
  - `ptt.optim`      : 100% Backprop-Free Local Physical Learning (`EquilibriumPropagation`, `HebbianRelaxation`)
  - `ptt.converter`  : Universal Model Cross-Compiler Hub (`PyTorchTranslator`, `GGUFParser`)
  - `ptt.utils`      : High-Speed Orchestration (`data.WavefrontDataLoader`, `tokenizer.BPETokenizer`, `diagnostics.TelemetryAuditor`)
  - `ptt.solver`     : System-2 Helmholtz Coding & Reasoning Solver (`HelmholtzSystem2Solver`)
"""

import numpy as np
from . import backend
from . import core
from . import nn
from . import functional
from . import thermal
from . import optim
from . import converter
from . import utils
from . import solver
from . import finetune

from .backend import _asm, _s4, ctypes_loader, hardware_audit, vector_gates
from .core import Tensor

__version__ = "4.0.0"


def hardware_info() -> dict:
    return hardware_audit.audit()


def tensor(data, requires_spin: bool = True, name: str = "") -> Tensor:
    return Tensor(data, requires_spin=requires_spin, name=name)


def zeros(shape, requires_spin: bool = False) -> Tensor:
    return Tensor(np.zeros(shape, dtype=np.float32), requires_spin=requires_spin)


def ones(shape, requires_spin: bool = False) -> Tensor:
    return Tensor(np.ones(shape, dtype=np.float32), requires_spin=requires_spin)


def randn(shape, T: float = 1.0) -> Tensor:
    return thermal.randn_rdseed(shape, T=T)


__all__ = [
    "__version__",
    "backend",
    "core",
    "nn",
    "functional",
    "thermal",
    "optim",
    "converter",
    "utils",
    "solver",
    "finetune",
    "Tensor",
    "tensor",
    "zeros",
    "ones",
    "randn",
    "hardware_info",
    "_asm",
    "_s4",
]
