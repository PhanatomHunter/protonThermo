"""
`ptt.nn.functional` (`ptt.functional` / `F`) — Physical Wave & Thermodynamic Operations
=======================================================================================
  - `softmax_boltzmann(logits, T=0.1)` : Executes the Boltzmann-Gibbs Softmax distribution at temperature T.
  - `gabor_diffraction(x, freq=2.0)`   : Spatial optical wave diffraction & phase-quadrature feature extraction.
  - `sparsity_gate(h, threshold=0.5)`  : Forces low-energy activation channels completely dark (0.0).
  - `silu_wave(x)`                     : Thermodynamic SwiGLU/SiLU wave activation.
  - `rmsnorm(x, weight, eps=1e-6)`     : Root-Mean-Square optical amplitude normalization.
"""

import numpy as np
from ..backend import vector_gates
from ..core import Tensor


def _unwrap(x):
    return x.data if isinstance(x, Tensor) else np.asarray(x, dtype=np.float32)


def _wrap(arr, ref):
    return Tensor(arr, requires_spin=ref.requires_spin) if isinstance(ref, Tensor) else arr


def softmax_boltzmann(logits, T: float = 0.1, axis: int = -1):
    """
    Executes the Boltzmann-Gibbs Softmax distribution natively using thermodynamic temperature `T`:
        P_i = exp(E_i / k_B T) / Z(T)
    """
    arr = _unwrap(logits).astype(np.float64)
    temp = max(float(T), 1e-5)
    shifted = (arr - np.max(arr, axis=axis, keepdims=True)) / temp
    exp_v = np.exp(np.clip(shifted, -60.0, 0.0))
    probs = (exp_v / (np.sum(exp_v, axis=axis, keepdims=True) + 1e-15)).astype(np.float32)
    return _wrap(probs, logits)


def gabor_diffraction(x, freq: float = 2.0, sigma: float = 1.0):
    """
    Implements spatial wave feature extraction inspired by optical 4f Fourier lenses:
        G(x) = exp(-x^2 / (2 * sigma^2)) * cos(2 * pi * freq * x) + x
    """
    arr = _unwrap(x)
    envelope = np.exp(-(arr ** 2) / (2.0 * (sigma ** 2) + 1e-9))
    diffracted = arr + 0.25 * envelope * np.cos(2.0 * np.pi * freq * arr)
    return _wrap(diffracted.astype(np.float32), x)


def sparsity_gate(h, threshold: float = 0.5):
    """
    Automatically forces low-energy activation signals to go completely dark (`0.0`)
    using our hand-written x86_64 AVX-512 Assembly zero-masking kernel
    `asm_sparsity_gate_avx512` (`vcmpps k1, zmm1, zmm30, 13` + `vmovups zmm2 {k1}{z}, zmm0`)!
    """
    arr = _unwrap(h)
    if arr.size == 0:
        return _wrap(arr.copy(), h)
    abs_arr = np.abs(arr)
    if 0.0 < threshold < 1.0:
        cutoff = float(np.quantile(abs_arr, threshold))
    elif threshold <= 0.0:
        return _wrap(arr.copy(), h)
    else:
        cutoff = float(threshold)
    gated, _ = vector_gates.sparsity_gate_avx512(arr, cutoff)
    return _wrap(gated, h)


def silu_wave(x):
    arr = _unwrap(x)
    out = arr / (1.0 + np.exp(-np.clip(arr, -20.0, 20.0)))
    return _wrap(out.astype(np.float32), x)


def rmsnorm(x, weight=None, eps: float = 1e-6):
    arr = _unwrap(x)
    rms = np.sqrt(np.mean(arr * arr, axis=-1, keepdims=True) + eps)
    normed = arr / rms
    if weight is not None:
        normed = normed * _unwrap(weight)
    return _wrap(normed.astype(np.float32), x)


__all__ = [
    "softmax_boltzmann",
    "gabor_diffraction",
    "sparsity_gate",
    "silu_wave",
    "rmsnorm",
]
