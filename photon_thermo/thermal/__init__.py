"""
Module 4: `ptt.thermal` — Hardware Silicon Thermal Noise Bath (`rdseed` / `k_B T`)
==================================================================================
  - `randn_rdseed(shape, T=1.0)`      : Hardware silicon diode entropy (`asm_rdseed_fill_u64`) -> Box-Muller Gaussian `ptt.Tensor`.
  - `langevin_bath(shape, tau=0.1)`   : Continuous-time Langevin stochastic heat bath to escape local minima.
  - `boltzmann_schedule(steps)`       : Thermodynamic annealing schedule across the Onsager critical point `T_c = 2.269`.
"""

import ctypes
import numpy as np
from ..backend import _asm
from ..core import Tensor


def randn_rdseed(shape, T: float = 1.0) -> Tensor:
    """
    Pulls genuine physical silicon thermal entropy (`k_B T`) straight from the CPU's
    on-die hardware entropy generator (`rdseed` / `rdrand` via `asm_rdseed_fill_u64`)
    and converts it via Box-Muller transform into a Gaussian `ptt.Tensor`.
    """
    if isinstance(shape, int):
        shape = (shape,)
    n_el = int(np.prod(shape))
    n_pairs = (n_el + 1) // 2

    raw_u64 = np.zeros(n_pairs * 2, dtype=np.uint64)
    _asm.asm_rdseed_fill_u64(
        raw_u64.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        n_pairs * 2,
    )

    # Convert 64-bit hardware entropy into uniform (0, 1) floats
    u1 = ((raw_u64[0::2] >> np.uint64(11)).astype(np.float64) + 1.0) / ((1 << 53) + 2.0)
    u2 = ((raw_u64[1::2] >> np.uint64(11)).astype(np.float64) + 1.0) / ((1 << 53) + 2.0)

    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2
    z = np.empty(n_pairs * 2, dtype=np.float32)
    z[0::2] = (r * np.cos(theta)).astype(np.float32)
    z[1::2] = (r * np.sin(theta)).astype(np.float32)

    scale = np.sqrt(max(float(T), 0.0))
    arr = (z[:n_el].reshape(shape) * scale).astype(np.float32)
    return Tensor(arr, requires_spin=False, name="thermal_rdseed")


def langevin_bath(shape, tau: float = 0.1, T: float = 0.25, steps: int = 4) -> Tensor:
    """
    Generates a correlated Ornstein-Uhlenbeck / Langevin stochastic heat bath:
        d_xi = -(xi / tau) * dt + sqrt(2 * T / tau) * dW_rdseed
    """
    if isinstance(shape, int):
        shape = (shape,)
    dt = 0.02
    decay = np.exp(-dt / max(float(tau), 1e-4))
    diffusion = np.sqrt(max(float(T), 1e-6) * (1.0 - decay * decay))
    state = randn_rdseed(shape, T=T).data
    for _ in range(max(int(steps) - 1, 0)):
        noise = randn_rdseed(shape, T=1.0).data
        state = decay * state + diffusion * noise
    return Tensor(state.astype(np.float32), requires_spin=False, name="langevin_bath")


def boltzmann_schedule(steps: int = 12, t_start: float = 2.50, t_end: float = 0.08) -> np.ndarray:
    """
    Exponential cooling schedule crossing the 2D Ising Onsager critical temperature `T_c = 2.269185`.
    """
    return np.geomspace(float(t_start), float(t_end), int(steps)).astype(np.float32)


__all__ = ["randn_rdseed", "langevin_bath", "boltzmann_schedule"]
