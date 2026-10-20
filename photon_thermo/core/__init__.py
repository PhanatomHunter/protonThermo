"""
Module 2: `ptt.core` — The Tensor Engine & Memory Landscapes
============================================================
  - `ptt.Tensor`                        : Multi-paradigm tensor object with automatic AVX-512 spin compilation.
  - `ptt.Tensor.to_ising_spins(tau)`    : Compiles continuous weights into `{-1, 0, +1}` bit-packed spin planes.
  - `ptt.Tensor.helmholtz_free_energy(T)`: Computes thermodynamic Helmholtz free energy `F = U - T * S`.
  - `ptt.Tensor.mmap_freeze(path)`      : Locks vectors into a zero-copy read-only `.ptc` `mmap` crystal.
"""

import math
import mmap
import os
import struct
import numpy as np
from ..backend import vector_gates


class Tensor:
    """
    Core Photonic-Thermodynamic Multi-Paradigm Tensor (`ptt.Tensor`).
    Supports both continuous wave amplitudes (`float32`) and AVX-512 bit-packed
    Ternary Ising Spin Planes (`pos_masks`, `neg_masks`).
    """

    def __init__(self, data, requires_spin: bool = True, name: str = ""):
        if isinstance(data, Tensor):
            self.data = data.data.copy()
        elif isinstance(data, np.ndarray):
            self.data = np.ascontiguousarray(data, dtype=np.float32)
        else:
            self.data = np.ascontiguousarray(np.array(data, dtype=np.float32), dtype=np.float32)

        self.requires_spin = requires_spin
        self.name = name
        self.pos_masks = None
        self.neg_masks = None
        self.active_spins = 0
        self.tau = 0.0
        self.alpha_scale = 1.0
        self.is_frozen = False
        self._mmap_file = None
        self._mmap_obj = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return Tensor(self.data.T, requires_spin=self.requires_spin)

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> float:
        return float(self.data.ravel()[0])

    def to_ising_spins(self, tau: float = None):
        """
        Compiles continuous weights into compact `{-1, 0, +1}` bit-packed spin planes
        via single-cycle AVX-512 vector masks (`vcmpgtps k1, zmm0, zmm30`, `kmovw`).
        """
        pos_m, neg_m, act, used_tau = vector_gates.quantize_ising_avx512(self.data, tau=tau)
        self.pos_masks = pos_m
        self.neg_masks = neg_m
        self.active_spins = act
        self.tau = used_tau
        abs_mean = float(np.mean(np.abs(self.data)))
        self.alpha_scale = abs_mean / max(act / max(self.size, 1), 1e-6)
        return self

    def spin_sparsity(self) -> dict:
        if self.pos_masks is None:
            self.to_ising_spins()
        active_pct = (self.active_spins / max(self.size, 1)) * 100.0
        return {
            "total_elements": self.size,
            "active_spins": self.active_spins,
            "active_pct": round(active_pct, 2),
            "dark_zero_pct": round(100.0 - active_pct, 2),
            "tau_threshold": round(self.tau, 6),
        }

    def helmholtz_free_energy(self, T: float = 0.25) -> float:
        """
        Measures the thermodynamic Helmholtz Free Energy `F = U - T * S`:
          - Internal Ginzburg-Landau potential `U = < -0.5 * x^2 + 0.25 * x^4 >`
          - Boltzmann-Shannon configurational entropy `S = - sum p_i ln(p_i)`
        """
        x = self.data.ravel().astype(np.float64)
        u_internal = float(np.mean(-0.5 * (x ** 2) + 0.25 * (x ** 4)))
        logits = x - np.max(x)
        exp_x = np.exp(np.clip(logits / max(T, 1e-4), -40.0, 0.0))
        probs = exp_x / (np.sum(exp_x) + 1e-12)
        entropy_s = -float(np.sum(probs * np.log(probs + 1e-15)))
        return float(u_internal - T * entropy_s)

    def mmap_freeze(self, file_path: str = "/home/user/models/frozen_tensor_crystal.ptc"):
        """
        Locks tensor data & Ising spin bit-planes into a zero-copy, read-only `.ptc`
        crystal file mapped straight into OS virtual address memory (`mmap.ACCESS_READ`).
        """
        if self.pos_masks is None:
            self.to_ising_spins()
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(struct.pack("<4sIQff", b"PTTC", 4, self.size, float(self.tau), float(self.alpha_scale)))
            f.write(self.pos_masks.tobytes())
            f.write(self.neg_masks.tobytes())

        self._mmap_file = open(file_path, "rb")
        self._mmap_obj = mmap.mmap(self._mmap_file.fileno(), 0, access=mmap.ACCESS_READ)
        hdr_sz = struct.calcsize("<4sIQff")
        n_u16 = self.pos_masks.size
        self.pos_masks = np.frombuffer(self._mmap_obj, dtype=np.uint16, count=n_u16, offset=hdr_sz)
        self.neg_masks = np.frombuffer(
            self._mmap_obj, dtype=np.uint16, count=n_u16, offset=hdr_sz + n_u16 * 2
        )
        self.is_frozen = True
        return file_path

    def close_mmap(self):
        if self._mmap_obj is not None:
            self.pos_masks = None
            self.neg_masks = None
            self._mmap_obj.close()
            self._mmap_file.close()
            self._mmap_obj = None
            self._mmap_file = None
            self.is_frozen = False

    def __matmul__(self, other):
        b = other.data if isinstance(other, Tensor) else np.asarray(other, dtype=np.float32)
        return Tensor(self.data @ b, requires_spin=self.requires_spin)

    def __add__(self, other):
        b = other.data if isinstance(other, Tensor) else np.asarray(other, dtype=np.float32)
        return Tensor(self.data + b, requires_spin=self.requires_spin)

    def __sub__(self, other):
        b = other.data if isinstance(other, Tensor) else np.asarray(other, dtype=np.float32)
        return Tensor(self.data - b, requires_spin=self.requires_spin)

    def __mul__(self, other):
        b = other.data if isinstance(other, Tensor) else np.asarray(other, dtype=np.float32)
        return Tensor(self.data * b, requires_spin=self.requires_spin)

    def __repr__(self):
        spin_tag = f", spins_compiled=True(active={self.active_spins})" if self.pos_masks is not None else ""
        frozen_tag = ", mmap_frozen=True" if self.is_frozen else ""
        return f"ptt.Tensor(shape={self.shape}, dtype=float32{spin_tag}{frozen_tag})"


__all__ = ["Tensor"]
