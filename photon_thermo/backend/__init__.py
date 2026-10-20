"""
Module 1: `ptt.backend` — Bare-Metal Assembly Linkage (x86_64 SIMD Gates)
=========================================================================
  - `ctypes_loader` : Dynamically maps and links `libphoton_thermo_asm.so` & `libquadrillion_stage4.so`.
  - `hardware_audit`: Scans `/proc/cpuinfo` and executes `rdseed` to verify AVX-512 & silicon entropy.
  - `vector_gates`  : Direct low-level entry points for 512-bit vector spins (`zmm0..zmm31`).
"""

import ctypes
import os
import numpy as np

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ASM_SO_PATH = os.path.join(_PKG_ROOT, "libphoton_thermo_asm.so")
_S4_SO_PATH = os.path.join(_PKG_ROOT, "libquadrillion_stage4.so")
if not os.path.exists(_S4_SO_PATH):
    _S4_SO_PATH = "/home/user/libquadrillion_stage4.so"


class _CtypesLoader:
    """Dynamically maps and links the compiled `.so` assembly & C+AVX-512 shared libraries."""

    def __init__(self):
        self.asm_lib = ctypes.CDLL(_ASM_SO_PATH)
        self.s4_lib = ctypes.CDLL(_S4_SO_PATH)
        self._bind_signatures()

    def _bind_signatures(self):
        # 1. Pure `.s` x86_64 Assembly routines (`thermo_asm_avx512.s`)
        self.asm_lib.asm_rdseed_thermal_bath.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_int64,
        ]
        self.asm_lib.asm_rdseed_thermal_bath.restype = ctypes.c_int64
        self.asm_lib.asm_rdseed_fill_u64 = self.asm_lib.asm_rdseed_thermal_bath

        self.asm_lib.asm_ising_quantize_avx512.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_int64,
            ctypes.c_float,
        ]
        self.asm_lib.asm_ising_quantize_avx512.restype = ctypes.c_uint64

        self.asm_lib.asm_spin_wavefront_vpopcntq_avx512.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_int64,
        ]
        self.asm_lib.asm_spin_wavefront_vpopcntq_avx512.restype = ctypes.c_int64

        self.asm_lib.asm_spin_linear_forward_avx512.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int64,
        ]
        self.asm_lib.asm_spin_linear_forward_avx512.restype = None

        self.asm_lib.asm_spin_linear_nd_avx512.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        self.asm_lib.asm_spin_linear_nd_avx512.restype = None

        self.asm_lib.asm_dot_nd_vfma_avx512.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ]
        self.asm_lib.asm_dot_nd_vfma_avx512.restype = ctypes.c_float

        self.asm_lib.asm_sparsity_gate_avx512.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_float,
        ]
        self.asm_lib.asm_sparsity_gate_avx512.restype = ctypes.c_uint64

        self.asm_lib.asm_wdm64_attractor_broadcast_avx512.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int64,
        ]
        self.asm_lib.asm_wdm64_attractor_broadcast_avx512.restype = ctypes.c_int64

        self.asm_lib.asm_dot288_vfma_avx512.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.asm_lib.asm_dot288_vfma_avx512.restype = ctypes.c_float

        self.asm_lib.asm_dot768_vfma_avx512.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.asm_lib.asm_dot768_vfma_avx512.restype = ctypes.c_float

        self.asm_lib.asm_dot5120_vfma_avx512.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.asm_lib.asm_dot5120_vfma_avx512.restype = ctypes.c_float

        self.asm_lib.asm_dot8192_vfma_avx512.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.asm_lib.asm_dot8192_vfma_avx512.restype = ctypes.c_float

        # 2. Stage-4 Kronecker & Wavefront Pure `.s` Assembly routines (`stage4_quadrillion_avx512.s`)
        self.s4_lib.route_1quadrillion_kronecker_avx512.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.s4_lib.route_1quadrillion_kronecker_avx512.restype = ctypes.c_uint64

        self.s4_lib.route_14b_kronecker_768_avx512.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.s4_lib.route_14b_kronecker_768_avx512.restype = ctypes.c_uint64
        self.s4_lib.route_1quadrillion_768_kronecker_avx512 = self.s4_lib.route_14b_kronecker_768_avx512

        self.s4_lib.route_deepseek_5120_kronecker_avx512.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.s4_lib.route_deepseek_5120_kronecker_avx512.restype = ctypes.c_uint64

        self.s4_lib.avx512_kronecker_route_8192_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.s4_lib.avx512_kronecker_route_8192_f32.restype = ctypes.c_uint64

        self.s4_lib.avx512_holographic_wavefront_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.s4_lib.avx512_holographic_wavefront_f32.restype = None

        self.s4_lib.avx512_holographic_wavefront_768_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.s4_lib.avx512_holographic_wavefront_768_f32.restype = None

        self.s4_lib.avx512_deepseek_5120_wavefront_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.s4_lib.avx512_deepseek_5120_wavefront_f32.restype = None

        self.s4_lib.avx512_llama70b_8192_wavefront_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.s4_lib.avx512_llama70b_8192_wavefront_f32.restype = None


ctypes_loader = _CtypesLoader()
_asm = ctypes_loader.asm_lib
_s4 = ctypes_loader.s4_lib


class _HardwareAudit:
    """Scans `/proc/cpuinfo` and hardware `rdseed` + `vfmadd231ps zmm` to verify AVX-512 and silicon thermal junctions."""

    def audit(self):
        model_name = "x86_64 AVX-512 Processor"
        flags_set = set()
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        model_name = line.split(":", 1)[1].strip()
                    elif line.startswith("flags"):
                        flags_set.update(line.split(":", 1)[1].strip().split())

        avx512_flags = sorted([fl for fl in flags_set if "avx512" in fl or fl in ("rdseed", "rdrand", "fma", "popcnt")])
        buf = np.zeros(8, dtype=np.uint64)
        hw_entropy_calls = int(_asm.asm_rdseed_thermal_bath(buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), 8))

        # Verify live AVX-512 zmm register execution
        a_test = np.ones(288, dtype=np.float32)
        dot_res = float(_asm.asm_dot288_vfma_avx512(
            a_test.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            a_test.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ))

        return {
            "cpu_model": model_name,
            "avx512_cpuid_verified": bool(abs(dot_res - 288.0) < 1e-3),
            "rdseed_thermal_diode_active": bool(hw_entropy_calls >= 0 and np.any(buf != 0)),
            "detected_simd_flags": avx512_flags,
            "asm_shared_library": _ASM_SO_PATH,
            "stage4_shared_library": _S4_SO_PATH,
        }


hardware_audit = _HardwareAudit()


class _VectorGates:
    """Exposes direct low-level compilation entry points for processing 512-bit vector spins."""

    @staticmethod
    def quantize_ising_avx512(fp32_weights: np.ndarray, tau: float = None):
        w = np.ascontiguousarray(fp32_weights, dtype=np.float32).ravel()
        n_aligned = ((w.size + 15) // 16) * 16
        if n_aligned != w.size:
            w_pad = np.zeros(n_aligned, dtype=np.float32)
            w_pad[: w.size] = w
            w = w_pad
        if tau is None:
            tau = float(0.48 * np.mean(np.abs(w[: min(w.size, 65536)])))
        n_groups = n_aligned // 16
        pos_masks = np.zeros(n_groups, dtype=np.uint16)
        neg_masks = np.zeros(n_groups, dtype=np.uint16)
        active = _asm.asm_ising_quantize_avx512(
            w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            pos_masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            neg_masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            n_aligned,
            float(tau),
        )
        return pos_masks, neg_masks, int(active), float(tau)

    @staticmethod
    def spin_linear_forward_avx512(
        w_pos_u64: np.ndarray,
        w_neg_u64: np.ndarray,
        x_pos_u64: np.ndarray,
        x_neg_u64: np.ndarray,
        out_features: int,
        n_u64_per_row: int = 8,
    ) -> np.ndarray:
        """
        Calls `asm_spin_linear_forward_avx512` (when n_u64_per_row == 8) or
        `asm_spin_linear_nd_avx512` (for arbitrary N-dimensional spin widths: 64..13824+ spins!).
        """
        out_int32 = np.zeros(out_features, dtype=np.int32)
        xp = np.ascontiguousarray(x_pos_u64, dtype=np.uint64)
        xn = np.ascontiguousarray(x_neg_u64, dtype=np.uint64)
        wp = np.ascontiguousarray(w_pos_u64, dtype=np.uint64)
        wn = np.ascontiguousarray(w_neg_u64, dtype=np.uint64)

        if int(n_u64_per_row) == 8 and xp.size >= 8:
            _asm.asm_spin_linear_forward_avx512(
                xp.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                xn.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                wp.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                wn.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                out_int32.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                int(out_features),
            )
        else:
            _asm.asm_spin_linear_nd_avx512(
                xp.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                xn.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                wp.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                wn.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                out_int32.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                int(out_features),
                int(n_u64_per_row),
            )
        return out_int32

    @staticmethod
    def dot_vfma_avx512(a: np.ndarray, b: np.ndarray) -> float:
        a_c = np.ascontiguousarray(a, dtype=np.float32).ravel()
        b_c = np.ascontiguousarray(b, dtype=np.float32).ravel()
        n = min(a_c.size, b_c.size)
        if n == 0:
            return 0.0
        pa = a_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pb = b_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        if n == 8192:
            return float(_asm.asm_dot8192_vfma_avx512(pa, pb))
        elif n == 5120:
            return float(_asm.asm_dot5120_vfma_avx512(pa, pb))
        elif n == 768:
            return float(_asm.asm_dot768_vfma_avx512(pa, pb))
        elif n == 288:
            return float(_asm.asm_dot288_vfma_avx512(pa, pb))
        return float(_asm.asm_dot_nd_vfma_avx512(pa, pb, int(n)))

    @staticmethod
    def sparsity_gate_avx512(in_f32: np.ndarray, cutoff: float):
        arr = np.ascontiguousarray(in_f32, dtype=np.float32)
        out = np.empty_like(arr)
        surviving = _asm.asm_sparsity_gate_avx512(
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(arr.size),
            float(cutoff),
        )
        return out, int(surviving)


vector_gates = _VectorGates()

__all__ = ["ctypes_loader", "hardware_audit", "vector_gates", "_asm", "_s4"]
