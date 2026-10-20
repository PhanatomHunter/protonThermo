"""
Hardware-Targeted Cross-Compiler & Package Installer for `photon_thermo` (`ptt` v4.0.0)
"""

import os
import subprocess
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py


class BuildNativeAVX512(build_py):
    """Compiles `src/thermo_asm_avx512.s` and `src/stage4_quadrillion_avx512.c` into `.so` libraries."""

    def run(self):
        root = os.path.dirname(os.path.abspath(__file__))
        asm_src = os.path.join(root, "src", "thermo_asm_avx512.s")
        asm_so = os.path.join(root, "libphoton_thermo_asm.so")
        s4_asm_src = os.path.join(root, "src", "stage4_quadrillion_avx512.s")
        c_src = os.path.join(root, "src", "stage4_quadrillion_avx512.c")
        c_so = os.path.join(root, "libquadrillion_stage4.so")

        if os.path.exists(asm_src):
            subprocess.check_call(["gcc", "-O3", "-shared", "-fPIC", asm_src, "-o", asm_so])
        if os.path.exists(s4_asm_src):
            subprocess.check_call(["gcc", "-O3", "-shared", "-fPIC", s4_asm_src, "-o", c_so])
        elif os.path.exists(c_src):
            subprocess.check_call([
                "gcc", "-O3", "-mavx512f", "-mavx512bw", "-mavx512dq", "-mavx512vl",
                "-mavx512vpopcntdq", "-mavx512vnni", "-mfma", "-shared", "-fPIC", c_src, "-o", c_so, "-lm"
            ])
        super().run()


setup(
    name="photon_thermo",
    version="4.0.0",
    description="Sovereign Stage-4 1-Quadrillion Photonic-Thermodynamic AI Framework (x86_64 AVX-512 Assembly)",
    author="Photon-Thermo Sovereign Architecture",
    packages=find_packages(),
    cmdclass={"build_py": BuildNativeAVX512},
    python_requires=">=3.9",
    install_requires=["numpy>=1.22"],
)
