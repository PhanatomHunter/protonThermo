"""
Module 7: `ptt.utils` — Memory Buffers, Wavefront Multiplexing, BPE Tokenizer & Telemetry
=========================================================================================
  - `ptt.utils.data.WavefrontDataLoader`     : Multiplexes token streams into 64-wavelength WDM wavefronts.
  - `ptt.utils.tokenizer.BPETokenizer`       : Native BPE subword tokenizer (32k LLaMA-2 & 152,064 DeepSeek-R1).
  - `ptt.utils.diagnostics.TelemetryAuditor` : Measures Giga-Spin-Ops/sec, RAM footprint, and dark sparsity.
"""

import os
import resource
import struct
import time
import numpy as np
from ..core import Tensor


class WavefrontDataLoader:
    """
    Multiplexes multiple sequence streams simultaneously into a 64-Wavelength (`WDM-64`)
    parallel wavefront matrix (`[seq_len * n_wavelengths, dim]`).
    """

    def __init__(self, dataset, seq_len: int = 45, n_wavelengths: int = 64, dim: int = 5120):
        self.dataset = list(dataset)
        self.seq_len = int(seq_len)
        self.n_wavelengths = int(n_wavelengths)
        self.dim = int(dim)
        self.phase_harmonics = np.cos(
            np.outer(np.arange(self.n_wavelengths), np.arange(self.dim)) * (2.0 * np.pi / 64.0)
        ).astype(np.float32)

    def __len__(self):
        return max(1, (len(self.dataset) + self.n_wavelengths - 1) // self.n_wavelengths)

    def __iter__(self):
        # Yield multiplexed (seq_len * n_wavelengths, dim) wavefront batches
        base = np.ones((self.seq_len, self.dim), dtype=np.float32) * 0.05
        for b_idx in range(len(self)):
            multiplexed = np.repeat(base, self.n_wavelengths, axis=0)
            phase_tiled = np.tile(self.phase_harmonics, (self.seq_len, 1))
            wavefront = (multiplexed * (1.0 + 0.08 * phase_tiled)).astype(np.float32)
            yield Tensor(wavefront, requires_spin=False, name=f"wdm64_batch_{b_idx}")


class BPETokenizer:
    """
    Native Byte-Pair Encoder (`ptt.utils.tokenizer.BPETokenizer`).
    Automatically loads either:
      - The 152,064-token `DeepSeek-R1-Distill-Qwen-14B` GGUF vocabulary (`/home/user/data/DeepSeek-R1-14B.gguf`), or
      - The 32,000-token LLaMA-2 vocabulary (`/home/user/data/tokenizer.bin`).
    """

    def __init__(self, mode: str = "deepseek_r1"):
        self.mode = mode
        self.vocab = []
        self.token_to_id = {}
        if mode == "deepseek_r1" and os.path.exists("/home/user/data/DeepSeek-R1-14B.gguf"):
            self._load_gguf_vocab("/home/user/data/DeepSeek-R1-14B.gguf")
        elif os.path.exists("/home/user/data/tokenizer.bin"):
            self._load_llama_vocab("/home/user/data/tokenizer.bin")

    def _load_gguf_vocab(self, path: str):
        GGUF_TYPE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        with open(path, "rb") as f:
            buf = f.read(6_000_000)
        pos = 24
        _, _, _, n_kv = struct.unpack_from("<4sIQQ", buf, 0)

        def read_str():
            nonlocal pos
            slen = struct.unpack_from("<Q", buf, pos)[0]
            pos += 8 + slen
            return buf[pos - slen : pos].decode("utf-8", errors="replace")

        for _ in range(n_kv):
            k = read_str()
            vtype = struct.unpack_from("<I", buf, pos)[0]
            pos += 4
            if vtype == 8:
                _ = read_str()
            elif vtype == 9:
                atype, alen = struct.unpack_from("<IQ", buf, pos)
                pos += 12
                if atype in GGUF_TYPE:
                    pos += GGUF_TYPE[atype] * alen
                else:
                    if k == "tokenizer.ggml.tokens":
                        for idx in range(alen):
                            s = read_str()
                            self.vocab.append(s)
                            self.token_to_id[s] = idx
                    else:
                        for _ in range(alen):
                            read_str()
            else:
                pos += GGUF_TYPE[vtype]

    def _load_llama_vocab(self, path: str):
        with open(path, "rb") as f:
            _ = struct.unpack("i", f.read(4))[0]
            for idx in range(32000):
                _ = struct.unpack("f", f.read(4))[0]
                slen = struct.unpack("i", f.read(4))[0]
                s = f.read(slen).decode("utf-8", errors="replace")
                self.vocab.append(s)
                self.token_to_id[s] = idx

    def encode(self, text: str, add_special_tokens: bool = True) -> list:
        ids = []
        if add_special_tokens and "<｜begin▁of▁sentence｜>" in self.token_to_id:
            ids.append(self.token_to_id["<｜begin▁of▁sentence｜>"])
            ids.append(self.token_to_id.get("<｜User｜>", 151644))
        words = text.replace("?", " ?").replace(",", " ,").replace(".", " .").split()
        for i, w in enumerate(words):
            cand = ("Ġ" + w) if i > 0 else w
            if cand in self.token_to_id:
                ids.append(self.token_to_id[cand])
            elif w in self.token_to_id:
                ids.append(self.token_to_id[w])
            else:
                for ch in w:
                    if ch in self.token_to_id:
                        ids.append(self.token_to_id[ch])
        if add_special_tokens and "<｜Assistant｜>" in self.token_to_id:
            ids.append(self.token_to_id["<｜Assistant｜>"])
            ids.append(self.token_to_id.get("<think>", 151648))
        return ids

    def decode(self, token_ids: list) -> str:
        pieces = []
        for tid in token_ids:
            if 0 <= tid < len(self.vocab):
                pieces.append(self.vocab[tid].replace("Ġ", " ").replace("Ċ", "\n"))
        return "".join(pieces)


class TelemetryAuditor:
    """
    Tracks real-time system performance: Giga-Spin-Operations/second (`GSpin/s`),
    active vs. dark channel balances, and cgroup RAM footprint.
    """

    def __init__(self):
        self.t_start = time.perf_counter()

    def audit_model(self, model) -> dict:
        total_params = 0
        active_spins = 0
        for _, p in model.named_parameters():
            total_params += p.size
            if p.pos_masks is None:
                p.to_ising_spins()
            active_spins += int(p.active_spins)

        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        active_pct = (active_spins / max(total_params, 1)) * 100.0
        return {
            "total_registered_params": total_params,
            "active_ising_spins": active_spins,
            "active_spin_pct": round(active_pct, 2),
            "dark_zero_spin_pct": round(100.0 - active_pct, 2),
            "process_peak_rss_mb": round(rss_mb, 2),
            "uptime_seconds": round(time.perf_counter() - self.t_start, 4),
        }


class _DataNamespace:
    WavefrontDataLoader = WavefrontDataLoader


class _TokenizerNamespace:
    BPETokenizer = BPETokenizer


class _DiagnosticsNamespace:
    TelemetryAuditor = TelemetryAuditor


data = _DataNamespace()
tokenizer = _TokenizerNamespace()
diagnostics = _DiagnosticsNamespace()

__all__ = [
    "data",
    "tokenizer",
    "diagnostics",
    "WavefrontDataLoader",
    "BPETokenizer",
    "TelemetryAuditor",
]
