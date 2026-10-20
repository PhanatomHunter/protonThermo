"""
Example: Fine-Tuning the Stage-4 1.004-Quadrillion `DeepSeek-R1-Distill-Qwen-14B` Crystal
with Backprop-Free Contrastive Equilibrium Propagation + `SpinLoRAAdapter` (`ptt.finetune`)
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import photon_thermo as ptt


def main():
    dataset = [
        {
            "prompt": "Find all positive integer solutions (x, y) to x^2 - y^2 = 77.",
            "think": "<think> (x-y)(x+y)=77 => (1,77) and (7,11) => (39,38) and (9,2). </think>",
            "answer": "(x, y) = (39, 38) and (9, 2).",
        },
        {
            "prompt": "Evaluate the exact integral from 0 to pi/2 of sin^3(x)/(sin^3(x)+cos^3(x)) dx.",
            "think": "<think> By symmetry u = pi/2 - x, 2I = pi/2 => I = pi/4. </think>",
            "answer": "I = pi / 4 = 0.7853981634.",
        },
        {
            "prompt": "Why does a 4-stage Kronecker metasurface route 68,000,000 columns in 368 dot products?",
            "think": "<think> Tensor-product separability reduces O(M1*M2*M3*M4) to O(M1+M2+M3+M4)=368. </think>",
            "answer": "184,782x fewer AVX-512 operations with 99.9999985% dark manifold.",
        },
        {
            "prompt": "What is the exact Onsager critical temperature k_B*T_c/J of a 2D square Ising model?",
            "think": "<think> sinh(2J / k_B T_c) = 1 => k_B T_c / J = 2 / ln(1 + sqrt(2)) = 2.269185. </think>",
            "answer": "k_B * T_c / J = 2.269185314.",
        },
    ]

    pipeline = ptt.finetune.Stage4CrystalFineTuningPipeline(
        dim=5120, subspace_dim=256, lora_rank=16, lr=0.45, beta=0.20
    )
    report = pipeline.fit(dataset, epochs=6)
    out = pipeline.export_finetuned_crystal("/home/user/models/deepseek_r1_14b_stage4_finetuned.ptc")
    print(f"Fine-tuned in {report['finetune_time_ms']} ms | Final Cos={report['final_cosine']:.5f} | P(target)={report['final_attractor_prob']:.5f}")
    print(f"Exported crystal: {out['finetuned_crystal_path']} ({out['finetuned_crystal_bytes']} bytes)")


if __name__ == "__main__":
    main()
