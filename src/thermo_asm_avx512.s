/*
 * /home/user/photon_thermo/src/thermo_asm_avx512.s
 * ================================================
 * 100% Pure Hand-Written x86_64 Intel-Syntax AVX-512 Assembly Core for `photon_thermo` (ptt v4.0.0)
 *
 * Exports 11 Bare-Metal x86_64 AVX-512 Assembly Kernels:
 *   1.  asm_rdseed_thermal_bath               : Silicon thermal diode entropy (`rdseed`/`rdrand`/`rdtscp`)
 *   2.  asm_ising_quantize_avx512             : 16-float -> 16-spin bit-packer (`vcmpps k1` + `kmovw` + `popcnt`)
 *   3.  asm_spin_wavefront_vpopcntq_avx512    : 512-spin single-cycle interference (`vpandq` + `vpopcntq zmm`)
 *   4.  asm_dot288_vfma_avx512                : 288-dim AVX-512 FMA inner product (`vfmadd231ps zmm`)
 *   5.  asm_dot768_vfma_avx512                : 768-dim AVX-512 FMA inner product
 *   6.  asm_dot5120_vfma_avx512               : 5120-dim AVX-512 FMA inner product (`DeepSeek-R1-14B`)
 *   7.  asm_dot_nd_vfma_avx512                : Arbitrary N-dim AVX-512 FMA inner product (with scalar tail)
 *   8.  asm_wdm64_attractor_broadcast_avx512  : 64-Wavelength WDM spin wavefront broadcast (`vpopcntq` + `vpbroadcastd`)
 *   9.  asm_spin_linear_forward_avx512        : Fast 512-spin single-zmm linear forward pass
 *   10. asm_spin_linear_nd_avx512             : Arbitrary N-u64 multi-zmm `vpopcntq` + 64-bit `popcnt` linear forward pass
 *   11. asm_sparsity_gate_avx512              : AVX-512 zero-masking Dark-Channel gate (`vcmpps k1` + `vmovups {k1}{z}`)
 */

.intel_syntax noprefix
.text

/* ====================================================================================
 * 1. asm_rdseed_thermal_bath(uint64_t *out_buf [rdi], int64_t count [rsi]) -> rax
 * ==================================================================================== */
.globl asm_rdseed_thermal_bath
.type asm_rdseed_thermal_bath, @function
asm_rdseed_thermal_bath:
    push rbx
    xor r8, r8
    xor r9, r9
    mov r10, rsi
    test r10, r10
    jle .L_thermal_done

.L_thermal_loop:
    xor rax, rax
    rdseed rax
    jc .L_got_hw_entropy
    rdrand rax
    jnc .L_clock_jitter
.L_got_hw_entropy:
    inc r9

.L_clock_jitter:
    mov rbx, rax
    rdtscp
    shl rdx, 32
    or rax, rdx
    mov rcx, 0x9E3779B97F4A7C15
    imul rax, rcx
    xor rbx, rax
    mov [rdi + r8*8], rbx
    inc r8
    cmp r8, r10
    jl .L_thermal_loop

.L_thermal_done:
    mov rax, r9
    pop rbx
    ret


/* ====================================================================================
 * 2. asm_ising_quantize_avx512(
 *        const float *weights [rdi],
 *        uint16_t *pos_masks  [rsi],
 *        uint16_t *neg_masks  [rdx],
 *        int64_t n_floats     [rcx],
 *        float tau            [xmm0]
 *    ) -> rax (total_active_spins)
 * ==================================================================================== */
.globl asm_ising_quantize_avx512
.type asm_ising_quantize_avx512, @function
asm_ising_quantize_avx512:
    vbroadcastss zmm30, xmm0
    vxorps zmm29, zmm29, zmm29
    vsubps zmm31, zmm29, zmm30

    xor r8, r8
    xor r9, r9
    xor rax, rax

    mov r10, rcx
    shr r10, 4
    test r10, r10
    jle .L_quant_done

.L_quant_loop:
    vmovups zmm0, [rdi + r8*4]
    vcmpps k1, zmm0, zmm30, 14          /* k1 = (w > +tau) */
    vcmpps k2, zmm0, zmm31, 1           /* k2 = (w < -tau) */

    kmovw [rsi + r9*2], k1
    kmovw [rdx + r9*2], k2

    korw k3, k1, k2
    kmovw r11d, k3
    popcnt r11d, r11d
    add rax, r11

    add r8, 16
    inc r9
    dec r10
    jnz .L_quant_loop

.L_quant_done:
    vzeroupper
    ret


/* ====================================================================================
 * 3. asm_spin_wavefront_vpopcntq_avx512(
 *        const uint64_t *x_pos [rdi],
 *        const uint64_t *x_neg [rsi],
 *        const uint64_t *w_pos [rdx],
 *        const uint64_t *w_neg [rcx],
 *        int64_t n_u64_chunks  [r8]
 *    ) -> rax (net_spin_interference)
 * ==================================================================================== */
.globl asm_spin_wavefront_vpopcntq_avx512
.type asm_spin_wavefront_vpopcntq_avx512, @function
asm_spin_wavefront_vpopcntq_avx512:
    vpxorq zmm10, zmm10, zmm10
    vpxorq zmm11, zmm11, zmm11

    xor r9, r9
    mov r10, r8
    shr r10, 3
    test r10, r10
    jle .L_spin_done

.L_spin_loop:
    vmovdqu64 zmm0, [rdi + r9*8]
    vmovdqu64 zmm1, [rsi + r9*8]
    vmovdqu64 zmm2, [rdx + r9*8]
    vmovdqu64 zmm3, [rcx + r9*8]

    vpandq zmm4, zmm0, zmm2
    vpandq zmm5, zmm1, zmm3
    vporq  zmm4, zmm4, zmm5
    vpopcntq zmm4, zmm4
    vpaddq zmm10, zmm10, zmm4

    vpandq zmm6, zmm0, zmm3
    vpandq zmm7, zmm1, zmm2
    vporq  zmm6, zmm6, zmm7
    vpopcntq zmm6, zmm6
    vpaddq zmm11, zmm11, zmm6

    add r9, 8
    dec r10
    jnz .L_spin_loop

.L_spin_done:
    vpsubq zmm12, zmm10, zmm11
    vextracti64x4 ymm0, zmm12, 1
    vpaddq ymm0, ymm0, ymm12
    vextracti128 xmm1, ymm0, 1
    vpaddq xmm0, xmm0, xmm1
    vpsrldq xmm1, xmm0, 8
    vpaddq xmm0, xmm0, xmm1
    vmovq rax, xmm0
    vzeroupper
    ret


/* ====================================================================================
 * 4. asm_dot288_vfma_avx512(const float *a [rdi], const float *b [rsi]) -> xmm0
 * ==================================================================================== */
.globl asm_dot288_vfma_avx512
.type asm_dot288_vfma_avx512, @function
asm_dot288_vfma_avx512:
    vxorps zmm0, zmm0, zmm0
    xor rcx, rcx
.L_dot288_loop:
    vmovups zmm1, [rdi + rcx*4]
    vfmadd231ps zmm0, zmm1, [rsi + rcx*4]
    add rcx, 16
    cmp rcx, 288
    jl .L_dot288_loop

    vextractf32x8 ymm1, zmm0, 1
    vaddps ymm0, ymm0, ymm1
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vzeroupper
    ret


/* ====================================================================================
 * 5. asm_dot768_vfma_avx512(const float *a [rdi], const float *b [rsi]) -> xmm0
 * ==================================================================================== */
.globl asm_dot768_vfma_avx512
.type asm_dot768_vfma_avx512, @function
asm_dot768_vfma_avx512:
    vxorps zmm0, zmm0, zmm0
    xor rcx, rcx
.L_dot768_loop:
    vmovups zmm1, [rdi + rcx*4]
    vfmadd231ps zmm0, zmm1, [rsi + rcx*4]
    vmovups zmm2, [rdi + rcx*4 + 64]
    vfmadd231ps zmm0, zmm2, [rsi + rcx*4 + 64]
    add rcx, 32
    cmp rcx, 768
    jl .L_dot768_loop

    vextractf32x8 ymm1, zmm0, 1
    vaddps ymm0, ymm0, ymm1
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vzeroupper
    ret


/* ====================================================================================
 * 6. asm_dot5120_vfma_avx512(const float *a [rdi], const float *b [rsi]) -> xmm0
 * ==================================================================================== */
.globl asm_dot5120_vfma_avx512
.type asm_dot5120_vfma_avx512, @function
asm_dot5120_vfma_avx512:
    vxorps zmm0, zmm0, zmm0
    vxorps zmm3, zmm3, zmm3
    xor rcx, rcx
.L_dot5120_loop:
    vmovups zmm1, [rdi + rcx*4]
    vfmadd231ps zmm0, zmm1, [rsi + rcx*4]
    vmovups zmm2, [rdi + rcx*4 + 64]
    vfmadd231ps zmm3, zmm2, [rsi + rcx*4 + 64]
    add rcx, 32
    cmp rcx, 5120
    jl .L_dot5120_loop

    vaddps zmm0, zmm0, zmm3
    vextractf32x8 ymm1, zmm0, 1
    vaddps ymm0, ymm0, ymm1
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vzeroupper
    ret


/* ====================================================================================
 * 7. asm_dot_nd_vfma_avx512(const float *a [rdi], const float *b [rsi], int64_t n [rdx]) -> xmm0
 * ==================================================================================== */
.globl asm_dot_nd_vfma_avx512
.type asm_dot_nd_vfma_avx512, @function
asm_dot_nd_vfma_avx512:
    vxorps zmm0, zmm0, zmm0
    xor rcx, rcx
    mov r8, rdx
    and r8, -16
    test r8, r8
    jle .L_dotnd_reduce

.L_dotnd_vec:
    vmovups zmm1, [rdi + rcx*4]
    vfmadd231ps zmm0, zmm1, [rsi + rcx*4]
    add rcx, 16
    cmp rcx, r8
    jl .L_dotnd_vec

.L_dotnd_reduce:
    vextractf32x8 ymm1, zmm0, 1
    vaddps ymm0, ymm0, ymm1
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0

    cmp rcx, rdx
    jge .L_dotnd_done
.L_dotnd_tail:
    vmovss xmm2, dword ptr [rdi + rcx*4]
    vfmadd231ss xmm0, xmm2, dword ptr [rsi + rcx*4]
    inc rcx
    cmp rcx, rdx
    jl .L_dotnd_tail

.L_dotnd_done:
    vzeroupper
    ret


/* ====================================================================================
 * 8. asm_wdm64_attractor_broadcast_avx512
 * ==================================================================================== */
.globl asm_wdm64_attractor_broadcast_avx512
.type asm_wdm64_attractor_broadcast_avx512, @function
asm_wdm64_attractor_broadcast_avx512:
    mov r10, [rsp + 8]
    xor r11, r11
    xor rax, rax
    test r10, r10
    jle .L_wdm64_done

.L_wdm64_loop:
    mov rcx, r11
    shl rcx, 6

    vmovdqu64 zmm0, [rdi + rcx]
    vmovdqu64 zmm1, [rsi + rcx]
    vmovdqu64 zmm2, [rdx + rcx]

    vpandq zmm4, zmm0, zmm2
    vpandq zmm5, zmm1, zmm1
    vporq  zmm4, zmm4, zmm5
    vpopcntq zmm4, zmm4

    vextracti64x4 ymm6, zmm4, 1
    vpaddq ymm6, ymm6, ymm4
    vextracti128 xmm7, ymm6, 1
    vpaddq xmm6, xmm6, xmm7
    vmovq rcx, xmm6
    add rax, rcx

    vpbroadcastd zmm8, dword ptr [r8 + r11*4]
    mov rcx, r11
    shl rcx, 8
    vmovdqu64 [r9 + rcx],       zmm8
    vmovdqu64 [r9 + rcx + 64],  zmm8
    vmovdqu64 [r9 + rcx + 128], zmm8
    vmovdqu64 [r9 + rcx + 192], zmm8

    inc r11
    cmp r11, r10
    jl .L_wdm64_loop

.L_wdm64_done:
    vzeroupper
    ret


/* ====================================================================================
 * 9. asm_spin_linear_forward_avx512 (Fast single-zmm 512-spin kernel)
 * ==================================================================================== */
.globl asm_spin_linear_forward_avx512
.type asm_spin_linear_forward_avx512, @function
asm_spin_linear_forward_avx512:
    vmovdqu64 zmm0, [rdi]
    vmovdqu64 zmm1, [rsi]
    xor r10, r10
    test r9, r9
    jle .L_lin_done

.L_lin_loop:
    mov r11, r10
    shl r11, 6
    vmovdqu64 zmm2, [rdx + r11]
    vmovdqu64 zmm3, [rcx + r11]

    vpandq zmm4, zmm0, zmm2
    vpandq zmm5, zmm1, zmm3
    vporq  zmm4, zmm4, zmm5
    vpopcntq zmm4, zmm4

    vpandq zmm6, zmm0, zmm3
    vpandq zmm7, zmm1, zmm2
    vporq  zmm6, zmm6, zmm7
    vpopcntq zmm6, zmm6

    vpsubq zmm4, zmm4, zmm6
    vextracti64x4 ymm5, zmm4, 1
    vpaddq ymm5, ymm5, ymm4
    vextracti128 xmm6, ymm5, 1
    vpaddq xmm5, xmm5, xmm6
    vpsrldq xmm6, xmm5, 8
    vpaddq xmm5, xmm5, xmm6
    vmovd eax, xmm5
    mov [r8 + r10*4], eax

    inc r10
    cmp r10, r9
    jl .L_lin_loop

.L_lin_done:
    vzeroupper
    ret


/* ====================================================================================
 * 10. asm_spin_linear_nd_avx512(
 *         const uint64_t *x_pos       [rdi],
 *         const uint64_t *x_neg       [rsi],
 *         const uint64_t *W_pos       [rdx],
 *         const uint64_t *W_neg       [rcx],
 *         int32_t        *Out_i32     [r8],
 *         int64_t         out_features[r9],
 *         int64_t         n_u64_row   [rsp+8]
 *     ) -> void
 *     Full N-dimensional Multi-ZMM + 64-bit Tail Spin Linear Layer in Pure `.s` Assembly!
 * ==================================================================================== */
.globl asm_spin_linear_nd_avx512
.type asm_spin_linear_nd_avx512, @function
asm_spin_linear_nd_avx512:
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov r10, [rsp + 56]                 /* r10 = n_u64_per_row */
    test r9, r9
    jle .L_nd_done
    test r10, r10
    jle .L_nd_done

    mov r11, r10
    and r11, -8                         /* r11 = (n_u64_per_row / 8) * 8 */
    xor r12, r12                        /* r12 = output row j = 0 */

.L_nd_row_loop:
    vpxorq zmm10, zmm10, zmm10
    xor rax, rax                        /* rax = u64 chunk index k = 0 */
    test r11, r11
    jle .L_nd_reduce

.L_nd_vec_loop:
    vmovdqu64 zmm0, [rdi + rax*8]
    vmovdqu64 zmm1, [rsi + rax*8]
    vmovdqu64 zmm2, [rdx + rax*8]
    vmovdqu64 zmm3, [rcx + rax*8]

    vpandq zmm4, zmm0, zmm2
    vpandq zmm5, zmm1, zmm3
    vporq  zmm4, zmm4, zmm5
    vpopcntq zmm4, zmm4

    vpandq zmm6, zmm0, zmm3
    vpandq zmm7, zmm1, zmm2
    vporq  zmm6, zmm6, zmm7
    vpopcntq zmm6, zmm6

    vpsubq zmm4, zmm4, zmm6
    vpaddq zmm10, zmm10, zmm4

    add rax, 8
    cmp rax, r11
    jl .L_nd_vec_loop

.L_nd_reduce:
    vextracti64x4 ymm0, zmm10, 1
    vpaddq ymm0, ymm0, ymm10
    vextracti128 xmm1, ymm0, 1
    vpaddq xmm0, xmm0, xmm1
    vpsrldq xmm1, xmm0, 8
    vpaddq xmm0, xmm0, xmm1
    vmovq rbx, xmm0                     /* rbx = accumulated spin sum */

    cmp rax, r10
    jge .L_nd_store

.L_nd_tail_loop:
    mov r13, [rdi + rax*8]              /* x_pos[k] */
    mov r14, [rsi + rax*8]              /* x_neg[k] */
    mov r15, [rdx + rax*8]              /* w_pos[j, k] */
    mov rbp, [rcx + rax*8]              /* w_neg[j, k] */

    /* Constructive: (x_pos & w_pos) | (x_neg & w_neg) */
    mov r13, [rdi + rax*8]
    and r13, r15
    mov r14, [rsi + rax*8]
    and r14, rbp
    or  r13, r14
    popcnt r13, r13
    add rbx, r13

    /* Destructive: (x_pos & w_neg) | (x_neg & w_pos) */
    mov r13, [rdi + rax*8]
    and r13, rbp
    mov r14, [rsi + rax*8]
    and r14, r15
    or  r13, r14
    popcnt r13, r13
    sub rbx, r13

    inc rax
    cmp rax, r10
    jl .L_nd_tail_loop

.L_nd_store:
    mov [r8 + r12*4], ebx
    lea rdx, [rdx + r10*8]              /* Advance W_pos to next row */
    lea rcx, [rcx + r10*8]              /* Advance W_neg to next row */
    inc r12
    cmp r12, r9
    jl .L_nd_row_loop

.L_nd_done:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret


/* ====================================================================================
 * 11. asm_sparsity_gate_avx512(
 *         const float *in_f32   [rdi],
 *         float       *out_f32  [rsi],
 *         int64_t      n_floats [rdx],
 *         float        cutoff   [xmm0]
 *     ) -> rax (surviving_active_count)
 *     Pure AVX-512 Zero-Masked Dark-Channel Sparsity Gate (`vcmpps k1` + `vmovups {k1}{z}`)!
 * ==================================================================================== */
.globl asm_sparsity_gate_avx512
.type asm_sparsity_gate_avx512, @function
asm_sparsity_gate_avx512:
    vbroadcastss zmm30, xmm0            /* zmm30 = cutoff */
    mov eax, 0x7FFFFFFF
    vpbroadcastd zmm29, eax             /* zmm29 = abs bitmask (clear sign bit) */

    xor rcx, rcx
    xor rax, rax
    mov r8, rdx
    and r8, -16
    test r8, r8
    jle .L_sg_tail_check

.L_sg_vec_loop:
    vmovups zmm0, [rdi + rcx*4]
    vpandd  zmm1, zmm0, zmm29           /* zmm1 = |x| */
    vcmpps  k1, zmm1, zmm30, 13         /* k1 = (|x| >= cutoff) (13 = _CMP_GE_OS) */
    vmovups zmm2 {k1}{z}, zmm0          /* Zero-mask dark channels in 1 clock cycle! */
    vmovups [rsi + rcx*4], zmm2

    kmovw r9d, k1
    popcnt r9d, r9d
    add rax, r9

    add rcx, 16
    cmp rcx, r8
    jl .L_sg_vec_loop

.L_sg_tail_check:
    cmp rcx, rdx
    jge .L_sg_done
.L_sg_tail_loop:
    vmovss xmm1, dword ptr [rdi + rcx*4]
    vpandd xmm2, xmm1, xmm29
    vcomiss xmm2, xmm30
    jb .L_sg_zero
    vmovss dword ptr [rsi + rcx*4], xmm1
    inc rax
    jmp .L_sg_next
.L_sg_zero:
    mov dword ptr [rsi + rcx*4], 0
.L_sg_next:
    inc rcx
    cmp rcx, rdx
    jl .L_sg_tail_loop

.L_sg_done:
    vzeroupper
    ret


/* ====================================================================================
 * 12. asm_dot8192_vfma_avx512(const float *a [rdi], const float *b [rsi]) -> xmm0
 *     Computes 8192-element FP32 dot product (`Llama-3.3-70B` dim=8192 = 512x zmm)
 *     using 4-way unrolled AVX-512 FMA (`vfmadd231ps`).
 * ==================================================================================== */
.globl asm_dot8192_vfma_avx512
.type asm_dot8192_vfma_avx512, @function
asm_dot8192_vfma_avx512:
    vxorps zmm0, zmm0, zmm0
    vxorps zmm1, zmm1, zmm1
    vxorps zmm2, zmm2, zmm2
    vxorps zmm3, zmm3, zmm3
    xor rcx, rcx

.L_loop_8192:
    vmovups zmm4, [rdi + rcx]
    vfmadd231ps zmm0, zmm4, [rsi + rcx]

    vmovups zmm5, [rdi + rcx + 64]
    vfmadd231ps zmm1, zmm5, [rsi + rcx + 64]

    vmovups zmm6, [rdi + rcx + 128]
    vfmadd231ps zmm2, zmm6, [rsi + rcx + 128]

    vmovups zmm7, [rdi + rcx + 192]
    vfmadd231ps zmm3, zmm7, [rsi + rcx + 192]

    add rcx, 256
    cmp rcx, 32768                      /* 8192 floats * 4 bytes = 32768 bytes */
    jl .L_loop_8192

    vaddps zmm0, zmm0, zmm1
    vaddps zmm2, zmm2, zmm3
    vaddps zmm0, zmm0, zmm2

    vextractf32x8 ymm1, zmm0, 1
    vaddps ymm0, ymm0, ymm1
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vzeroupper
    ret

.section .note.GNU-stack,"",@progbits

