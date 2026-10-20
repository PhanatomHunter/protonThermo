	.file	"stage4_quadrillion_avx512.c"
	.intel_syntax noprefix
	.text
	.p2align 4
	.globl	route_1quadrillion_kronecker_avx512
	.type	route_1quadrillion_kronecker_avx512, @function
route_1quadrillion_kronecker_avx512:
.LFB6753:
	.cfi_startproc
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	r10, rdx
	xor	r11d, r11d
	mov	rdx, rsi
	vmovss	xmm2, DWORD PTR .LC0[rip]
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	push	r14
	push	r13
	push	r12
	push	rbx
	.cfi_offset 14, -24
	.cfi_offset 13, -32
	.cfi_offset 12, -40
	.cfi_offset 3, -48
	mov	rbx, r9
	xor	r9d, r9d
	.p2align 4
	.p2align 3
.L4:
	xor	eax, eax
	vxorps	xmm0, xmm0, xmm0
	.p2align 5
	.p2align 4
	.p2align 3
.L2:
	vmovups	zmm5, ZMMWORD PTR [rdi+rax]
	vfmadd231ps	zmm0, zmm5, ZMMWORD PTR [rdx+rax]
	add	rax, 64
	cmp	rax, 1152
	jne	.L2
	vextractf64x4	ymm1, zmm0, 0x1
	vaddps	ymm0, ymm1, ymm0
	vextractf32x4	xmm1, ymm0, 0x1
	vaddps	xmm1, xmm1, xmm0
	vpermilps	xmm0, xmm1, 78
	vaddps	xmm0, xmm0, xmm1
	vmovaps	xmm1, xmm0
	vshufps	xmm0, xmm0, xmm0, 85
	vaddss	xmm0, xmm1, xmm0
	vcomiss	xmm0, xmm2
	vmaxss	xmm2, xmm0, xmm2
	cmova	r9d, r11d
	add	r11d, 1
	add	rdx, 1152
	cmp	r11d, 100
	jne	.L4
	vmovss	xmm2, DWORD PTR .LC0[rip]
	mov	rdx, r10
	xor	r12d, r12d
	xor	r11d, r11d
	.p2align 4
	.p2align 3
.L7:
	xor	eax, eax
	vxorps	xmm0, xmm0, xmm0
	.p2align 5
	.p2align 4
	.p2align 3
.L5:
	vmovups	zmm6, ZMMWORD PTR [rdi+rax]
	vfmadd231ps	zmm0, zmm6, ZMMWORD PTR [rdx+rax]
	add	rax, 64
	cmp	rax, 1152
	jne	.L5
	vextractf64x4	ymm1, zmm0, 0x1
	vaddps	ymm0, ymm1, ymm0
	vextractf32x4	xmm1, ymm0, 0x1
	vaddps	xmm1, xmm1, xmm0
	vpermilps	xmm0, xmm1, 78
	vaddps	xmm0, xmm0, xmm1
	vmovaps	xmm1, xmm0
	vshufps	xmm0, xmm0, xmm0, 85
	vaddss	xmm0, xmm1, xmm0
	vcomiss	xmm0, xmm2
	vmaxss	xmm2, xmm0, xmm2
	cmova	r11d, r12d
	add	r12d, 1
	add	rdx, 1152
	cmp	r12d, 100
	jne	.L7
	vmovss	xmm2, DWORD PTR .LC0[rip]
	mov	rdx, rcx
	xor	r12d, r12d
	xor	r13d, r13d
	.p2align 4
	.p2align 3
.L10:
	xor	eax, eax
	vxorps	xmm0, xmm0, xmm0
	.p2align 5
	.p2align 4
	.p2align 3
.L8:
	vmovups	zmm7, ZMMWORD PTR [rdi+rax]
	vfmadd231ps	zmm0, zmm7, ZMMWORD PTR [rdx+rax]
	add	rax, 64
	cmp	rax, 1152
	jne	.L8
	vextractf64x4	ymm1, zmm0, 0x1
	vaddps	ymm0, ymm1, ymm0
	vextractf32x4	xmm1, ymm0, 0x1
	vaddps	xmm1, xmm1, xmm0
	vpermilps	xmm0, xmm1, 78
	vaddps	xmm0, xmm0, xmm1
	vmovaps	xmm1, xmm0
	vshufps	xmm0, xmm0, xmm0, 85
	vaddss	xmm0, xmm1, xmm0
	vcomiss	xmm0, xmm2
	vmaxss	xmm2, xmm0, xmm2
	cmova	r13d, r12d
	add	r12d, 1
	add	rdx, 1152
	cmp	r12d, 100
	jne	.L10
	vmovss	xmm2, DWORD PTR .LC0[rip]
	mov	rdx, r8
	xor	r12d, r12d
	xor	r14d, r14d
	.p2align 4
	.p2align 3
.L13:
	xor	eax, eax
	vxorps	xmm0, xmm0, xmm0
	.p2align 5
	.p2align 4
	.p2align 3
.L11:
	vmovups	zmm3, ZMMWORD PTR [rdi+rax]
	vfmadd231ps	zmm0, zmm3, ZMMWORD PTR [rdx+rax]
	add	rax, 64
	cmp	rax, 1152
	jne	.L11
	vextractf64x4	ymm1, zmm0, 0x1
	vaddps	ymm0, ymm1, ymm0
	vextractf32x4	xmm1, ymm0, 0x1
	vaddps	xmm1, xmm1, xmm0
	vpermilps	xmm0, xmm1, 78
	vaddps	xmm0, xmm0, xmm1
	vmovaps	xmm1, xmm0
	vshufps	xmm0, xmm0, xmm0, 85
	vaddss	xmm0, xmm1, xmm0
	vcomiss	xmm0, xmm2
	vmaxss	xmm2, xmm0, xmm2
	cmova	r14d, r12d
	add	r12d, 1
	add	rdx, 1152
	cmp	r12d, 66
	jne	.L13
	vmovd	xmm3, r13d
	vmovd	xmm4, r9d
	movsx	r9, r9d
	mov	rax, QWORD PTR 16[rbp]
	vpinsrd	xmm1, xmm3, r14d, 1
	movsx	r13, r13d
	movsx	r14, r14d
	vbroadcastss	zmm2, DWORD PTR .LC2[rip]
	vpinsrd	xmm0, xmm4, r11d, 1
	movsx	r11, r11d
	vpunpcklqdq	xmm0, xmm0, xmm1
	lea	rdx, [r11+r11*8]
	vmovdqu	XMMWORD PTR [rax], xmm0
	lea	rax, [r9+r9*8]
	sal	rdx, 7
	sal	rax, 7
	add	rdx, r10
	add	rsi, rax
	lea	rax, 0[r13+r13*8]
	sal	rax, 7
	add	rcx, rax
	lea	rax, [r14+r14*8]
	sal	rax, 7
	add	r8, rax
	xor	eax, eax
.L14:
	vmovups	zmm0, ZMMWORD PTR [r8+rax]
	vmovups	zmm1, ZMMWORD PTR [rdx+rax]
	vaddps	zmm0, zmm0, ZMMWORD PTR [rcx+rax]
	vaddps	zmm1, zmm1, ZMMWORD PTR [rsi+rax]
	vaddps	zmm0, zmm0, zmm1
	vmulps	zmm0, zmm0, zmm2
	vmovups	ZMMWORD PTR [rbx+rax], zmm0
	add	rax, 64
	cmp	rax, 1152
	jne	.L14
	imul	r9, r9, 660000
	mov	rdx, r13
	imul	r11, r11, 6600
	sal	rdx, 5
	add	rdx, r13
	lea	rax, [r9+r11]
	lea	rax, [rax+rdx*2]
	add	rax, r14
	vzeroupper
	pop	rbx
	pop	r12
	pop	r13
	pop	r14
	pop	rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6753:
	.size	route_1quadrillion_kronecker_avx512, .-route_1quadrillion_kronecker_avx512
	.p2align 4
	.globl	avx512_wdm64_wavefront_relax
	.type	avx512_wdm64_wavefront_relax, @function
avx512_wdm64_wavefront_relax:
.LFB6754:
	.cfi_startproc
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	push	r15
	push	r14
	push	r13
	push	r12
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	mov	r12, rdx
	push	rbx
	.cfi_offset 3, -56
	mov	rbx, rcx
	and	rsp, -64
	sub	rsp, 8
	mov	rax, QWORD PTR 16[rbp]
	mov	r10d, DWORD PTR 24[rbp]
	movsx	rdx, DWORD PTR 32[rbp]
	rdrand	rcx
	test	edx, edx
	jle	.L51
	sal	rdx, 6
	movsx	rcx, r10d
	mov	r15, r9
	mov	r13, rsi
	vmovq	xmm6, rcx
	lea	rcx, [rdi+rdx]
	mov	esi, r10d
	mov	QWORD PTR -24[rsp], r8
	mov	QWORD PTR -48[rsp], rcx
	lea	ecx, -16[r10]
	mov	edx, ecx
	and	ecx, -16
	shr	edx, 4
	lea	r9d, 16[rcx]
	xor	ecx, ecx
	cmp	r10d, 15
	cmovle	r9d, ecx
	add	edx, 1
	sal	rdx, 4
	sub	esi, r9d
	mov	QWORD PTR -56[rsp], rdx
	lea	ecx, -1[rsi]
	mov	edx, esi
	mov	DWORD PTR -72[rsp], esi
	and	edx, -8
	mov	DWORD PTR -32[rsp], ecx
	movsx	rcx, r9d
	mov	DWORD PTR -40[rsp], edx
	vmovq	xmm7, rcx
	mov	edx, esi
	lea	ecx, 1[r9]
	and	edx, 7
	vmovd	xmm8, ecx
	lea	ecx, 2[r9]
	mov	DWORD PTR -28[rsp], r9d
	mov	DWORD PTR -60[rsp], edx
	vmovd	xmm9, ecx
	lea	ecx, 3[r9]
	mov	DWORD PTR -64[rsp], edx
	lea	edx, -1[rdx]
	vmovd	xmm10, ecx
	lea	ecx, 4[r9]
	mov	DWORD PTR -68[rsp], edx
	vmovd	xmm11, ecx
	lea	ecx, 5[r9]
	mov	r9, r15
	vmovd	xmm12, ecx
	xor	ecx, ecx
	.p2align 4
	.p2align 3
.L41:
	vmovdqu64	zmm4, ZMMWORD PTR [rbx]
	vmovdqu64	zmm2, ZMMWORD PTR 0[r13]
	vmovdqu64	zmm5, ZMMWORD PTR [r12]
	vmovdqu64	zmm3, ZMMWORD PTR [rdi]
	vpandd	zmm0, zmm4, zmm2
	mov	rdx, QWORD PTR -24[rsp]
	vpandd	zmm2, zmm2, zmm5
	vpternlogd	zmm0, zmm3, zmm5, 248
	vpopcntq	zmm0, zmm0
	vpternlogd	zmm2, zmm3, zmm4, 248
	mov	r15d, DWORD PTR [rdx]
	vpopcntq	zmm2, zmm2
	vpsubq	zmm0, zmm0, zmm2
	vextracti64x4	ymm2, zmm0, 0x1
	vpaddq	ymm0, ymm0, ymm2
	vextracti64x2	xmm2, ymm0, 0x1
	vpaddq	xmm0, xmm0, xmm2
	vpbroadcastd	zmm2, r15d
	vpsrldq	xmm4, xmm0, 8
	vpaddq	xmm0, xmm4, xmm0
	vmovq	rdx, xmm0
	neg	edx
	vpbroadcastd	zmm0, edx
	cmp	r10d, 15
	jle	.L32
	mov	r14, QWORD PTR -56[rsp]
	lea	rsi, 0[0+rcx*4]
	lea	r8, [r14+rcx]
	sal	r8, 2
	.p2align 5
	.p2align 4
	.p2align 3
.L33:
	vmovdqu64	ZMMWORD PTR [r9+rsi], zmm2
	vmovdqu64	ZMMWORD PTR [rax+rsi], zmm0
	add	rsi, 64
	cmp	r8, rsi
	jne	.L33
.L32:
	mov	esi, DWORD PTR -28[rsp]
	cmp	r10d, esi
	jle	.L34
	cmp	DWORD PTR -32[rsp], 2
	mov	r14d, ecx
	jbe	.L35
	movsx	r8, ecx
	vmovq	r11, xmm7
	add	r11, r8
	mov	QWORD PTR -16[rsp], r11
	sal	r11, 2
	lea	rsi, [rax+r11]
	lea	r11, 4[r9+r11]
	mov	r8, rsi
	sub	r8, r11
	cmp	r8, 24
	jbe	.L35
	cmp	DWORD PTR -32[rsp], 6
	jbe	.L42
	mov	r11, QWORD PTR -16[rsp]
	vpbroadcastd	ymm0, r15d
	mov	r8d, DWORD PTR -40[rsp]
	vmovdqu	YMMWORD PTR [r9+r11*4], ymm0
	vpbroadcastd	ymm0, edx
	mov	r11d, r8d
	vmovdqu	YMMWORD PTR [rsi], ymm0
	mov	esi, DWORD PTR -28[rsp]
	add	esi, r8d
	mov	r8d, DWORD PTR -60[rsp]
	test	r8d, r8d
	je	.L34
	mov	r8d, DWORD PTR -64[rsp]
	cmp	DWORD PTR -68[rsp], 2
	mov	DWORD PTR -36[rsp], r8d
	jbe	.L38
.L36:
	mov	r8, QWORD PTR -16[rsp]
	vpbroadcastd	xmm0, r15d
	add	r11, r8
	vmovdqu	XMMWORD PTR [r9+r11*4], xmm0
	vpbroadcastd	xmm0, edx
	vmovdqu	XMMWORD PTR [rax+r11*4], xmm0
	mov	r11d, DWORD PTR -36[rsp]
	mov	r8d, r11d
	and	r8d, -4
	add	esi, r8d
	and	r11d, 3
	je	.L34
.L38:
	lea	r8d, [r14+rsi]
	movsx	r8, r8d
	mov	DWORD PTR [r9+r8*4], r15d
	mov	DWORD PTR [rax+r8*4], edx
	lea	r8d, 1[rsi]
	cmp	r10d, r8d
	jle	.L34
	add	r8d, r14d
	add	esi, 2
	movsx	r8, r8d
	mov	DWORD PTR [r9+r8*4], r15d
	mov	DWORD PTR [rax+r8*4], edx
	cmp	r10d, esi
	jle	.L34
	add	esi, r14d
	movsx	rsi, esi
	mov	DWORD PTR [r9+rsi*4], r15d
	mov	DWORD PTR [rax+rsi*4], edx
.L34:
	add	rdi, 64
	add	r13, 64
	add	r12, 64
	add	rbx, 64
	vmovq	rdx, xmm6
	add	QWORD PTR -24[rsp], 4
	add	rcx, rdx
	cmp	QWORD PTR -48[rsp], rdi
	jne	.L41
	vzeroupper
.L51:
	lea	rsp, -40[rbp]
	pop	rbx
	pop	r12
	pop	r13
	pop	r14
	pop	r15
	pop	rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L35:
	.cfi_restore_state
	mov	r11d, DWORD PTR -28[rsp]
	lea	esi, [r14+r11]
	movsx	rsi, esi
	mov	DWORD PTR [r9+rsi*4], r15d
	mov	DWORD PTR [rax+rsi*4], edx
	vmovd	esi, xmm8
	cmp	r10d, esi
	jle	.L34
	add	esi, r14d
	movsx	rsi, esi
	mov	DWORD PTR [r9+rsi*4], r15d
	mov	DWORD PTR [rax+rsi*4], edx
	vmovd	esi, xmm9
	cmp	r10d, esi
	jle	.L34
	add	esi, r14d
	movsx	rsi, esi
	mov	DWORD PTR [r9+rsi*4], r15d
	mov	DWORD PTR [rax+rsi*4], edx
	vmovd	esi, xmm10
	cmp	r10d, esi
	jle	.L34
	add	esi, r14d
	movsx	rsi, esi
	mov	DWORD PTR [r9+rsi*4], r15d
	mov	DWORD PTR [rax+rsi*4], edx
	vmovd	esi, xmm11
	cmp	r10d, esi
	jle	.L34
	add	esi, r14d
	movsx	rsi, esi
	mov	DWORD PTR [r9+rsi*4], r15d
	mov	DWORD PTR [rax+rsi*4], edx
	vmovd	esi, xmm12
	cmp	r10d, esi
	jle	.L34
	add	esi, r14d
	movsx	rsi, esi
	mov	DWORD PTR [r9+rsi*4], r15d
	mov	DWORD PTR [rax+rsi*4], edx
	lea	esi, 6[r11]
	cmp	r10d, esi
	jle	.L34
	add	esi, r14d
	movsx	rsi, esi
	mov	DWORD PTR [r9+rsi*4], r15d
	mov	DWORD PTR [rax+rsi*4], edx
	lea	esi, 7[r11]
	cmp	r10d, esi
	jle	.L34
	add	esi, r14d
	movsx	rsi, esi
	mov	DWORD PTR [r9+rsi*4], r15d
	mov	DWORD PTR [rax+rsi*4], edx
	lea	esi, 8[r11]
	cmp	r10d, esi
	jle	.L34
	add	esi, r14d
	movsx	rsi, esi
	mov	DWORD PTR [r9+rsi*4], r15d
	mov	DWORD PTR [rax+rsi*4], edx
	lea	esi, 9[r11]
	cmp	r10d, esi
	jle	.L34
	add	esi, r14d
	movsx	rsi, esi
	mov	DWORD PTR [r9+rsi*4], r15d
	mov	DWORD PTR [rax+rsi*4], edx
	lea	esi, 10[r11]
	cmp	r10d, esi
	jle	.L34
	add	esi, r14d
	movsx	rsi, esi
	mov	DWORD PTR [r9+rsi*4], r15d
	mov	DWORD PTR [rax+rsi*4], edx
	lea	esi, 11[r11]
	cmp	r10d, esi
	jle	.L34
	add	esi, r14d
	movsx	rsi, esi
	mov	DWORD PTR [r9+rsi*4], r15d
	mov	DWORD PTR [rax+rsi*4], edx
	lea	esi, 12[r11]
	cmp	r10d, esi
	jle	.L34
	add	esi, r14d
	movsx	rsi, esi
	mov	DWORD PTR [r9+rsi*4], r15d
	mov	DWORD PTR [rax+rsi*4], edx
	lea	esi, 13[r11]
	cmp	r10d, esi
	jle	.L34
	add	esi, r14d
	movsx	rsi, esi
	mov	DWORD PTR [r9+rsi*4], r15d
	mov	DWORD PTR [rax+rsi*4], edx
	lea	esi, 14[r11]
	cmp	r10d, esi
	jle	.L34
	add	r14d, esi
	movsx	r14, r14d
	mov	DWORD PTR [r9+r14*4], r15d
	mov	DWORD PTR [rax+r14*4], edx
	jmp	.L34
.L42:
	mov	r11d, DWORD PTR -72[rsp]
	mov	esi, DWORD PTR -28[rsp]
	mov	DWORD PTR -36[rsp], r11d
	xor	r11d, r11d
	jmp	.L36
	.cfi_endproc
.LFE6754:
	.size	avx512_wdm64_wavefront_relax, .-avx512_wdm64_wavefront_relax
	.p2align 4
	.globl	avx512_holographic_wavefront_f32
	.type	avx512_holographic_wavefront_f32, @function
avx512_holographic_wavefront_f32:
.LFB6755:
	.cfi_startproc
	movsx	rax, r8d
	test	eax, eax
	jle	.L69
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	r11, rdx
	mov	r10d, r9d
	mov	r8, rdi
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	push	r12
	.cfi_offset 12, -24
	lea	r12, [rcx+rax*4]
	push	rbx
	.cfi_offset 3, -32
	mov	rbx, rcx
	and	rsp, -64
	sub	rsp, 1032
	lea	rdx, -120[rsp]
	vbroadcastss	zmm4, DWORD PTR .LC4[rip]
	vbroadcastss	zmm3, DWORD PTR .LC6[rip]
.L60:
	xor	eax, eax
	.p2align 6
	.p2align 4
	.p2align 3
.L55:
	vmovaps	zmm0, zmm4
	vfmadd132ps	zmm0, zmm3, ZMMWORD PTR [rsi+rax]
	vmulps	zmm0, zmm0, ZMMWORD PTR [r8+rax]
	vmovaps	ZMMWORD PTR [rdx+rax], zmm0
	add	rax, 64
	cmp	rax, 1152
	jne	.L55
	test	r10d, r10d
	jle	.L61
	vmovss	xmm2, DWORD PTR .LC0[rip]
	mov	rcx, r11
	xor	edi, edi
	xor	r9d, r9d
	.p2align 4
	.p2align 3
.L59:
	xor	eax, eax
	vxorps	xmm0, xmm0, xmm0
	.p2align 5
	.p2align 4
	.p2align 3
.L57:
	vmovaps	zmm5, ZMMWORD PTR [rdx+rax]
	vfmadd231ps	zmm0, zmm5, ZMMWORD PTR [rcx+rax]
	add	rax, 64
	cmp	rax, 1152
	jne	.L57
	vextractf64x4	ymm1, zmm0, 0x1
	vaddps	ymm0, ymm1, ymm0
	vextractf32x4	xmm1, ymm0, 0x1
	vaddps	xmm1, xmm1, xmm0
	vpermilps	xmm0, xmm1, 78
	vaddps	xmm0, xmm0, xmm1
	vmovaps	xmm1, xmm0
	vshufps	xmm0, xmm0, xmm0, 85
	vaddss	xmm0, xmm1, xmm0
	vcomiss	xmm0, xmm2
	vmaxss	xmm2, xmm0, xmm2
	cmova	r9d, edi
	add	edi, 1
	add	rcx, 1152
	cmp	r10d, edi
	jne	.L59
.L56:
	mov	DWORD PTR [rbx], r9d
	add	rbx, 4
	add	r8, 1152
	cmp	r12, rbx
	jne	.L60
	vzeroupper
	lea	rsp, -16[rbp]
	pop	rbx
	pop	r12
	pop	rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L61:
	.cfi_restore_state
	xor	r9d, r9d
	jmp	.L56
.L69:
	.cfi_def_cfa 7, 8
	.cfi_restore 3
	.cfi_restore 6
	.cfi_restore 12
	ret
	.cfi_endproc
.LFE6755:
	.size	avx512_holographic_wavefront_f32, .-avx512_holographic_wavefront_f32
	.p2align 4
	.globl	route_14b_kronecker_768_avx512
	.type	route_14b_kronecker_768_avx512, @function
route_14b_kronecker_768_avx512:
.LFB6757:
	.cfi_startproc
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rax, rdi
	mov	r10, rcx
	mov	r11, r9
	vmovss	xmm2, DWORD PTR .LC0[rip]
	mov	rdi, rdx
	mov	rcx, rsi
	xor	r9d, r9d
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	push	r14
	push	r13
	push	r12
	push	rbx
	.cfi_offset 14, -24
	.cfi_offset 13, -32
	.cfi_offset 12, -40
	.cfi_offset 3, -48
	xor	ebx, ebx
	.p2align 4
	.p2align 3
.L75:
	vxorps	xmm1, xmm1, xmm1
	xor	edx, edx
	vmovaps	zmm0, zmm1
	.p2align 6
	.p2align 4
	.p2align 3
.L73:
	vmovups	zmm5, ZMMWORD PTR [rax+rdx]
	vmovups	zmm6, ZMMWORD PTR 64[rax+rdx]
	vfmadd231ps	zmm0, zmm5, ZMMWORD PTR [rcx+rdx]
	vfmadd231ps	zmm1, zmm6, ZMMWORD PTR 64[rdx+rcx]
	sub	rdx, -128
	cmp	rdx, 3072
	jne	.L73
	vaddps	zmm0, zmm0, zmm1
	vextractf64x4	ymm1, zmm0, 0x1
	vaddps	ymm0, ymm1, ymm0
	vextractf32x4	xmm1, ymm0, 0x1
	vaddps	xmm1, xmm1, xmm0
	vpermilps	xmm0, xmm1, 78
	vaddps	xmm0, xmm0, xmm1
	vmovaps	xmm1, xmm0
	vshufps	xmm0, xmm0, xmm0, 85
	vaddss	xmm0, xmm1, xmm0
	vcomiss	xmm0, xmm2
	vmaxss	xmm2, xmm0, xmm2
	cmova	r9d, ebx
	add	ebx, 1
	add	rcx, 3072
	cmp	ebx, 100
	jne	.L75
	vmovss	xmm2, DWORD PTR .LC0[rip]
	mov	rcx, rdi
	xor	r12d, r12d
	xor	ebx, ebx
	.p2align 4
	.p2align 3
.L78:
	vxorps	xmm1, xmm1, xmm1
	xor	edx, edx
	vmovaps	zmm0, zmm1
	.p2align 6
	.p2align 4
	.p2align 3
.L76:
	vmovups	zmm7, ZMMWORD PTR [rax+rdx]
	vmovups	zmm3, ZMMWORD PTR 64[rax+rdx]
	vfmadd231ps	zmm0, zmm7, ZMMWORD PTR [rcx+rdx]
	vfmadd231ps	zmm1, zmm3, ZMMWORD PTR 64[rdx+rcx]
	sub	rdx, -128
	cmp	rdx, 3072
	jne	.L76
	vaddps	zmm0, zmm0, zmm1
	vextractf64x4	ymm1, zmm0, 0x1
	vaddps	ymm0, ymm1, ymm0
	vextractf32x4	xmm1, ymm0, 0x1
	vaddps	xmm1, xmm1, xmm0
	vpermilps	xmm0, xmm1, 78
	vaddps	xmm0, xmm0, xmm1
	vmovaps	xmm1, xmm0
	vshufps	xmm0, xmm0, xmm0, 85
	vaddss	xmm0, xmm1, xmm0
	vcomiss	xmm0, xmm2
	vmaxss	xmm2, xmm0, xmm2
	cmova	ebx, r12d
	add	r12d, 1
	add	rcx, 3072
	cmp	r12d, 100
	jne	.L78
	vmovss	xmm2, DWORD PTR .LC0[rip]
	mov	rcx, r10
	xor	r12d, r12d
	xor	r13d, r13d
	.p2align 4
	.p2align 3
.L81:
	vxorps	xmm1, xmm1, xmm1
	xor	edx, edx
	vmovaps	zmm0, zmm1
	.p2align 6
	.p2align 4
	.p2align 3
.L79:
	vmovups	zmm4, ZMMWORD PTR [rax+rdx]
	vmovups	zmm5, ZMMWORD PTR 64[rax+rdx]
	vfmadd231ps	zmm0, zmm4, ZMMWORD PTR [rcx+rdx]
	vfmadd231ps	zmm1, zmm5, ZMMWORD PTR 64[rdx+rcx]
	sub	rdx, -128
	cmp	rdx, 3072
	jne	.L79
	vaddps	zmm0, zmm0, zmm1
	vextractf64x4	ymm1, zmm0, 0x1
	vaddps	ymm0, ymm1, ymm0
	vextractf32x4	xmm1, ymm0, 0x1
	vaddps	xmm1, xmm1, xmm0
	vpermilps	xmm0, xmm1, 78
	vaddps	xmm0, xmm0, xmm1
	vmovaps	xmm1, xmm0
	vshufps	xmm0, xmm0, xmm0, 85
	vaddss	xmm0, xmm1, xmm0
	vcomiss	xmm0, xmm2
	vmaxss	xmm2, xmm0, xmm2
	cmova	r13d, r12d
	add	r12d, 1
	add	rcx, 3072
	cmp	r12d, 100
	jne	.L81
	vmovss	xmm2, DWORD PTR .LC0[rip]
	mov	rcx, r8
	xor	r12d, r12d
	xor	r14d, r14d
	.p2align 4
	.p2align 3
.L84:
	vxorps	xmm1, xmm1, xmm1
	xor	edx, edx
	vmovaps	zmm0, zmm1
	.p2align 6
	.p2align 4
	.p2align 3
.L82:
	vmovups	zmm6, ZMMWORD PTR [rax+rdx]
	vmovups	zmm7, ZMMWORD PTR 64[rax+rdx]
	vfmadd231ps	zmm0, zmm6, ZMMWORD PTR [rcx+rdx]
	vfmadd231ps	zmm1, zmm7, ZMMWORD PTR 64[rdx+rcx]
	sub	rdx, -128
	cmp	rdx, 3072
	jne	.L82
	vaddps	zmm0, zmm0, zmm1
	vextractf64x4	ymm1, zmm0, 0x1
	vaddps	ymm0, ymm1, ymm0
	vextractf32x4	xmm1, ymm0, 0x1
	vaddps	xmm1, xmm1, xmm0
	vpermilps	xmm0, xmm1, 78
	vaddps	xmm0, xmm0, xmm1
	vmovaps	xmm1, xmm0
	vshufps	xmm0, xmm0, xmm0, 85
	vaddss	xmm0, xmm1, xmm0
	vcomiss	xmm0, xmm2
	vmaxss	xmm2, xmm0, xmm2
	cmova	r14d, r12d
	add	r12d, 1
	add	rcx, 3072
	cmp	r12d, 66
	jne	.L84
	vmovd	xmm3, r13d
	vmovd	xmm4, r9d
	movsx	r13, r13d
	mov	rax, QWORD PTR 16[rbp]
	vpinsrd	xmm1, xmm3, r14d, 1
	movsx	r9, r9d
	movsx	r14, r14d
	vbroadcastss	zmm2, DWORD PTR .LC2[rip]
	vpinsrd	xmm0, xmm4, ebx, 1
	movsx	rbx, ebx
	lea	rcx, [r9+r9*2]
	vpunpcklqdq	xmm0, xmm0, xmm1
	lea	rdx, [rbx+rbx*2]
	sal	rcx, 10
	vmovdqu	XMMWORD PTR [rax], xmm0
	lea	rax, 0[r13+r13*2]
	sal	rdx, 10
	add	rcx, rsi
	sal	rax, 10
	add	rdx, rdi
	add	r10, rax
	lea	rax, [r14+r14*2]
	sal	rax, 10
	add	r8, rax
	xor	eax, eax
	.p2align 6
	.p2align 4
	.p2align 3
.L85:
	vmovups	zmm0, ZMMWORD PTR [r8+rax]
	vmovups	zmm1, ZMMWORD PTR [rdx+rax]
	vaddps	zmm0, zmm0, ZMMWORD PTR [r10+rax]
	vaddps	zmm1, zmm1, ZMMWORD PTR [rcx+rax]
	vaddps	zmm0, zmm0, zmm1
	vmulps	zmm0, zmm0, zmm2
	vmovups	ZMMWORD PTR [r11+rax], zmm0
	add	rax, 64
	cmp	rax, 3072
	jne	.L85
	imul	rbx, rbx, 6600
	mov	rdx, r13
	imul	r9, r9, 660000
	sal	rdx, 5
	add	rdx, r13
	lea	rax, [r9+rbx]
	lea	rax, [rax+rdx*2]
	add	rax, r14
	vzeroupper
	pop	rbx
	pop	r12
	pop	r13
	pop	r14
	pop	rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6757:
	.size	route_14b_kronecker_768_avx512, .-route_14b_kronecker_768_avx512
	.p2align 4
	.globl	avx512_holographic_wavefront_768_f32
	.type	avx512_holographic_wavefront_768_f32, @function
avx512_holographic_wavefront_768_f32:
.LFB6758:
	.cfi_startproc
	movsx	rax, r8d
	test	eax, eax
	jle	.L116
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	r8, rdi
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	push	r15
	.cfi_offset 15, -24
	mov	r15, rdx
	mov	edx, r9d
	push	r14
	push	r13
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	lea	r13, [rcx+rax*4]
	push	r12
	.cfi_offset 12, -48
	mov	r12, rcx
	push	rbx
	and	rsp, -64
	sub	rsp, 2952
	.cfi_offset 3, -56
	lea	r9, -120[rsp]
	lea	rbx, -56[rsp]
	lea	rdi, 3016[rsp]
	vbroadcastss	zmm3, DWORD PTR .LC4[rip]
	vbroadcastss	zmm2, DWORD PTR .LC6[rip]
.L107:
	xor	eax, eax
	.p2align 6
	.p2align 4
	.p2align 3
.L102:
	vmovaps	zmm0, zmm3
	vfmadd132ps	zmm0, zmm2, ZMMWORD PTR [rsi+rax]
	vmulps	zmm0, zmm0, ZMMWORD PTR [r8+rax]
	vmovaps	ZMMWORD PTR [r9+rax], zmm0
	add	rax, 64
	cmp	rax, 3072
	jne	.L102
	test	edx, edx
	jle	.L108
	vmovss	xmm4, DWORD PTR .LC0[rip]
	lea	r11, 64[r15]
	xor	r10d, r10d
	xor	r14d, r14d
	.p2align 4
	.p2align 3
.L106:
	vxorps	xmm1, xmm1, xmm1
	mov	rax, rbx
	mov	rcx, r11
	vmovaps	zmm0, zmm1
	.p2align 6
	.p2align 4
	.p2align 3
.L104:
	vmovaps	zmm5, ZMMWORD PTR -64[rax]
	vmovaps	zmm6, ZMMWORD PTR [rax]
	sub	rax, -128
	sub	rcx, -128
	vfmadd231ps	zmm0, zmm5, ZMMWORD PTR -192[rcx]
	vfmadd231ps	zmm1, zmm6, ZMMWORD PTR -128[rcx]
	cmp	rax, rdi
	jne	.L104
	vaddps	zmm0, zmm0, zmm1
	vextractf64x4	ymm1, zmm0, 0x1
	vaddps	ymm0, ymm1, ymm0
	vextractf32x4	xmm1, ymm0, 0x1
	vaddps	xmm1, xmm1, xmm0
	vpermilps	xmm0, xmm1, 78
	vaddps	xmm0, xmm0, xmm1
	vmovaps	xmm1, xmm0
	vshufps	xmm0, xmm0, xmm0, 85
	vaddss	xmm0, xmm1, xmm0
	vcomiss	xmm0, xmm4
	vmaxss	xmm4, xmm0, xmm4
	cmova	r14d, r10d
	add	r10d, 1
	add	r11, 3072
	cmp	edx, r10d
	jne	.L106
.L103:
	mov	DWORD PTR [r12], r14d
	add	r12, 4
	add	r8, 3072
	cmp	r13, r12
	jne	.L107
	vzeroupper
	lea	rsp, -40[rbp]
	pop	rbx
	pop	r12
	pop	r13
	pop	r14
	pop	r15
	pop	rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L108:
	.cfi_restore_state
	xor	r14d, r14d
	jmp	.L103
.L116:
	.cfi_def_cfa 7, 8
	.cfi_restore 3
	.cfi_restore 6
	.cfi_restore 12
	.cfi_restore 13
	.cfi_restore 14
	.cfi_restore 15
	ret
	.cfi_endproc
.LFE6758:
	.size	avx512_holographic_wavefront_768_f32, .-avx512_holographic_wavefront_768_f32
	.p2align 4
	.globl	route_deepseek_5120_kronecker_avx512
	.type	route_deepseek_5120_kronecker_avx512, @function
route_deepseek_5120_kronecker_avx512:
.LFB6760:
	.cfi_startproc
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	r11, r9
	xor	r9d, r9d
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	push	r15
	push	r14
	push	r13
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	mov	r13, rsi
	push	r12
	.cfi_offset 12, -48
	xor	r12d, r12d
	push	rbx
	.cfi_offset 3, -56
	lea	rbx, 20480[rsi]
	and	rsp, -64
	mov	QWORD PTR -8[rsp], rdx
	vmovss	xmm4, DWORD PTR .LC0[rip]
	.p2align 4
	.p2align 3
.L122:
	vxorps	xmm2, xmm2, xmm2
	mov	rdx, rdi
	mov	rax, r13
	vmovaps	zmm1, zmm2
	vmovaps	zmm3, zmm2
	vmovaps	zmm0, zmm2
	.p2align 4
	.p2align 3
.L120:
	vmovups	zmm6, ZMMWORD PTR [rdx]
	vmovups	zmm7, ZMMWORD PTR 64[rdx]
	add	rax, 256
	add	rdx, 256
	vfmadd231ps	zmm0, zmm6, ZMMWORD PTR -256[rax]
	vmovups	zmm5, ZMMWORD PTR -128[rdx]
	vmovups	zmm6, ZMMWORD PTR -64[rdx]
	vfmadd231ps	zmm1, zmm5, ZMMWORD PTR -128[rax]
	vfmadd231ps	zmm3, zmm7, ZMMWORD PTR -192[rax]
	vfmadd231ps	zmm2, zmm6, ZMMWORD PTR -64[rax]
	cmp	rbx, rax
	jne	.L120
	vaddps	zmm1, zmm1, zmm2
	vaddps	zmm0, zmm0, zmm3
	vaddps	zmm0, zmm0, zmm1
	vextractf64x4	ymm1, zmm0, 0x1
	vaddps	ymm0, ymm1, ymm0
	vextractf32x4	xmm1, ymm0, 0x1
	vaddps	xmm1, xmm1, xmm0
	vpermilps	xmm0, xmm1, 78
	vaddps	xmm0, xmm0, xmm1
	vmovaps	xmm1, xmm0
	vshufps	xmm0, xmm0, xmm0, 85
	vaddss	xmm0, xmm1, xmm0
	vcomiss	xmm0, xmm4
	vmaxss	xmm4, xmm0, xmm4
	cmova	r9d, r12d
	add	r12d, 1
	add	r13, 20480
	add	rbx, 20480
	cmp	r12d, 100
	jne	.L122
	mov	rax, QWORD PTR -8[rsp]
	xor	r13d, r13d
	xor	ebx, ebx
	vmovss	xmm4, DWORD PTR .LC0[rip]
	mov	r14, rax
	lea	r12, 20480[rax]
	.p2align 4
	.p2align 3
.L125:
	vxorps	xmm2, xmm2, xmm2
	mov	rdx, rdi
	mov	rax, r14
	vmovaps	zmm1, zmm2
	vmovaps	zmm3, zmm2
	vmovaps	zmm0, zmm2
	.p2align 4
	.p2align 3
.L123:
	vmovups	zmm7, ZMMWORD PTR [rdx]
	vmovups	zmm5, ZMMWORD PTR 64[rdx]
	add	rax, 256
	add	rdx, 256
	vfmadd231ps	zmm0, zmm7, ZMMWORD PTR -256[rax]
	vmovups	zmm6, ZMMWORD PTR -128[rdx]
	vmovups	zmm7, ZMMWORD PTR -64[rdx]
	vfmadd231ps	zmm1, zmm6, ZMMWORD PTR -128[rax]
	vfmadd231ps	zmm3, zmm5, ZMMWORD PTR -192[rax]
	vfmadd231ps	zmm2, zmm7, ZMMWORD PTR -64[rax]
	cmp	r12, rax
	jne	.L123
	vaddps	zmm1, zmm1, zmm2
	vaddps	zmm0, zmm0, zmm3
	vaddps	zmm0, zmm0, zmm1
	vextractf64x4	ymm1, zmm0, 0x1
	vaddps	ymm0, ymm1, ymm0
	vextractf32x4	xmm1, ymm0, 0x1
	vaddps	xmm1, xmm1, xmm0
	vpermilps	xmm0, xmm1, 78
	vaddps	xmm0, xmm0, xmm1
	vmovaps	xmm1, xmm0
	vshufps	xmm0, xmm0, xmm0, 85
	vaddss	xmm0, xmm1, xmm0
	vcomiss	xmm0, xmm4
	vmaxss	xmm4, xmm0, xmm4
	cmova	ebx, r13d
	add	r13d, 1
	add	r14, 20480
	add	r12, 20480
	cmp	r13d, 100
	jne	.L125
	mov	r14, rcx
	lea	r12, 20480[rcx]
	xor	r13d, r13d
	xor	r10d, r10d
	vmovss	xmm5, DWORD PTR .LC0[rip]
	.p2align 4
	.p2align 3
.L128:
	vxorps	xmm2, xmm2, xmm2
	mov	rdx, rdi
	mov	rax, r14
	vmovaps	zmm1, zmm2
	vmovaps	zmm3, zmm2
	vmovaps	zmm0, zmm2
	.p2align 4
	.p2align 3
.L126:
	vmovups	zmm4, ZMMWORD PTR [rdx]
	add	rax, 256
	vmovups	zmm6, ZMMWORD PTR 192[rdx]
	add	rdx, 256
	vfmadd231ps	zmm0, zmm4, ZMMWORD PTR -256[rax]
	vfmadd231ps	zmm2, zmm6, ZMMWORD PTR -64[rax]
	vmovups	zmm4, ZMMWORD PTR -192[rdx]
	vfmadd231ps	zmm3, zmm4, ZMMWORD PTR -192[rax]
	vmovups	zmm4, ZMMWORD PTR -128[rdx]
	vfmadd231ps	zmm1, zmm4, ZMMWORD PTR -128[rax]
	cmp	r12, rax
	jne	.L126
	vaddps	zmm1, zmm1, zmm2
	vaddps	zmm0, zmm0, zmm3
	vaddps	zmm0, zmm0, zmm1
	vextractf64x4	ymm1, zmm0, 0x1
	vaddps	ymm0, ymm1, ymm0
	vextractf32x4	xmm1, ymm0, 0x1
	vaddps	xmm1, xmm1, xmm0
	vpermilps	xmm0, xmm1, 78
	vaddps	xmm0, xmm0, xmm1
	vmovaps	xmm1, xmm0
	vshufps	xmm0, xmm0, xmm0, 85
	vaddss	xmm0, xmm1, xmm0
	vcomiss	xmm0, xmm5
	vmaxss	xmm5, xmm0, xmm5
	cmova	r10d, r13d
	add	r13d, 1
	add	r14, 20480
	add	r12, 20480
	cmp	r13d, 100
	jne	.L128
	mov	r14, r8
	lea	r12, 20480[r8]
	xor	r13d, r13d
	xor	r15d, r15d
	vmovss	xmm5, DWORD PTR .LC0[rip]
.L131:
	vxorps	xmm2, xmm2, xmm2
	mov	rdx, rdi
	mov	rax, r14
	vmovaps	zmm1, zmm2
	vmovaps	zmm3, zmm2
	vmovaps	zmm0, zmm2
	.p2align 4
	.p2align 3
.L129:
	vmovups	zmm7, ZMMWORD PTR [rdx]
	vmovups	zmm4, ZMMWORD PTR 64[rdx]
	add	rax, 256
	add	rdx, 256
	vfmadd231ps	zmm0, zmm7, ZMMWORD PTR -256[rax]
	vfmadd231ps	zmm3, zmm4, ZMMWORD PTR -192[rax]
	vmovups	zmm7, ZMMWORD PTR -128[rdx]
	vmovups	zmm4, ZMMWORD PTR -64[rdx]
	vfmadd231ps	zmm1, zmm7, ZMMWORD PTR -128[rax]
	vfmadd231ps	zmm2, zmm4, ZMMWORD PTR -64[rax]
	cmp	r12, rax
	jne	.L129
	vaddps	zmm1, zmm1, zmm2
	vaddps	zmm0, zmm0, zmm3
	vaddps	zmm0, zmm0, zmm1
	vextractf64x4	ymm1, zmm0, 0x1
	vaddps	ymm0, ymm1, ymm0
	vextractf32x4	xmm1, ymm0, 0x1
	vaddps	xmm1, xmm1, xmm0
	vpermilps	xmm0, xmm1, 78
	vaddps	xmm0, xmm0, xmm1
	vmovaps	xmm1, xmm0
	vshufps	xmm0, xmm0, xmm0, 85
	vaddss	xmm0, xmm1, xmm0
	vcomiss	xmm0, xmm5
	vmaxss	xmm5, xmm0, xmm5
	cmova	r15d, r13d
	add	r13d, 1
	add	r14, 20480
	add	r12, 20480
	cmp	r13d, 68
	jne	.L131
	vmovd	xmm7, r9d
	vmovd	xmm6, r10d
	mov	rax, QWORD PTR 16[rbp]
	movsx	r9, r9d
	vpinsrd	xmm1, xmm6, r15d, 1
	movsx	rdi, r10d
	movsx	r15, r15d
	vbroadcastss	zmm2, DWORD PTR .LC2[rip]
	vpinsrd	xmm0, xmm7, ebx, 1
	movsx	rbx, ebx
	vpunpcklqdq	xmm0, xmm0, xmm1
	lea	rdx, [rbx+rbx*4]
	vmovdqu	XMMWORD PTR [rax], xmm0
	lea	rax, [r9+r9*4]
	sal	rdx, 12
	sal	rax, 12
	add	rsi, rax
	mov	rax, QWORD PTR -8[rsp]
	add	rax, rdx
	mov	rdx, rax
	lea	rax, [rdi+rdi*4]
	sal	rax, 12
	add	rcx, rax
	lea	rax, [r15+r15*4]
	sal	rax, 12
	add	r8, rax
	xor	eax, eax
	.p2align 6
	.p2align 4
	.p2align 3
.L132:
	vmovups	zmm0, ZMMWORD PTR [r8+rax]
	vmovups	zmm1, ZMMWORD PTR [rdx+rax]
	vaddps	zmm0, zmm0, ZMMWORD PTR [rcx+rax]
	vaddps	zmm1, zmm1, ZMMWORD PTR [rsi+rax]
	vaddps	zmm0, zmm0, zmm1
	vmulps	zmm0, zmm0, zmm2
	vmovups	ZMMWORD PTR [r11+rax], zmm0
	add	rax, 64
	cmp	rax, 20480
	jne	.L132
	imul	rbx, rbx, 6800
	mov	rdx, rdi
	imul	r9, r9, 680000
	sal	rdx, 4
	add	rdx, rdi
	lea	rax, [r9+rbx]
	lea	rax, [rax+rdx*4]
	add	rax, r15
	vzeroupper
	lea	rsp, -40[rbp]
	pop	rbx
	pop	r12
	pop	r13
	pop	r14
	pop	r15
	pop	rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6760:
	.size	route_deepseek_5120_kronecker_avx512, .-route_deepseek_5120_kronecker_avx512
	.p2align 4
	.globl	avx512_deepseek_5120_wavefront_f32
	.type	avx512_deepseek_5120_wavefront_f32, @function
avx512_deepseek_5120_wavefront_f32:
.LFB6761:
	.cfi_startproc
	movsx	rax, r8d
	test	eax, eax
	jle	.L163
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	r8, rdi
	mov	r11, rdx
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	push	r14
	push	r13
	.cfi_offset 14, -24
	.cfi_offset 13, -32
	lea	r13, [rcx+rax*4]
	push	r12
	.cfi_offset 12, -40
	mov	r12, rcx
	push	rbx
	.cfi_offset 3, -48
	mov	ebx, r9d
	and	rsp, -64
	sub	rsp, 20360
	lea	rdi, -120[rsp]
	lea	rcx, 20360[rsp]
	vbroadcastss	zmm5, DWORD PTR .LC4[rip]
	vbroadcastss	zmm4, DWORD PTR .LC6[rip]
.L154:
	xor	eax, eax
	.p2align 6
	.p2align 4
	.p2align 3
.L149:
	vmovaps	zmm0, zmm5
	vfmadd132ps	zmm0, zmm4, ZMMWORD PTR [rsi+rax]
	vmulps	zmm0, zmm0, ZMMWORD PTR [r8+rax]
	add	rax, 64
	vmovaps	ZMMWORD PTR -64[rax+rdi], zmm0
	cmp	rax, 20480
	jne	.L149
	test	ebx, ebx
	jle	.L155
	vmovss	xmm6, DWORD PTR .LC0[rip]
	mov	r10, r11
	xor	r9d, r9d
	xor	r14d, r14d
	.p2align 4
	.p2align 3
.L153:
	vxorps	xmm2, xmm2, xmm2
	mov	rax, rdi
	mov	rdx, r10
	vmovaps	zmm1, zmm2
	vmovaps	zmm3, zmm2
	vmovaps	zmm0, zmm2
	.p2align 4
	.p2align 3
.L151:
	vmovaps	zmm7, ZMMWORD PTR [rax]
	add	rax, 256
	vfmadd231ps	zmm0, zmm7, ZMMWORD PTR [rdx]
	add	rdx, 256
	vmovaps	zmm7, ZMMWORD PTR -192[rax]
	vfmadd231ps	zmm3, zmm7, ZMMWORD PTR -192[rdx]
	vmovaps	zmm7, ZMMWORD PTR -128[rax]
	vfmadd231ps	zmm1, zmm7, ZMMWORD PTR -128[rdx]
	vmovaps	zmm7, ZMMWORD PTR -64[rax]
	vfmadd231ps	zmm2, zmm7, ZMMWORD PTR -64[rdx]
	cmp	rax, rcx
	jne	.L151
	vaddps	zmm1, zmm1, zmm2
	vaddps	zmm0, zmm0, zmm3
	vaddps	zmm0, zmm0, zmm1
	vextractf64x4	ymm1, zmm0, 0x1
	vaddps	ymm0, ymm1, ymm0
	vextractf32x4	xmm1, ymm0, 0x1
	vaddps	xmm1, xmm1, xmm0
	vpermilps	xmm0, xmm1, 78
	vaddps	xmm0, xmm0, xmm1
	vmovaps	xmm1, xmm0
	vshufps	xmm0, xmm0, xmm0, 85
	vaddss	xmm0, xmm1, xmm0
	vcomiss	xmm0, xmm6
	vmaxss	xmm6, xmm0, xmm6
	cmova	r14d, r9d
	add	r9d, 1
	add	r10, 20480
	cmp	ebx, r9d
	jne	.L153
.L150:
	mov	DWORD PTR [r12], r14d
	add	r12, 4
	add	r8, 20480
	cmp	r13, r12
	jne	.L154
	vzeroupper
	lea	rsp, -32[rbp]
	pop	rbx
	pop	r12
	pop	r13
	pop	r14
	pop	rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L155:
	.cfi_restore_state
	xor	r14d, r14d
	jmp	.L150
.L163:
	.cfi_def_cfa 7, 8
	.cfi_restore 3
	.cfi_restore 6
	.cfi_restore 12
	.cfi_restore 13
	.cfi_restore 14
	ret
	.cfi_endproc
.LFE6761:
	.size	avx512_deepseek_5120_wavefront_f32, .-avx512_deepseek_5120_wavefront_f32
	.p2align 4
	.globl	avx512_kronecker_route_8192_f32
	.type	avx512_kronecker_route_8192_f32, @function
avx512_kronecker_route_8192_f32:
.LFB6764:
	.cfi_startproc
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rax, rsi
	mov	r11, r9
	mov	r10, rdi
	xor	r9d, r9d
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	push	r15
	push	r14
	push	r13
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	mov	r13, rax
	push	r12
	.cfi_offset 12, -48
	xor	r12d, r12d
	push	rbx
	.cfi_offset 3, -56
	lea	rbx, 8192[rax]
	and	rsp, -64
	mov	QWORD PTR -8[rsp], rsi
	mov	rsi, rdx
	vmovss	xmm4, DWORD PTR .LC0[rip]
	.p2align 4
	.p2align 3
.L169:
	vxorps	xmm3, xmm3, xmm3
	mov	rdx, r10
	mov	rax, r13
	vmovaps	zmm0, zmm3
	vmovaps	zmm2, zmm3
	vmovaps	zmm1, zmm3
	.p2align 4
	.p2align 3
.L167:
	vmovups	zmm6, ZMMWORD PTR [rdx]
	vmovups	zmm7, ZMMWORD PTR 64[rdx]
	add	rax, 256
	add	rdx, 256
	vfmadd231ps	zmm1, zmm6, ZMMWORD PTR -256[rax]
	vmovups	zmm5, ZMMWORD PTR -128[rdx]
	vmovups	zmm6, ZMMWORD PTR -64[rdx]
	vfmadd231ps	zmm0, zmm5, ZMMWORD PTR -128[rax]
	vfmadd231ps	zmm2, zmm7, ZMMWORD PTR -192[rax]
	vfmadd231ps	zmm3, zmm6, ZMMWORD PTR -64[rax]
	cmp	rbx, rax
	jne	.L167
	vaddps	zmm1, zmm1, zmm2
	vaddps	zmm0, zmm0, zmm3
	vaddps	zmm0, zmm0, zmm1
	vextractf64x4	ymm1, zmm0, 0x1
	vaddps	ymm0, ymm1, ymm0
	vextractf32x4	xmm1, ymm0, 0x1
	vaddps	xmm1, xmm1, xmm0
	vpermilps	xmm0, xmm1, 78
	vaddps	xmm0, xmm0, xmm1
	vmovaps	xmm1, xmm0
	vshufps	xmm0, xmm0, xmm0, 85
	vaddss	xmm0, xmm1, xmm0
	vcomiss	xmm0, xmm4
	vmaxss	xmm4, xmm0, xmm4
	cmova	r9d, r12d
	add	r12d, 1
	add	r13, 32768
	add	rbx, 32768
	cmp	r12d, 100
	jne	.L169
	vmovss	xmm4, DWORD PTR .LC0[rip]
	lea	r14, 8192[rsi]
	xor	r13d, r13d
	xor	ebx, ebx
	lea	r12, 16384[rsi]
	lea	r15, 8192[r10]
	.p2align 4
	.p2align 3
.L172:
	vxorps	xmm3, xmm3, xmm3
	mov	rdx, r15
	mov	rax, r14
	vmovaps	zmm0, zmm3
	vmovaps	zmm2, zmm3
	vmovaps	zmm1, zmm3
	.p2align 4
	.p2align 3
.L170:
	vmovups	zmm7, ZMMWORD PTR [rdx]
	vmovups	zmm5, ZMMWORD PTR 64[rdx]
	add	rax, 256
	add	rdx, 256
	vfmadd231ps	zmm1, zmm7, ZMMWORD PTR -256[rax]
	vmovups	zmm6, ZMMWORD PTR -128[rdx]
	vmovups	zmm7, ZMMWORD PTR -64[rdx]
	vfmadd231ps	zmm0, zmm6, ZMMWORD PTR -128[rax]
	vfmadd231ps	zmm2, zmm5, ZMMWORD PTR -192[rax]
	vfmadd231ps	zmm3, zmm7, ZMMWORD PTR -64[rax]
	cmp	r12, rax
	jne	.L170
	vaddps	zmm1, zmm1, zmm2
	vaddps	zmm0, zmm0, zmm3
	vaddps	zmm0, zmm0, zmm1
	vextractf64x4	ymm1, zmm0, 0x1
	vaddps	ymm0, ymm1, ymm0
	vextractf32x4	xmm1, ymm0, 0x1
	vaddps	xmm1, xmm1, xmm0
	vpermilps	xmm0, xmm1, 78
	vaddps	xmm0, xmm0, xmm1
	vmovaps	xmm1, xmm0
	vshufps	xmm0, xmm0, xmm0, 85
	vaddss	xmm0, xmm1, xmm0
	vcomiss	xmm0, xmm4
	vmaxss	xmm4, xmm0, xmm4
	cmova	ebx, r13d
	add	r13d, 1
	add	r14, 32768
	add	r12, 32768
	cmp	r13d, 100
	jne	.L172
	vmovss	xmm5, DWORD PTR .LC0[rip]
	lea	r14, 16384[rcx]
	xor	r13d, r13d
	xor	edi, edi
	lea	r12, 24576[rcx]
	lea	r15, 16384[r10]
	.p2align 4
	.p2align 3
.L175:
	vxorps	xmm3, xmm3, xmm3
	mov	rdx, r15
	mov	rax, r14
	vmovaps	zmm0, zmm3
	vmovaps	zmm2, zmm3
	vmovaps	zmm1, zmm3
	.p2align 4
	.p2align 3
.L173:
	vmovups	zmm4, ZMMWORD PTR [rdx]
	add	rax, 256
	vmovups	zmm6, ZMMWORD PTR 192[rdx]
	add	rdx, 256
	vfmadd231ps	zmm1, zmm4, ZMMWORD PTR -256[rax]
	vfmadd231ps	zmm3, zmm6, ZMMWORD PTR -64[rax]
	vmovups	zmm4, ZMMWORD PTR -192[rdx]
	vfmadd231ps	zmm2, zmm4, ZMMWORD PTR -192[rax]
	vmovups	zmm4, ZMMWORD PTR -128[rdx]
	vfmadd231ps	zmm0, zmm4, ZMMWORD PTR -128[rax]
	cmp	r12, rax
	jne	.L173
	vaddps	zmm1, zmm1, zmm2
	vaddps	zmm0, zmm0, zmm3
	vaddps	zmm0, zmm0, zmm1
	vextractf64x4	ymm1, zmm0, 0x1
	vaddps	ymm0, ymm1, ymm0
	vextractf32x4	xmm1, ymm0, 0x1
	vaddps	xmm1, xmm1, xmm0
	vpermilps	xmm0, xmm1, 78
	vaddps	xmm0, xmm0, xmm1
	vmovaps	xmm1, xmm0
	vshufps	xmm0, xmm0, xmm0, 85
	vaddss	xmm0, xmm1, xmm0
	vcomiss	xmm0, xmm5
	vmaxss	xmm5, xmm0, xmm5
	cmova	edi, r13d
	add	r13d, 1
	add	r14, 32768
	add	r12, 32768
	cmp	r13d, 100
	jne	.L175
	vmovss	xmm5, DWORD PTR .LC0[rip]
	lea	r14, 24576[r8]
	xor	r13d, r13d
	xor	r15d, r15d
	lea	r12, 32768[r8]
	add	r10, 24576
	.p2align 4
	.p2align 3
.L178:
	vxorps	xmm3, xmm3, xmm3
	mov	rdx, r10
	mov	rax, r14
	vmovaps	zmm0, zmm3
	vmovaps	zmm2, zmm3
	vmovaps	zmm1, zmm3
	.p2align 4
	.p2align 3
.L176:
	vmovups	zmm7, ZMMWORD PTR [rdx]
	vmovups	zmm4, ZMMWORD PTR 64[rdx]
	add	rax, 256
	add	rdx, 256
	vfmadd231ps	zmm1, zmm7, ZMMWORD PTR -256[rax]
	vfmadd231ps	zmm2, zmm4, ZMMWORD PTR -192[rax]
	vmovups	zmm7, ZMMWORD PTR -128[rdx]
	vmovups	zmm4, ZMMWORD PTR -64[rdx]
	vfmadd231ps	zmm0, zmm7, ZMMWORD PTR -128[rax]
	vfmadd231ps	zmm3, zmm4, ZMMWORD PTR -64[rax]
	cmp	r12, rax
	jne	.L176
	vaddps	zmm1, zmm1, zmm2
	vaddps	zmm0, zmm0, zmm3
	vaddps	zmm0, zmm0, zmm1
	vextractf64x4	ymm1, zmm0, 0x1
	vaddps	ymm0, ymm1, ymm0
	vextractf32x4	xmm1, ymm0, 0x1
	vaddps	xmm1, xmm1, xmm0
	vpermilps	xmm0, xmm1, 78
	vaddps	xmm0, xmm0, xmm1
	vmovaps	xmm1, xmm0
	vshufps	xmm0, xmm0, xmm0, 85
	vaddss	xmm0, xmm1, xmm0
	vcomiss	xmm0, xmm5
	vmaxss	xmm5, xmm0, xmm5
	cmova	r15d, r13d
	add	r13d, 1
	add	r14, 32768
	add	r12, 32768
	cmp	r13d, 68
	jne	.L178
	vmovd	xmm7, r9d
	vmovd	xmm6, edi
	mov	rax, QWORD PTR 16[rbp]
	movsx	r9, r9d
	vpinsrd	xmm1, xmm6, r15d, 1
	vpinsrd	xmm0, xmm7, ebx, 1
	mov	rdx, r9
	movsx	rbx, ebx
	vpunpcklqdq	xmm0, xmm0, xmm1
	sal	rdx, 15
	movsx	r10, edi
	movsx	r15, r15d
	vbroadcastss	zmm2, DWORD PTR .LC2[rip]
	vmovdqu	XMMWORD PTR [rax], xmm0
	mov	rax, QWORD PTR -8[rsp]
	add	rax, rdx
	mov	rdx, rax
	mov	rax, rbx
	sal	rax, 15
	add	rsi, rax
	mov	rax, r10
	sal	rax, 15
	add	rcx, rax
	mov	rax, r15
	sal	rax, 15
	add	r8, rax
	xor	eax, eax
	.p2align 6
	.p2align 4
	.p2align 3
.L179:
	vmovups	zmm0, ZMMWORD PTR [r8+rax]
	vmovups	zmm1, ZMMWORD PTR [rsi+rax]
	vaddps	zmm0, zmm0, ZMMWORD PTR [rcx+rax]
	vaddps	zmm1, zmm1, ZMMWORD PTR [rdx+rax]
	vaddps	zmm0, zmm0, zmm1
	vmulps	zmm0, zmm0, zmm2
	vmovups	ZMMWORD PTR [r11+rax], zmm0
	add	rax, 64
	cmp	rax, 32768
	jne	.L179
	imul	rbx, rbx, 6800
	mov	rdx, r10
	imul	r9, r9, 680000
	sal	rdx, 4
	add	rdx, r10
	lea	rax, [r9+rbx]
	lea	rax, [rax+rdx*4]
	add	rax, r15
	vzeroupper
	lea	rsp, -40[rbp]
	pop	rbx
	pop	r12
	pop	r13
	pop	r14
	pop	r15
	pop	rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6764:
	.size	avx512_kronecker_route_8192_f32, .-avx512_kronecker_route_8192_f32
	.p2align 4
	.globl	avx512_llama70b_8192_wavefront_f32
	.type	avx512_llama70b_8192_wavefront_f32, @function
avx512_llama70b_8192_wavefront_f32:
.LFB6765:
	.cfi_startproc
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	push	r15
	push	r14
	push	r13
	push	r12
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	movsx	r12, r8d
	push	rbx
	and	rsp, -64
	sub	rsp, 32648
	.cfi_offset 3, -56
	test	r12d, r12d
	jle	.L208
	mov	r15, rcx
	mov	rbx, rdx
	mov	ecx, r9d
	xor	r13d, r13d
	vbroadcastss	zmm5, DWORD PTR .LC4[rip]
	mov	r9, rdi
	lea	r8, -120[rsp]
	vbroadcastss	zmm4, DWORD PTR .LC6[rip]
	lea	rdi, 32648[rsp]
.L201:
	xor	eax, eax
	.p2align 6
	.p2align 4
	.p2align 3
.L196:
	vmovaps	zmm0, zmm5
	vfmadd132ps	zmm0, zmm4, ZMMWORD PTR [rsi+rax]
	vmulps	zmm0, zmm0, ZMMWORD PTR [r9+rax]
	add	rax, 64
	vmovaps	ZMMWORD PTR -64[rax+r8], zmm0
	cmp	rax, 32768
	jne	.L196
	test	ecx, ecx
	jle	.L202
	vmovss	xmm6, DWORD PTR .LC0[rip]
	mov	r11, rbx
	xor	r10d, r10d
	xor	r14d, r14d
.L200:
	vxorps	xmm3, xmm3, xmm3
	mov	rax, r8
	mov	rdx, r11
	vmovaps	zmm0, zmm3
	vmovaps	zmm2, zmm3
	vmovaps	zmm1, zmm3
	.p2align 4
	.p2align 3
.L198:
	vmovaps	zmm7, ZMMWORD PTR [rax]
	add	rax, 256
	vfmadd231ps	zmm1, zmm7, ZMMWORD PTR [rdx]
	add	rdx, 256
	vmovaps	zmm7, ZMMWORD PTR -192[rax]
	vfmadd231ps	zmm2, zmm7, ZMMWORD PTR -192[rdx]
	vmovaps	zmm7, ZMMWORD PTR -128[rax]
	vfmadd231ps	zmm0, zmm7, ZMMWORD PTR -128[rdx]
	vmovaps	zmm7, ZMMWORD PTR -64[rax]
	vfmadd231ps	zmm3, zmm7, ZMMWORD PTR -64[rdx]
	cmp	rax, rdi
	jne	.L198
	vaddps	zmm1, zmm1, zmm2
	vaddps	zmm0, zmm0, zmm3
	vaddps	zmm0, zmm0, zmm1
	vextractf64x4	ymm1, zmm0, 0x1
	vaddps	ymm0, ymm1, ymm0
	vextractf32x4	xmm1, ymm0, 0x1
	vaddps	xmm1, xmm1, xmm0
	vpermilps	xmm0, xmm1, 78
	vaddps	xmm0, xmm0, xmm1
	vmovaps	xmm1, xmm0
	vshufps	xmm0, xmm0, xmm0, 85
	vaddss	xmm0, xmm1, xmm0
	vcomiss	xmm0, xmm6
	vmaxss	xmm6, xmm0, xmm6
	cmova	r14d, r10d
	add	r10d, 1
	add	r11, 32768
	cmp	ecx, r10d
	jne	.L200
.L197:
	mov	DWORD PTR [r15+r13*4], r14d
	add	r13, 1
	add	r9, 32768
	cmp	r12, r13
	jne	.L201
	vzeroupper
.L208:
	lea	rsp, -40[rbp]
	pop	rbx
	pop	r12
	pop	r13
	pop	r14
	pop	r15
	pop	rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L202:
	.cfi_restore_state
	xor	r14d, r14d
	jmp	.L197
	.cfi_endproc
.LFE6765:
	.size	avx512_llama70b_8192_wavefront_f32, .-avx512_llama70b_8192_wavefront_f32
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC0:
	.long	-246811958
	.align 4
.LC2:
	.long	1048576000
	.align 4
.LC4:
	.long	1028443341
	.align 4
.LC6:
	.long	1065353216
	.ident	"GCC: (Debian 14.2.0-19) 14.2.0"
	.section	.note.GNU-stack,"",@progbits
