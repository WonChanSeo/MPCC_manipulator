	.file	"test_fma_contract.c"
	.text
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC0:
	.string	"%s = %.15f (0x%08X)\n"
	.text
	.p2align 4
	.globl	print_float_bits
	.type	print_float_bits, @function
print_float_bits:
.LFB35:
	.cfi_startproc
	endbr64
	movq	%rdi, %rdx
	vmovd	%xmm0, %ecx
	vcvtss2sd	%xmm0, %xmm0, %xmm1
	leaq	.LC0(%rip), %rsi
	vmovsd	%xmm1, %xmm1, %xmm0
	movl	$1, %edi
	movl	$1, %eax
	jmp	__printf_chk@PLT
	.cfi_endproc
.LFE35:
	.size	print_float_bits, .-print_float_bits
	.section	.rodata.str1.1
.LC2:
	.string	"a = %.15f\n"
.LC4:
	.string	"b = %.15f\n"
.LC6:
	.string	"c = %.15f\n\n"
.LC8:
	.string	"Expression (a*b+c):     "
.LC9:
	.string	""
.LC11:
	.string	"Explicit FMA:           "
.LC13:
	.string	"Separated (volatile):   "
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC14:
	.string	">>> Expression != FMA: \354\273\264\355\214\214\354\235\274\353\237\254\352\260\200 FMA\353\245\274 \354\202\254\354\232\251\355\225\230\354\247\200 \354\225\212\354\235\214"
	.align 8
.LC16:
	.string	">>> Expression == Separated: \353\266\204\353\246\254\353\220\234 \354\227\260\354\202\260\352\263\274 \352\260\231\354\235\214"
	.align 8
.LC17:
	.string	">>> Expression != Separated: \353\266\204\353\246\254\353\220\234 \354\227\260\354\202\260\352\263\274 \353\213\244\353\246\204"
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB36:
	.cfi_startproc
	endbr64
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	movl	$1, %edi
	leaq	.LC2(%rip), %rsi
	movl	$1, %eax
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	leaq	.LC9(%rip), %r12
	leaq	.LC0(%rip), %rbp
	subq	$24, %rsp
	.cfi_def_cfa_offset 48
	vmovsd	.LC1(%rip), %xmm0
	call	__printf_chk@PLT
	movl	$1, %edi
	movl	$1, %eax
	vmovsd	.LC3(%rip), %xmm0
	leaq	.LC4(%rip), %rsi
	call	__printf_chk@PLT
	movl	$1, %edi
	movl	$1, %eax
	vmovsd	.LC5(%rip), %xmm0
	leaq	.LC6(%rip), %rsi
	call	__printf_chk@PLT
	vmovss	.LC7(%rip), %xmm1
	leaq	.LC8(%rip), %rsi
	xorl	%eax, %eax
	movl	$1, %edi
	vmovss	%xmm1, 8(%rsp)
	vmovss	8(%rsp), %xmm0
	vsubss	%xmm1, %xmm0, %xmm0
	vmovss	%xmm0, 12(%rsp)
	call	__printf_chk@PLT
	xorl	%ecx, %ecx
	vxorpd	%xmm0, %xmm0, %xmm0
	movq	%r12, %rdx
	movq	%rbp, %rsi
	movl	$1, %edi
	movl	$1, %eax
	call	__printf_chk@PLT
	leaq	.LC11(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	movl	$687865856, %ecx
	movq	%r12, %rdx
	movq	%rbp, %rsi
	vmovsd	.LC12(%rip), %xmm0
	movl	$1, %edi
	movl	$1, %eax
	call	__printf_chk@PLT
	leaq	.LC13(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	movq	%r12, %rdx
	movq	%rbp, %rsi
	movl	$1, %edi
	vmovss	12(%rsp), %xmm0
	movl	$1, %eax
	vmovd	%xmm0, %ecx
	vcvtss2sd	%xmm0, %xmm0, %xmm1
	vmovsd	%xmm1, %xmm1, %xmm0
	call	__printf_chk@PLT
	movl	$10, %edi
	call	putchar@PLT
	leaq	.LC14(%rip), %rdi
	call	puts@PLT
	vmovss	12(%rsp), %xmm0
	vucomiss	.LC15(%rip), %xmm0
	jp	.L4
	jne	.L4
	leaq	.LC16(%rip), %rdi
	call	puts@PLT
.L6:
	addq	$24, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 24
	xorl	%eax, %eax
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
.L4:
	.cfi_restore_state
	leaq	.LC17(%rip), %rdi
	call	puts@PLT
	jmp	.L6
	.cfi_endproc
.LFE36:
	.size	main, .-main
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC1:
	.long	536870912
	.long	1072693248
	.align 8
.LC3:
	.long	1073741824
	.long	1072693248
	.align 8
.LC5:
	.long	1610612736
	.long	-1074790400
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC7:
	.long	1065353219
	.section	.rodata.cst8
	.align 8
.LC12:
	.long	0
	.long	1025507328
	.set	.LC15,.LC12
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04.2) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
