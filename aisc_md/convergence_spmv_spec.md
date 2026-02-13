# Convergence Check SpMV — Bit-Exact Specification

## 목적

ASIC RTL (admm_top.v)의 convergence check에서 사용하는 A\*x, P\*x, A'*y 연산의
**정확한 FP32 연산 순서**를 기술한다. C golden model에서 동일한 연산 순서를
재현하면 bit-exact 정합이 가능하다.

---

## 1. 기본 연산 프리미티브

모든 연산은 IEEE 754 single-precision (FP32)이며, 두 가지 프리미티브만 사용한다:

```c
uint32_t fp32_mul(uint32_t a, uint32_t b);  // fp32_mul_wrapper
uint32_t fp32_add(uint32_t a, uint32_t b);  // fp32_add_wrapper
```

C 코드에서 bit-exact 구현:
```c
// C에서 FP32 연산을 bit-exact로 재현하는 방법
float a_f, b_f;
memcpy(&a_f, &a_bits, 4);
memcpy(&b_f, &b_bits, 4);
float result_f = a_f * b_f;   // 또는 a_f + b_f
memcpy(&result_bits, &result_f, 4);
```

> **핵심**: FP32 덧셈은 교환법칙(a+b = b+a)은 성립하지만,
> 결합법칙((a+b)+c ≠ a+(b+c))은 성립하지 않는다.
> 따라서 **트리 구조의 덧셈 순서**가 bit-exact 정합의 핵심이다.

---

## 2. adder_tree_4 연산 순서

`adder_tree_4`는 4개의 FP32 입력을 1개로 축약하는 조합 로직이다.

### 입력: `din[127:0]` = `{in3, in2, in1, in0}` (MSB→LSB 순서)

```
din[31:0]   = in0
din[63:32]  = in1
din[95:64]  = in2
din[127:96] = in3
```

### 내부 연산

```
temp0 = fp32_add(in0, in1)      // stage1[0] = din[31:0] + din[63:32]
temp1 = fp32_add(in2, in3)      // stage1[1] = din[95:64] + din[127:96]
result = fp32_add(temp0, temp1)  // stage2 = stage1[0] + stage1[1]
```

### C 함수

```c
uint32_t adder_tree_4(uint32_t in3, uint32_t in2, uint32_t in1, uint32_t in0) {
    uint32_t temp0 = fp32_add(in0, in1);
    uint32_t temp1 = fp32_add(in2, in3);
    return fp32_add(temp0, temp1);
}
```

> 인자 순서 주의: Verilog `{in3, in2, in1, in0}`에서 **in0이 LSB**이므로
> `stage1[0] = in0 + in1`, `stage1[1] = in2 + in3` 순서.

---

## 3. A\*x (행렬-벡터 곱, 479 스칼라 출력)

```
A: 479×179,  x: 179-dim
Ax[row] = Σ_{j=0}^{178} A[row][j] * x[j]    (for row = 0..478)
```

179개 곱의 합을 PE adder tree로 축약한다. adder tree는 **128 lanes(0-127)만** 지원하므로
**2-pass**로 나누어 연산한다.

### 3.1 Step 1: 179개 곱셈 (모든 lane 동시)

```c
uint32_t mul[179];
for (int j = 0; j < 179; j++)
    mul[j] = fp32_mul(A[row][j], x[j]);
```

### 3.2 Step 2: PASS 1 — lanes 0-127

#### 2a. ODD MAC pair sum (64개)

ODD MAC[k] = lanes (2k, 2k+1)의 곱을 더한 것:

```c
uint32_t pair[64];
for (int k = 0; k < 64; k++)
    pair[k] = fp32_add(mul[2*k+1], mul[2*k]);
    //                  ^^^^^^^^    ^^^^^^^^
    //                  din_A(odd)  din_B(even)
```

> `fp32_add_mux_3in` mode=00: `fp32_add(din_A=mul[odd], din_B=mul[even])`

#### 2b. Adder Tree Stage 1: 64 → 16

16개 그룹, 각 4개의 pair sum을 축약:

```c
uint32_t stage1[16];
for (int g = 0; g < 16; g++) {
    // Verilog: din = {pair[4g], pair[4g+1], pair[4g+2], pair[4g+3]}
    //          in3 = pair[4g], in2 = pair[4g+1], in1 = pair[4g+2], in0 = pair[4g+3]
    stage1[g] = adder_tree_4(pair[4*g], pair[4*g+1], pair[4*g+2], pair[4*g+3]);
}
```

풀어서 쓰면:
```c
stage1[g] = fp32_add(
    fp32_add(pair[4*g+3], pair[4*g+2]),    // temp0 = in0 + in1
    fp32_add(pair[4*g+1], pair[4*g])       // temp1 = in2 + in3
);
```

**구체적 pair index 매핑 (Verilog `i = 8*g+1`):**

| g  | Verilog i | MAC indices       | pair indices     | din order {in3,in2,in1,in0} |
|----|-----------|-------------------|------------------|-----------------------------|
| 0  | 1         | 1, 3, 5, 7        | 0, 1, 2, 3      | {pair[0], pair[1], pair[2], pair[3]} |
| 1  | 9         | 9, 11, 13, 15     | 4, 5, 6, 7      | {pair[4], pair[5], pair[6], pair[7]} |
| …  | …         | …                 | …                | … |
| 15 | 121       | 121,123,125,127   | 60,61,62,63      | {pair[60],pair[61],pair[62],pair[63]}|

#### 2c. Adder Tree Stage 2: 16 → 4

```c
uint32_t stage2[4];
for (int g = 0; g < 4; g++) {
    // din = {stage1[4g], stage1[4g+1], stage1[4g+2], stage1[4g+3]}
    stage2[g] = adder_tree_4(stage1[4*g], stage1[4*g+1], stage1[4*g+2], stage1[4*g+3]);
}
```

#### 2d. Adder Tree Stage 3: 4 → 1

```c
// din = {stage2[0], stage2[1], stage2[2], stage2[3]}
uint32_t stage3_pass1 = adder_tree_4(stage2[0], stage2[1], stage2[2], stage2[3]);
```

#### 2e. Stage 4 (acc_clear=1): 값 보관

```c
// din = {stage3_pass1, acc_value=0, bias=0, 0}
// result = fp32_add(fp32_add(0, 0), fp32_add(0, stage3_pass1))
// fp32_add(0, x) = x  이므로:
uint32_t pass1_result = stage3_pass1;
```

> Stage 4 adder_tree_4는 `{stage3, acc_value, bias_gated, 32'd0}`을 입력받는다.
> acc_clear=1일 때 acc_value=0, bias_add_en=0일 때 bias_gated=0이므로
> 결과는 stage3과 동일하다.

### 3.3 Step 3: PASS 2 — lanes 128-178 (shifted to 0-50)

lanes 128-178의 51개 값을 lanes 0-50에 매핑하여 동일한 tree를 통과시킨다.

#### 3a. Shifted 곱셈

```c
uint32_t mul2[179];  // shifted: original lane 128+j → shifted lane j
for (int j = 0; j < 51; j++)
    mul2[j] = fp32_mul(A[row][128+j], x[128+j]);
for (int j = 51; j < 179; j++)
    mul2[j] = 0;  // zero-padded
```

#### 3b. ODD MAC pair sums (shifted)

```c
uint32_t pair2[64];
for (int k = 0; k < 64; k++)
    pair2[k] = fp32_add(mul2[2*k+1], mul2[2*k]);
```

유효한 pair들:
- pair2[0..24]: 완전한 쌍 (lanes 0-49 → A[128..177])
- pair2[25]: `fp32_add(0, mul2[50])` = `fp32_add(0, A[row][178]*x[178])`
  → lane 50은 EVEN이지만, ODD MAC[51]이 `mul2[51](=0) + mul2[50]`을 계산
- pair2[26..63]: `fp32_add(0, 0)` = 0

#### 3c. Stage 1-3: 동일한 tree 구조

```c
uint32_t stage1_p2[16], stage2_p2[4];
for (int g = 0; g < 16; g++)
    stage1_p2[g] = adder_tree_4(pair2[4*g], pair2[4*g+1], pair2[4*g+2], pair2[4*g+3]);
for (int g = 0; g < 4; g++)
    stage2_p2[g] = adder_tree_4(stage1_p2[4*g], stage1_p2[4*g+1], stage1_p2[4*g+2], stage1_p2[4*g+3]);
uint32_t stage3_pass2 = adder_tree_4(stage2_p2[0], stage2_p2[1], stage2_p2[2], stage2_p2[3]);
```

#### 3d. Stage 4 (acc_clear=0): pass1과 누적

```c
// din = {stage3_pass2, pass1_result, 0, 0}
// result = fp32_add(fp32_add(0, 0), fp32_add(pass1_result, stage3_pass2))
uint32_t Ax_row = fp32_add(pass1_result, stage3_pass2);
```

> Stage 4에서 `in3=stage3_pass2`, `in2=acc_value(=pass1_result)`, `in1=0`, `in0=0`
> → `fp32_add(fp32_add(0, 0), fp32_add(pass1_result, stage3_pass2))`
> → `fp32_add(0, fp32_add(pass1_result, stage3_pass2))`
> → `fp32_add(pass1_result, stage3_pass2)`

### 3.4 전체 A\*x C 코드

```c
void compute_Ax(uint32_t Ax[479], uint32_t A[479][179], uint32_t x[179]) {
    for (int row = 0; row < 479; row++) {
        // --- 179 multiplications ---
        uint32_t mul[179];
        for (int j = 0; j < 179; j++)
            mul[j] = fp32_mul(A[row][j], x[j]);

        // --- PASS 1: lanes 0-127 ---
        uint32_t pair[64];
        for (int k = 0; k < 64; k++)
            pair[k] = fp32_add(mul[2*k+1], mul[2*k]);

        uint32_t s1[16];
        for (int g = 0; g < 16; g++)
            s1[g] = adder_tree_4(pair[4*g], pair[4*g+1], pair[4*g+2], pair[4*g+3]);

        uint32_t s2[4];
        for (int g = 0; g < 4; g++)
            s2[g] = adder_tree_4(s1[4*g], s1[4*g+1], s1[4*g+2], s1[4*g+3]);

        uint32_t pass1 = adder_tree_4(s2[0], s2[1], s2[2], s2[3]);

        // --- PASS 2: lanes 128-178 shifted to 0-50 ---
        uint32_t mul2[179];
        memset(mul2, 0, sizeof(mul2));
        for (int j = 0; j < 51; j++)
            mul2[j] = fp32_mul(A[row][128+j], x[128+j]);

        uint32_t pair2[64];
        for (int k = 0; k < 64; k++)
            pair2[k] = fp32_add(mul2[2*k+1], mul2[2*k]);

        uint32_t s1b[16];
        for (int g = 0; g < 16; g++)
            s1b[g] = adder_tree_4(pair2[4*g], pair2[4*g+1], pair2[4*g+2], pair2[4*g+3]);

        uint32_t s2b[4];
        for (int g = 0; g < 4; g++)
            s2b[g] = adder_tree_4(s1b[4*g], s1b[4*g+1], s1b[4*g+2], s1b[4*g+3]);

        uint32_t pass2 = adder_tree_4(s2b[0], s2b[1], s2b[2], s2b[3]);

        // --- Stage 4: accumulate ---
        Ax[row] = fp32_add(pass1, pass2);
    }
}
```

---

## 4. P\*x (행렬-벡터 곱, 179 스칼라 출력)

```
P: 179×179,  x: 179-dim
Px[i] = Σ_{j=0}^{178} P[i][j] * x[j]    (for i = 0..178)
```

A\*x와 **완전히 동일한 2-pass adder tree** 구조이다.
유일한 차이: 행 수가 479 → 179.

```c
void compute_Px(uint32_t Px[179], uint32_t P[179][179], uint32_t x[179]) {
    for (int row = 0; row < 179; row++) {
        // A*x와 동일한 2-pass adder tree (Section 3 참조)
        Px[row] = two_pass_adder_tree_dot(P[row], x, 179);
    }
}
```

---

## 5. A'*y (Transpose 곱, 179 벡터 출력)

```
A: 479×179,  y: 479-dim
(A'y)[j] = Σ_{i=0}^{478} A[i][j] * y[i]    (for j = 0..178)
```

### 핵심: adder tree를 사용하지 않음

A'*y는 479-element 내적이 필요하지만 adder tree는 128개까지만 지원한다.
대신 **열 방향 누적 (outer product accumulation)** 방식을 사용한다:

```
A'y = Σ_{i=0}^{478} y[i] · A[i,:]
```

즉 A의 각 행에 대응하는 y 스칼라를 곱해서 179-dim 벡터끼리 누적한다.

### PE 매핑: mac_mode=2'b10 (external C)

각 iteration (row i):
```
result[j] = fp32_add(fp32_mul(A[i][j], y[i]), accum[j])
                      ^^^^^^^^  ^^^^    ^^^^^   ^^^^^^^^
                      din_A     din_B   │       din_C
                                  (broadcast)
```

**모든 179 lanes 동시 연산**, 479회 반복.

### 5.1 ODD MAC (j = 1, 3, 5, ..., 127)

`fp32_add_mux_3in` mode=2'b10:
```
op_B = din_C = accum[j]
result = fp32_add(din_A, op_B) = fp32_add(fp32_mul(A[i][j], y[i]), accum[j])
```

> din_A = mul output = fp32_mul(A[i][j], y[i])  (combinational)
> din_C = accum[j]  (from z_unclamped_reg)

### 5.2 EVEN MAC (j = 0, 2, 4, ..., 126) 및 EXTRA MAC (j = 128..178)

`fp32_add_mux` mode=0 (external):
```
op_B = din_B = accum[j]
result = fp32_add(din_A, op_B) = fp32_add(fp32_mul(A[i][j], y[i]), accum[j])
```

> din_A = mul output = fp32_mul(A[i][j], y[i])
> din_B = din_C[j] = accum[j]

### 5.3 결론: 모든 lane이 동일한 연산

ODD/EVEN/EXTRA 구분 없이 모든 179 lanes에서:
```
accum[j] = fp32_add(fp32_mul(A[i][j], y[i]), accum[j])
```

### 5.4 전체 A'*y C 코드

```c
void compute_Aty(uint32_t Aty[179], uint32_t A[479][179], uint32_t y[479]) {
    // Initialize accumulator to zero
    for (int j = 0; j < 179; j++)
        Aty[j] = 0;  // +0.0 in FP32

    // Outer product accumulation: 479 rows × 179 lanes
    for (int i = 0; i < 479; i++) {
        for (int j = 0; j < 179; j++) {
            uint32_t prod = fp32_mul(A[i][j], y[i]);
            Aty[j] = fp32_add(prod, Aty[j]);
            //                ^^^^  ^^^^^^
            //                din_A  din_C (or din_B)
        }
    }
}
```

> **주의**: `fp32_add`의 인자 순서: **prod가 .a, accum이 .b**
> IEEE 754에서 a+b = b+a이므로 실제로는 순서 무관하지만,
> 정확한 RTL 매칭을 위해 위 순서를 권장.

---

## 6. C 코드 수정 가이드

### 6.1 기존 C 코드 (OSQP)와의 차이

| 항목 | 기존 C 코드 (double) | ASIC (FP32) |
|------|---------------------|-------------|
| 정밀도 | double (64-bit) | float (32-bit) |
| A*x | CSC SpMV (순차 누적) | 2-pass adder tree |
| A'*y | CSC SpMV (순차 누적) | outer product 누적 |
| A 저장 형식 | CSC (sparse) | dense 179-lane (banked) |

### 6.2 수정 절차

1. **FP32로 변환**: 모든 연산을 `float`으로 수행 (또는 `uint32_t` + softfloat)

2. **A*x 함수 교체**: 기존 sparse CSC 곱을 Section 3.4의 `compute_Ax()`로 교체.
   핵심은 `adder_tree_4()` 함수의 **operand 순서**를 정확히 맞추는 것.

3. **P*x 함수 교체**: A*x와 동일한 구조 (행 수만 179).

4. **A'*y 함수 교체**: 기존 CSC transpose multiply를 Section 5.4의
   `compute_Aty()`로 교체. 여기서 **행 순서 (i=0 → 478)**와
   **fp32_add(prod, accum)** 순서가 일치해야 한다.

5. **A 행렬 dense 형식**: C 코드에서 A를 `A[row][col]` (479×179) dense 배열로
   준비해야 한다. ASIC에서 A_comp SRAM에 저장된 것과 동일한 scaled 행렬을 사용.

### 6.3 batch 구조 (A'*y의 y 접근 패턴)

y는 3 batch로 나뉘어 SRAM에 저장되어 있다:
- Batch 0: y[0:178]   (SRAM[77])
- Batch 1: y[179:357]  (SRAM[78])
- Batch 2: y[358:478]  (SRAM[79])

FSM은 각 batch 시작 시 y를 로드한다. C 코드에서는 단순히 `y[i]`로 접근하면
되므로 batch는 무시해도 bit-exact 결과에 영향 없다.

### 6.4 Convergence check 전체 흐름 (C 코드)

```c
void convergence_check(
    uint32_t A[479][179], uint32_t P[179][179],
    uint32_t x[179], uint32_t y[479], uint32_t z[479], uint32_t q[179],
    int9_t d_exp[179], int9_t e_exp[479]  // Dinv/Einv exponents
) {
    // 1. A*x (Section 3)
    uint32_t Ax[479];
    compute_Ax(Ax, A, x);

    // 2. Primal residual (per-row running max)
    uint32_t max_prim_res = 0, max_scaled_prim = 0;
    uint32_t max_einv_ax = 0, max_einv_z = 0;
    for (int i = 0; i < 479; i++) {
        uint32_t diff = fp32_sub(Ax[i], z[i]);  // fp32_add(Ax[i], negate(z[i]))
        update_max_abs(&max_scaled_prim, diff);
        update_max_abs(&max_prim_res,  exp_adjust(diff, e_exp[i]));
        update_max_abs(&max_einv_ax, exp_adjust(Ax[i], e_exp[i]));
        update_max_abs(&max_einv_z,  exp_adjust(z[i],  e_exp[i]));
    }

    // 3. P*x (Section 4)
    uint32_t Px[179];
    compute_Px(Px, P, x);

    // 4. A'*y (Section 5)
    uint32_t Aty[179];
    compute_Aty(Aty, A, y);

    // 5. Dual residual: dr = q + Px + Aty (PE로 2-step 계산)
    // Step 1: temp = 1.0*q + Px  (mac_mode=10: fp32_add(fp32_mul(1.0,q), Px))
    // Step 2: dr   = 1.0*temp + Aty
    uint32_t temp[179], dr[179];
    for (int j = 0; j < 179; j++) {
        temp[j] = fp32_add(fp32_mul(FP32_ONE, q[j]), Px[j]);
        dr[j]   = fp32_add(fp32_mul(FP32_ONE, temp[j]), Aty[j]);
    }

    // 6. Dual inf-norm (sequential, 179 lanes)
    uint32_t max_dual_res = 0, max_scaled_dual = 0;
    uint32_t max_dinv_q = 0, max_dinv_px = 0, max_dinv_aty = 0;
    for (int j = 0; j < 179; j++) {
        update_max_abs(&max_scaled_dual, dr[j]);
        update_max_abs(&max_dual_res, exp_adjust(dr[j], d_exp[j]));
        update_max_abs(&max_dinv_q,   exp_adjust(q[j],  d_exp[j]));
        update_max_abs(&max_dinv_px,  exp_adjust(Px[j], d_exp[j]));
        update_max_abs(&max_dinv_aty, exp_adjust(Aty[j], d_exp[j]));
    }

    // 7. Tolerance check
    // eps_prim = 2^(-10) * (1.0 + max(max_einv_ax, max_einv_z))
    // eps_dual = 2^(-10) * (1.0 + max(max_dinv_q, max_dinv_px, max_dinv_aty))
    // converged = (max_prim_res <= eps_prim) && (max_dual_res <= eps_dual)
}
```

### 6.5 주의사항

1. **`fp32_mul(1.0, q[j])`**: 하드웨어에서 PE의 multiplier를 항상 통과하므로
   C 코드에서도 `1.0 * q`를 명시적으로 수행해야 한다. 대부분의 경우 결과는
   q[j]와 동일하지만, denormalized number 등 edge case에서 차이 발생 가능.

2. **Ax-z 뺄셈**: RTL에서는 `fp32_add(Ax, z ^ 0x80000000)`으로 구현.
   C에서도 sign bit flip 후 add로 구현하면 bit-exact:
   ```c
   uint32_t fp32_sub(uint32_t a, uint32_t b) {
       return fp32_add(a, b ^ 0x80000000);
   }
   ```

3. **exp_adjust**: 순수 exponent 덧셈. zero(exp==0)는 0 유지:
   ```c
   uint32_t exp_adjust(uint32_t val, int9_t adj) {
       if ((val & 0x7F800000) == 0) return 0;  // zero/denorm
       uint32_t exp = (val >> 23) & 0xFF;
       exp = (exp + (uint8_t)adj) & 0xFF;
       return (val & 0x807FFFFF) | (exp << 23);
   }
   ```

4. **Infinity norm의 max 비교**: `|a| > |b|` ⟺ `(a & 0x7FFFFFFF) > (b & 0x7FFFFFFF)`
   (부호 비트 제거 후 unsigned 비교)
