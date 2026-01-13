#include <stdio.h>
#include <fenv.h>
#include <stdint.h>
#include <string.h>

// float의 비트 표현을 출력
void print_float_bits(const char* name, float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    printf("%s = %.10f (0x%08X)\n", name, f, bits);
}

// Rounding 모드 (기본값)
float mul_add_rounding(float a, float b, float c) {
    int old_round = fegetround();
    fesetround(FE_TONEAREST);  // Round to nearest (기본값)

    volatile float tmp = a * b;  // volatile로 FMA 방지
    volatile float result = tmp + c;

    fesetround(old_round);
    return result;
}

// Truncation 모드
float mul_add_truncate(float a, float b, float c) {
    int old_round = fegetround();
    fesetround(FE_TOWARDZERO);  // Truncation

    volatile float tmp = a * b;
    volatile float result = tmp + c;

    fesetround(old_round);
    return result;
}

// FMA (비교용)
float mul_add_fma(float a, float b, float c) {
    return __builtin_fmaf(a, b, c);
}

int main() {
    printf("=== Rounding vs Truncation Test ===\n\n");

    // 테스트 케이스 1: 정밀도 경계에서 차이가 나는 값
    printf("Test 1: Precision boundary values\n");
    {
        float a = 1.0000001f;
        float b = 1.0000001f;
        float c = 0.0f;

        print_float_bits("a", a);
        print_float_bits("b", b);
        print_float_bits("c", c);

        float r_round = mul_add_rounding(a, b, c);
        float r_trunc = mul_add_truncate(a, b, c);
        float r_fma = mul_add_fma(a, b, c);

        printf("\nResults:\n");
        print_float_bits("rounding ", r_round);
        print_float_bits("truncate ", r_trunc);
        print_float_bits("fma      ", r_fma);
        printf("round == trunc? %s\n", r_round == r_trunc ? "YES" : "NO");
    }

    printf("\n----------------------------------------\n\n");

    // 테스트 케이스 2: 차이가 확실히 나는 값
    printf("Test 2: Values with clear difference\n");
    {
        float a = 1.5f;
        float b = 1.3333333f;
        float c = 0.1f;

        print_float_bits("a", a);
        print_float_bits("b", b);
        print_float_bits("c", c);

        float r_round = mul_add_rounding(a, b, c);
        float r_trunc = mul_add_truncate(a, b, c);
        float r_fma = mul_add_fma(a, b, c);

        printf("\nResults:\n");
        print_float_bits("rounding ", r_round);
        print_float_bits("truncate ", r_trunc);
        print_float_bits("fma      ", r_fma);
        printf("round == trunc? %s\n", r_round == r_trunc ? "YES" : "NO");
    }

    printf("\n----------------------------------------\n\n");

    // 테스트 케이스 3: 음수 값 (truncation은 0 방향)
    printf("Test 3: Negative values (truncation toward zero)\n");
    {
        float a = -1.5f;
        float b = 1.3333333f;
        float c = 0.0f;

        print_float_bits("a", a);
        print_float_bits("b", b);
        print_float_bits("c", c);

        float r_round = mul_add_rounding(a, b, c);
        float r_trunc = mul_add_truncate(a, b, c);
        float r_fma = mul_add_fma(a, b, c);

        printf("\nResults:\n");
        print_float_bits("rounding ", r_round);
        print_float_bits("truncate ", r_trunc);
        print_float_bits("fma      ", r_fma);
        printf("round == trunc? %s\n", r_round == r_trunc ? "YES" : "NO");
    }

    printf("\n----------------------------------------\n\n");

    // 테스트 케이스 4: Newton-Raphson 역수 계산
    printf("Test 4: Newton-Raphson reciprocal iteration\n");
    {
        float d = 3.0f;
        float x = 0.333333f;  // 초기 근사값

        printf("Computing 1/%f, initial x = %f\n", d, x);

        // Rounding 버전
        fesetround(FE_TONEAREST);
        float x_round = x;
        for (int i = 0; i < 3; i++) {
            volatile float tmp = (-d) * x_round;
            volatile float t = tmp + 2.0f;
            x_round = x_round * t;
        }

        // Truncation 버전
        fesetround(FE_TOWARDZERO);
        float x_trunc = x;
        for (int i = 0; i < 3; i++) {
            volatile float tmp = (-d) * x_trunc;
            volatile float t = tmp + 2.0f;
            x_trunc = x_trunc * t;
        }

        fesetround(FE_TONEAREST);  // 복원

        float exact = 1.0f / d;

        printf("\nResults:\n");
        print_float_bits("exact    ", exact);
        print_float_bits("rounding ", x_round);
        print_float_bits("truncate ", x_trunc);
        printf("\nError (rounding): %.15e\n", x_round - exact);
        printf("Error (truncate): %.15e\n", x_trunc - exact);
        printf("round == trunc? %s\n", x_round == x_trunc ? "YES" : "NO");
    }

    printf("\n=== Test Complete ===\n");
    return 0;
}
