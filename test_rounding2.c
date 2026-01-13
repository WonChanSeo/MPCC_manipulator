#include <stdio.h>
#include <fenv.h>
#include <stdint.h>
#include <string.h>

void print_float_bits(const char* name, float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    printf("%s = %.10f (0x%08X)\n", name, f, bits);
}

int main() {
    printf("=== Test: Complex expression a = b * c + d - e * 5 ===\n\n");

    float b = 1.5f;
    float c = 1.3333333f;
    float d = 0.7777777f;
    float e = 0.1234567f;

    print_float_bits("b", b);
    print_float_bits("c", c);
    print_float_bits("d", d);
    print_float_bits("e", e);
    printf("\n");

    // 방법 1: 한 줄로 작성 (컴파일러가 최적화할 수 있음)
    printf("=== Method 1: Single expression ===\n");
    {
        fesetround(FE_TONEAREST);
        float r_round = b * c + d - e * 5.0f;

        fesetround(FE_TOWARDZERO);
        float r_trunc = b * c + d - e * 5.0f;

        fesetround(FE_TONEAREST);

        print_float_bits("rounding ", r_round);
        print_float_bits("truncate ", r_trunc);
        printf("같은 결과? %s\n", r_round == r_trunc ? "YES" : "NO");
    }

    printf("\n=== Method 2: volatile로 각 단계 강제 분리 ===\n");
    {
        fesetround(FE_TONEAREST);
        volatile float t1 = b * c;      // step 1: b * c
        volatile float t2 = t1 + d;     // step 2: + d
        volatile float t3 = e * 5.0f;   // step 3: e * 5
        volatile float r_round = t2 - t3; // step 4: - (e * 5)

        fesetround(FE_TOWARDZERO);
        volatile float s1 = b * c;
        volatile float s2 = s1 + d;
        volatile float s3 = e * 5.0f;
        volatile float r_trunc = s2 - s3;

        fesetround(FE_TONEAREST);

        print_float_bits("rounding ", (float)r_round);
        print_float_bits("truncate ", (float)r_trunc);
        printf("같은 결과? %s\n", r_round == r_trunc ? "YES" : "NO");
    }

    printf("\n=== Method 3: 각 연산 결과 비교 ===\n");
    {
        printf("\nStep-by-step comparison:\n");

        // Step 1: b * c
        fesetround(FE_TONEAREST);
        volatile float t1_round = b * c;
        fesetround(FE_TOWARDZERO);
        volatile float t1_trunc = b * c;
        printf("Step1 (b*c):     round=0x%08X, trunc=0x%08X, diff=%s\n",
               *(uint32_t*)&t1_round, *(uint32_t*)&t1_trunc,
               t1_round == t1_trunc ? "NO" : "YES");

        // Step 2: t1 + d
        fesetround(FE_TONEAREST);
        volatile float t2_round = t1_round + d;
        fesetround(FE_TOWARDZERO);
        volatile float t2_trunc = t1_trunc + d;
        printf("Step2 (t1+d):    round=0x%08X, trunc=0x%08X, diff=%s\n",
               *(uint32_t*)&t2_round, *(uint32_t*)&t2_trunc,
               t2_round == t2_trunc ? "NO" : "YES");

        // Step 3: e * 5
        fesetround(FE_TONEAREST);
        volatile float t3_round = e * 5.0f;
        fesetround(FE_TOWARDZERO);
        volatile float t3_trunc = e * 5.0f;
        printf("Step3 (e*5):     round=0x%08X, trunc=0x%08X, diff=%s\n",
               *(uint32_t*)&t3_round, *(uint32_t*)&t3_trunc,
               t3_round == t3_trunc ? "NO" : "YES");

        // Step 4: t2 - t3
        fesetround(FE_TONEAREST);
        volatile float r_round = t2_round - t3_round;
        fesetround(FE_TOWARDZERO);
        volatile float r_trunc = t2_trunc - t3_trunc;
        printf("Step4 (t2-t3):   round=0x%08X, trunc=0x%08X, diff=%s\n",
               *(uint32_t*)&r_round, *(uint32_t*)&r_trunc,
               r_round == r_trunc ? "NO" : "YES");

        fesetround(FE_TONEAREST);
    }

    printf("\n=== 결론 ===\n");
    printf("- 한 줄 표현식에서도 fesetround가 각 연산에 적용됨\n");
    printf("- 단, 컴파일러가 FMA로 합칠 수 있으므로 -ffp-contract=off 필요\n");
    printf("- volatile을 쓰면 각 연산이 확실히 분리됨\n");

    return 0;
}
