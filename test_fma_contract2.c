#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

void print_float_bits(const char* name, float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    printf("%s = %.15f (0x%08X)\n", name, f, bits);
}

// noinline으로 컴파일러 최적화 방지
__attribute__((noinline))
float compute_expr(float a, float b, float c) {
    return a * b + c;
}

__attribute__((noinline))
float compute_fma(float a, float b, float c) {
    return __builtin_fmaf(a, b, c);
}

__attribute__((noinline))
float compute_sep(float a, float b, float c) {
    volatile float tmp = a * b;
    return tmp + c;
}

int main(int argc, char** argv) {
    // 런타임에 값 설정 (상수 폴딩 방지)
    float a = argc > 1 ? atof(argv[1]) : 1.0000001f;
    float b = argc > 2 ? atof(argv[2]) : 1.0000002f;
    float c = argc > 3 ? atof(argv[3]) : -1.0000003f;

    printf("a = %.15f\n", a);
    printf("b = %.15f\n", b);
    printf("c = %.15f\n\n", c);

    float result_expr = compute_expr(a, b, c);
    float result_fma = compute_fma(a, b, c);
    float result_sep = compute_sep(a, b, c);

    printf("Expression (a*b+c):     ");
    print_float_bits("", result_expr);

    printf("Explicit FMA:           ");
    print_float_bits("", result_fma);

    printf("Separated (volatile):   ");
    print_float_bits("", result_sep);

    printf("\n");
    if (result_expr == result_fma) {
        printf(">>> Expression == FMA: 컴파일러가 FMA를 사용함!\n");
    } else if (result_expr == result_sep) {
        printf(">>> Expression == Separated: 컴파일러가 FMA를 사용하지 않음 (mul + add 분리)\n");
    } else {
        printf(">>> 다른 결과\n");
    }

    return 0;
}
