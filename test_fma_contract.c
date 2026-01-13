#include <stdio.h>
#include <stdint.h>
#include <string.h>

void print_float_bits(const char* name, float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    printf("%s = %.15f (0x%08X)\n", name, f, bits);
}

int main() {
    float a = 1.0000001f;
    float b = 1.0000002f;
    float c = -1.0000003f;

    printf("a = %.15f\n", a);
    printf("b = %.15f\n", b);
    printf("c = %.15f\n\n", c);

    // a*b + c 를 계산
    // FMA: (a*b + c)를 한 번에 계산 (중간 rounding 없음)
    // 분리: a*b 계산 후 rounding, 그 다음 +c 후 rounding

    float result_expr = a * b + c;  // 컴파일러가 FMA로 합칠 수 있음
    float result_fma = __builtin_fmaf(a, b, c);  // 명시적 FMA

    // 분리된 계산
    volatile float tmp = a * b;
    volatile float result_sep = tmp + c;

    printf("Expression (a*b+c):     ");
    print_float_bits("", result_expr);

    printf("Explicit FMA:           ");
    print_float_bits("", result_fma);

    printf("Separated (volatile):   ");
    print_float_bits("", (float)result_sep);

    printf("\n");
    if (result_expr == result_fma) {
        printf(">>> Expression == FMA: 컴파일러가 FMA를 사용했을 가능성 높음\n");
    } else {
        printf(">>> Expression != FMA: 컴파일러가 FMA를 사용하지 않음\n");
    }

    if (result_expr == result_sep) {
        printf(">>> Expression == Separated: 분리된 연산과 같음\n");
    } else {
        printf(">>> Expression != Separated: 분리된 연산과 다름\n");
    }

    return 0;
}
