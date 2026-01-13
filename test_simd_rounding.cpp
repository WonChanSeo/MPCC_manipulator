#include <iostream>
#include <iomanip>
#include <cfenv>
#include <cstring>
#include <cstdint>

// Eigen 포함
#include <Eigen/Dense>

void print_float_bits(const char* name, float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    std::cout << name << " = " << std::setprecision(10) << f
              << " (0x" << std::hex << std::setfill('0') << std::setw(8) << bits << ")"
              << std::dec << std::endl;
}

// SIMD를 사용하지 않도록 강제하는 버전
__attribute__((noinline))
float scalar_dot_product(const float* a, const float* b, int n) {
    float result = 0.0f;
    for (int i = 0; i < n; ++i) {
        volatile float prod = a[i] * b[i];  // volatile로 SIMD 방지
        volatile float sum = result + prod;
        result = sum;
    }
    return result;
}

int main() {
    std::cout << "=== SIMD Rounding Mode Test ===" << std::endl << std::endl;

    // Eigen이 SIMD를 사용하는지 확인
    std::cout << "Eigen SIMD info:" << std::endl;
#ifdef EIGEN_VECTORIZE
    std::cout << "  EIGEN_VECTORIZE is defined" << std::endl;
#else
    std::cout << "  EIGEN_VECTORIZE is NOT defined" << std::endl;
#endif
#ifdef EIGEN_VECTORIZE_SSE
    std::cout << "  EIGEN_VECTORIZE_SSE is defined" << std::endl;
#endif
#ifdef EIGEN_VECTORIZE_SSE2
    std::cout << "  EIGEN_VECTORIZE_SSE2 is defined" << std::endl;
#endif
#ifdef EIGEN_VECTORIZE_SSE4_1
    std::cout << "  EIGEN_VECTORIZE_SSE4_1 is defined" << std::endl;
#endif
#ifdef EIGEN_VECTORIZE_AVX
    std::cout << "  EIGEN_VECTORIZE_AVX is defined" << std::endl;
#endif
#ifdef EIGEN_VECTORIZE_AVX2
    std::cout << "  EIGEN_VECTORIZE_AVX2 is defined" << std::endl;
#endif
    std::cout << std::endl;

    // 테스트용 벡터 생성 (rounding 차이가 발생하도록 설계)
    const int N = 8;
    Eigen::VectorXf vec_a(N), vec_b(N);

    // 값 설정 - rounding 차이가 나도록 설계
    vec_a << 1.0000001f, 1.3333333f, 0.7777777f, 1.1111111f,
             0.9999999f, 2.2222222f, 0.5555555f, 1.4444444f;
    vec_b << 1.0000002f, 0.6666666f, 1.2222222f, 0.8888888f,
             1.0000001f, 0.4444444f, 1.7777777f, 0.6111111f;

    std::cout << "=== Test 1: Eigen dot product ===" << std::endl;

    // Rounding (기본값)
    std::fesetround(FE_TONEAREST);
    float eigen_round = vec_a.dot(vec_b);

    // Truncation
    std::fesetround(FE_TOWARDZERO);
    float eigen_trunc = vec_a.dot(vec_b);

    std::fesetround(FE_TONEAREST);

    print_float_bits("Eigen rounding ", eigen_round);
    print_float_bits("Eigen truncate ", eigen_trunc);
    std::cout << "Same result? " << (eigen_round == eigen_trunc ? "YES (SIMD ignores rounding mode)" : "NO (rounding mode works)") << std::endl;
    std::cout << std::endl;

    // 스칼라 버전 (비교용)
    std::cout << "=== Test 2: Scalar dot product (no SIMD) ===" << std::endl;

    std::fesetround(FE_TONEAREST);
    float scalar_round = scalar_dot_product(vec_a.data(), vec_b.data(), N);

    std::fesetround(FE_TOWARDZERO);
    float scalar_trunc = scalar_dot_product(vec_a.data(), vec_b.data(), N);

    std::fesetround(FE_TONEAREST);

    print_float_bits("Scalar rounding", scalar_round);
    print_float_bits("Scalar truncate", scalar_trunc);
    std::cout << "Same result? " << (scalar_round == scalar_trunc ? "YES" : "NO (rounding mode works)") << std::endl;
    std::cout << std::endl;

    // 행렬 곱셈 테스트
    std::cout << "=== Test 3: Eigen matrix multiplication ===" << std::endl;

    Eigen::MatrixXf mat_a(4, 4), mat_b(4, 4);
    mat_a << 1.0000001f, 1.3333333f, 0.7777777f, 1.1111111f,
             0.9999999f, 2.2222222f, 0.5555555f, 1.4444444f,
             1.2345678f, 0.8765432f, 1.5432109f, 0.6543210f,
             0.1234567f, 1.9876543f, 0.3456789f, 1.7654321f;

    mat_b << 1.0000002f, 0.6666666f, 1.2222222f, 0.8888888f,
             1.0000001f, 0.4444444f, 1.7777777f, 0.6111111f,
             0.9876543f, 1.0123456f, 0.5678901f, 1.4321098f,
             1.1111111f, 0.8888888f, 1.3333333f, 0.6666666f;

    std::fesetround(FE_TONEAREST);
    Eigen::MatrixXf result_round = mat_a * mat_b;

    std::fesetround(FE_TOWARDZERO);
    Eigen::MatrixXf result_trunc = mat_a * mat_b;

    std::fesetround(FE_TONEAREST);

    // 결과 비교
    bool all_same = true;
    int diff_count = 0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (result_round(i, j) != result_trunc(i, j)) {
                all_same = false;
                diff_count++;
                if (diff_count <= 3) {  // 처음 3개만 출력
                    std::cout << "Diff at (" << i << "," << j << "): ";
                    print_float_bits("round", result_round(i, j));
                    std::cout << "                    ";
                    print_float_bits("trunc", result_trunc(i, j));
                }
            }
        }
    }

    if (all_same) {
        std::cout << "All elements same: YES (SIMD ignores rounding mode)" << std::endl;
    } else {
        std::cout << "Total different elements: " << diff_count << " out of 16" << std::endl;
        std::cout << "Matrix results different: rounding mode WORKS with SIMD!" << std::endl;
    }
    std::cout << std::endl;

    // 결론
    std::cout << "=== Conclusion ===" << std::endl;
    if (eigen_round == eigen_trunc && all_same) {
        std::cout << "SIMD operations IGNORE fesetround() - need EIGEN_DONT_VECTORIZE" << std::endl;
    } else {
        std::cout << "SIMD operations RESPECT fesetround() - can use fesetround directly!" << std::endl;
    }

    return 0;
}
