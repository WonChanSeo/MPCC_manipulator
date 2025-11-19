# MPCC Build Options Guide

이 문서는 Neural Network 추론 정밀도 설정을 위한 빌드 옵션 사용 방법을 설명합니다.

## 🚀 빠른 시작

### 방법 1: 빌드 스크립트 사용 (권장)

```bash
cd /path/to/cpp
./build_with_options.sh [preset]
```

**사용 가능한 Preset:**
- `default` - 기본 빌드 (Eigen::bfloat16 + FP32 accumulators)
- `fp32` - FlexFloat FP32 (E8M23)
- `bf16` - FlexFloat BF16 (E8M7)
- `fp16` - FlexFloat FP16 (E5M10)
- `e8m6` - FlexFloat E8M6
- `e8m5` - FlexFloat E8M5
- `e8m4` - FlexFloat E8M4
- `custom` - 사용자 정의 정밀도
- `clean` - 빌드 디렉토리 정리

**예시:**
```bash
# BF16으로 빌드
./build_with_options.sh bf16

# 기본 설정으로 빌드
./build_with_options.sh default

# 빌드 정리
./build_with_options.sh clean
```

### 방법 2: CMake 직접 사용

```bash
cd /path/to/cpp
mkdir -p build
cd build

# BF16으로 빌드
cmake .. -DNN_USE_FLEXFLOAT=ON -DNN_FF_EXPONENT_BITS=8 -DNN_FF_MANTISSA_BITS=7

# 기본 빌드 (FlexFloat OFF)
cmake .. -DNN_USE_FLEXFLOAT=OFF

# 빌드 실행
make -j$(nproc)
```

## 📊 정밀도 프리셋 상세 설명

### 1. Default (기본)
```bash
./build_with_options.sh default
```
- **FlexFloat**: OFF
- **구현**: Eigen::bfloat16
- **특징**:
  - Weights/activations: BF16
  - Accumulators: FP32
  - 하드웨어 최적화된 구현
  - 가장 빠른 성능

### 2. FP32 (E8M23)
```bash
./build_with_options.sh fp32
```
- **Exponent**: 8 bits
- **Mantissa**: 23 bits
- **특징**:
  - IEEE 754 단정밀도
  - FlexFloat로 구현
  - FMA-level quantization 테스트용
  - 높은 정확도

### 3. BF16 (E8M7)
```bash
./build_with_options.sh bf16
```
- **Exponent**: 8 bits
- **Mantissa**: 7 bits
- **특징**:
  - Google Brain Float 16
  - FP32와 같은 exponent range
  - 메모리 효율적
  - ML 학습/추론에 적합

### 4. FP16 (E5M10)
```bash
./build_with_options.sh fp16
```
- **Exponent**: 5 bits
- **Mantissa**: 10 bits
- **특징**:
  - IEEE 754 반정밀도
  - 작은 exponent range
  - 높은 mantissa 정밀도

### 5. E8M6
```bash
./build_with_options.sh e8m6
```
- **Exponent**: 8 bits
- **Mantissa**: 6 bits
- **특징**: BF16보다 1bit 낮은 정밀도

### 6. E8M5
```bash
./build_with_options.sh e8m5
```
- **Exponent**: 8 bits
- **Mantissa**: 5 bits

### 7. E8M4
```bash
./build_with_options.sh e8m4
```
- **Exponent**: 8 bits
- **Mantissa**: 4 bits
- **특징**: 최소 정밀도, 테스트용

### 8. Custom (사용자 정의)
```bash
./build_with_options.sh custom
```
- 대화형으로 exponent/mantissa bits 입력
- 1-8 exponent bits
- 1-23 mantissa bits

## 🔧 CMake 옵션 상세

### NN_USE_FLEXFLOAT
FlexFloat 라이브러리 사용 여부

```cmake
-DNN_USE_FLEXFLOAT=ON   # FlexFloat 활성화
-DNN_USE_FLEXFLOAT=OFF  # Eigen::bfloat16 사용 (기본)
```

### NN_FF_EXPONENT_BITS
Exponent 비트 수 (FlexFloat 활성화 시)

```cmake
-DNN_FF_EXPONENT_BITS=8  # 8 bits (FP32, BF16)
-DNN_FF_EXPONENT_BITS=5  # 5 bits (FP16)
```

**범위**: 1-8
**기본값**: 8

### NN_FF_MANTISSA_BITS
Mantissa 비트 수 (FlexFloat 활성화 시)

```cmake
-DNN_FF_MANTISSA_BITS=23  # 23 bits (FP32)
-DNN_FF_MANTISSA_BITS=7   # 7 bits (BF16)
-DNN_FF_MANTISSA_BITS=10  # 10 bits (FP16)
```

**범위**: 1-23
**기본값**: 23

## 📈 성능 vs 정확도 트레이드오프

| 설정 | 정확도 | 속도 | 메모리 | 용도 |
|------|--------|------|--------|------|
| Default (Eigen BF16) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **프로덕션** |
| FP32 (E8M23) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 검증/테스트 |
| BF16 (E8M7) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **ML 추론** |
| FP16 (E5M10) | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 모바일/임베디드 |
| E8M6 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 실험용 |
| E8M5 | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 실험용 |
| E8M4 | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 최소 정밀도 테스트 |

## 🔍 동작 원리

### Default 모드 (FlexFloat OFF)
```cpp
// Eigen::bfloat16 사용
Eigen::Matrix<Eigen::bfloat16, ...> weights;
Eigen::Matrix<Eigen::bfloat16, ...> activations;

// Matmul에서 자동으로 FP32 accumulator 사용
output = weights * input;  // 내부적으로 FP32 accumulation
```

### FlexFloat 모드 (FlexFloat ON)
```cpp
#ifdef NN_USE_FLEXFLOAT
// 각 FMA 연산마다 quantization
for (int k = 0; k < n; ++k) {
    ff_init_float(&ff_w, (float)weight(i, k), NN_FF_DESC);
    ff_init_float(&ff_x, (float)input(k, j), NN_FF_DESC);

    // FMA with quantization
    ff_fma(&ff_sum, &ff_w, &ff_x, &ff_sum);
    // 결과가 E{exp}M{mant} 정밀도로 quantize됨
}
#endif
```

## 📝 사용 예시

### 시나리오 1: Python과 동일한 결과 얻기 (검증용)

```bash
# Python FMA validation은 E8M23 (FP32) 사용
./build_with_options.sh fp32
cd build
./test_validation_inference

# Python과 비교
cd ../../Neural-JSDF/learning/nn_validation
python3 compare_cpp_python.py
```

**예상 결과**: Max difference < 1e-5

### 시나리오 2: BF16 성능 테스트

```bash
# FlexFloat BF16
./build_with_options.sh bf16
cd build
./test_validation_inference

# vs Default Eigen BF16
./build_with_options.sh default
cd build
./test_validation_inference
```

**비교**: FlexFloat는 모든 FMA를 quantize, Eigen은 FP32 accumulator 사용

### 시나리오 3: 최소 정밀도 테스트

```bash
# E8M4로 빌드
./build_with_options.sh e8m4
cd build
./test_validation_inference
```

**예상**: 큰 오차, 하지만 여전히 동작 가능 여부 확인

## 🛠️ 문제 해결

### CMake 캐시 문제
```bash
# 빌드 디렉토리 완전 정리
./build_with_options.sh clean

# 다시 빌드
./build_with_options.sh bf16
```

### 링크 에러
```bash
# FlexFloat 소스 파일 확인
ls External/flexfloat/src/flexfloat.c

# 존재하지 않으면 submodule 업데이트
git submodule update --init --recursive
```

### OSQP 에러 (낮은 정밀도 사용 시)
```
ERROR in LDL_factor: Error in KKT matrix LDL factorization
The problem seems to be non-convex
```

**원인**: E8M5 이하 정밀도는 OSQP solver에 부족
**해결**: E8M6 이상 사용 또는 Default 모드 사용

## 📚 참고 자료

- **FlexFloat 라이브러리**: `External/flexfloat/`
- **Python 검증 스크립트**: `Neural-JSDF/learning/nn_validation/`
- **C++ 구현**: `src/Constraints/EnvCollision/EnvCollisionModel.cpp`
- **빌드 설정**: `CMakeLists.txt`

## 🔗 관련 문서

- [README_CPP_VALIDATION.md](../Neural-JSDF/learning/nn_validation/README_CPP_VALIDATION.md) - C++/Python 검증 가이드
- [flexfloat_config.h](External/flexfloat/include/flexfloat_config.h) - FlexFloat 설정
