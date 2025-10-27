# Scaling에서 FlexFloat 제외 수정사항

## 문제 상황

FlexFloat ON 모드에서 `osqp_setup()` 단계의 `scale_data()` 함수가 bounds 배열(`l`, `u`)을 스케일링할 때 FlexFloat 양자화가 적용되어 매우 작은 값들이 손상되었습니다.

**에러**:
```
ERROR in validate_data: Lower bound at index 142 is greater than upper bound: 1.5967e-314 > 5.3568e-315
```

## 해결 방법

Scaling 및 Unscaling 연산 중에는 FlexFloat를 비활성화하여 full double precision을 유지합니다.

## 수정된 파일들

### 1. `include/private/flexfloat_wrapper.h`
- 전역 플래그 `g_disable_flexfloat_for_scaling` 선언 추가
- `flexfloat_from_double()` 함수에 플래그 체크 로직 추가

**변경 내용**:
```c
/* Global flag to disable FlexFloat during scaling operations */
extern int g_disable_flexfloat_for_scaling;

static inline flexfloat_t flexfloat_from_double(double x) {
    // ...
    /* Bypass FlexFloat during scaling operations to preserve precision */
    if (g_disable_flexfloat_for_scaling) {
        result.value = x;
        result.bits = 0;
        return result;
    }
    // ... 나머지 양자화 로직
}
```

### 2. `src/flexfloat_scaling_control.c` (새 파일)
- 전역 플래그 정의
- 기본값: 0 (FlexFloat 활성화)

```c
#ifdef OSQP_USE_FLEXFLOAT
int g_disable_flexfloat_for_scaling = 0;
#endif
```

### 3. `src/CMakeLists.txt`
- `flexfloat_scaling_control.c`를 빌드에 추가

```cmake
if(OSQP_USE_FLEXFLOAT)
  target_sources(OSQPLIB PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/flexfloat_scaling_control.c")
endif()
```

### 4. `src/scaling.c`
- 외부 변수 선언 추가
- 세 함수에서 플래그 제어:
  1. `scale_data()` - 시작 시 비활성화, 종료 시 활성화
  2. `unscale_data()` - 시작 시 비활성화, 종료 시 활성화
  3. `unscale_solution()` - 시작 시 비활성화, 종료 시 활성화

**변경 내용**:
```c
#ifdef OSQP_USE_FLEXFLOAT
extern int g_disable_flexfloat_for_scaling;
#endif

OSQPInt scale_data(OSQPSolver* solver) {
    // ...
    #ifdef OSQP_USE_FLEXFLOAT
    g_disable_flexfloat_for_scaling = 1;  // Scaling 시작
    #endif

    // ... scaling 연산들

    #ifdef OSQP_USE_FLEXFLOAT
    g_disable_flexfloat_for_scaling = 0;  // Scaling 종료
    #endif
    return 0;
}
```

## 동작 원리

### FlexFloat 활성화 상태 (플래그 = 0)
- **ADMM 반복 루프**: FlexFloat 양자화 적용
- 벡터 연산: `OSQPVectorf_mult_scalar()`, `OSQPVectorf_plus()`, `OSQPVectorf_add_scaled()` 등

### FlexFloat 비활성화 상태 (플래그 = 1)
- **Scaling/Unscaling 연산**: Full double precision 유지
- Bounds 배열(`l`, `u`)의 작은 값들이 손상되지 않음

## 실행 흐름

```
osqp_setup() 호출
  ↓
validate_data() ✅ 통과
  ↓
데이터 복사
  ↓
IF (scaling enabled):
  scale_data() 호출
    ↓
    g_disable_flexfloat_for_scaling = 1  ← FlexFloat OFF
    ↓
    OSQPVectorf_ew_prod(l, l, E)  ← Full double precision
    OSQPVectorf_ew_prod(u, u, E)  ← Full double precision
    ↓
    g_disable_flexfloat_for_scaling = 0  ← FlexFloat ON
  ↓
init_linsys_solver()
  ↓
  ✅ Bounds가 손상되지 않음!
```

## 테스트 방법

```bash
# 빌드
cd /home/mms-wonchan/git/MPCC_manipulator
./AllBuild.sh ON

# 테스트 실행
./test

# 예상 결과: validate_data 에러 없이 정상 실행
```

## 주의사항

- FlexFloat는 ADMM 반복 루프에서는 여전히 활성화됩니다
- Scaling/Unscaling만 full precision으로 처리됩니다
- 이 방식으로 numerical stability를 유지하면서 ADMM에서 FlexFloat 테스트가 가능합니다

## 참고

- 원래 목표: ADMM 반복에서 reduced precision 테스트
- 문제: Setup 단계에서도 FlexFloat가 적용되어 bounds 손상
- 해결: Setup/Scaling에서만 FlexFloat 제외, ADMM은 유지
