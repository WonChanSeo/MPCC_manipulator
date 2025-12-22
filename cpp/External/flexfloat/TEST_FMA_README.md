# FlexFloat FMA Test

## WSL에서 빌드 및 실행

```bash
# 디렉토리 이동
cd /mnt/c/Users/MMS_Wonchan/Desktop/Git/MPCC_manipulator/cpp/External/flexfloat

# 컴파일
gcc -I include -O2 -o test_fma test_fma.c src/flexfloat.c -lm

# 실행
./test_fma testdata
```

## 기존 라이브러리 사용 시

```bash
cd /mnt/c/Users/MMS_Wonchan/Desktop/Git/MPCC_manipulator/cpp/External/flexfloat

# 기존 빌드된 libflexfloat.a 링크
gcc -I include -O2 -o test_fma test_fma.c -L build -lflexfloat -lm

./test_fma testdata
```

## 테스트 데이터

- `testdata/fma_din_A.hex` - FMA 입력 A
- `testdata/fma_din_B.hex` - FMA 입력 B
- `testdata/fma_din_C.hex` - FMA 입력 C
- `testdata/fma_result.hex` - 기대 결과 (레퍼런스)

## 설정 확인

현재 `include/flexfloat_config.h` 설정:
- `FLEXFLOAT_ON_DOUBLE` - 백엔드로 double 사용
- `FLEXFLOAT_ROUNDING` 활성화 (NO_ROUNDING이 undef)

## 출력 예시

```
Loading test data from: testdata
Loaded 16 test cases

========================================
FlexFloat FMA Test (E8M23)
========================================
Total test cases: 16

========================================
Results: 16 passed, 0 failed out of 16
========================================

All tests PASSED!
```

실패 시 상세 정보 출력:
```
[FAIL] Test 0:
  A        = 0x3F9DF3B6 (1.234e+00)
  B        = 0x40B5B22D (5.678e+00)
  C        = 0x41102DE0 (9.012e+00)
  Expected = 0x4180248f
  Actual   = 0x41802490
  std fmaf = 0x4180248f
```
## 실제 출력

wonchan@DESKTOP-H8CSEUS:/mnt/c/Users/MMS_Wonchan/Desktop/Git/MPCC_manipulator/cpp/External/flexfloat$ ./test_fma testdata
Loading test data from: testdata

Loaded 16 test cases

========================================
FlexFloat FMA Test (E8M23)
========================================
Total test cases: 16

========================================
Results: 16 passed, 0 failed out of 16
========================================

All tests PASSED!