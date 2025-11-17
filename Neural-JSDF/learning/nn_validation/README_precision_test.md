# Neural-JSDF Mantissa Precision Comparison

이 디렉토리에는 BF16보다 낮은 mantissa precision을 테스트하는 코드가 포함되어 있습니다.

## 파일 설명

### 1. `validation_precision.py`
- BF16 (7-bit mantissa)부터 2-bit mantissa까지 다양한 precision 레벨을 테스트
- 각 precision 설정에 대해 Neural-JSDF 예측 결과와 ground truth를 비교
- 결과를 `.npz` 파일로 저장

**주요 기능:**
- 7, 6, 5, 4, 3, 2 bit mantissa 설정 테스트
- 각 precision에 대해 5000개 샘플 생성 및 테스트
- NN 예측값과 mesh 기반 ground truth 비교
- 통계 분석 (mean, std, max, median, percentiles)
- Per-link error 분석

### 2. `visualize_precision_comparison.py`
- `validation_precision.py` 실행 결과를 시각화
- 9개의 subplot으로 구성된 상세 분석 차트
- 4개의 subplot으로 구성된 요약 차트

**생성되는 그래프:**
1. Mean Error vs Mantissa Bits
2. Max Error vs Mantissa Bits
3. Error Percentiles (50th, 95th, 99th)
4. Relative Degradation vs BF16 Baseline
5. Inference Time Comparison
6. Error Distribution Histogram
7. Per-Link Error Comparison
8. Cumulative Distribution Function (CDF)
9. Memory-Accuracy Trade-off
10. Summary Statistics Table

## 사용 방법

### 1. 환경 준비
기존 validation 코드와 동일한 Python 환경 필요:
```bash
# PyTorch, NumPy, SciPy, Matplotlib 필요
# 기존 validation_bf16.py가 실행되는 환경과 동일
```

### 2. Precision 테스트 실행
```bash
cd /home/mms-wonchan/git/MPCC_manipulator/Neural-JSDF/learning/nn_validation
python validation_precision.py
```

**예상 실행 시간:** 약 5-10분 (5000 samples × 6 precision levels)

**출력 파일:**
- `data_mantissa_7bit.npz` - BF16 (7-bit mantissa) 결과
- `data_mantissa_6bit.npz` - 6-bit mantissa 결과
- `data_mantissa_5bit.npz` - 5-bit mantissa 결과
- `data_mantissa_4bit.npz` - 4-bit mantissa 결과
- `data_mantissa_3bit.npz` - 3-bit mantissa 결과
- `data_mantissa_2bit.npz` - 2-bit mantissa 결과
- `precision_comparison_summary.npz` - 전체 요약 통계

### 3. 결과 시각화
```bash
python visualize_precision_comparison.py
```

**출력 파일:**
- `precision_comparison_analysis.png` - 9개 subplot 상세 분석
- `precision_comparison_summary.png` - 4개 subplot 요약 분석

## 구현 세부 사항

### Mantissa Quantization 방식

BF16 (7-bit mantissa)보다 낮은 precision은 직접 구현:

```python
def quantize_mantissa(self, tensor, mantissa_bits):
    """
    Float32 기반으로 mantissa bit 수를 제한하여 lower precision 에뮬레이션

    IEEE 754 Float32: sign(1) + exponent(8) + mantissa(23)
    목표: mantissa를 mantissa_bits만큼만 사용
    """
    bits_to_zero = 23 - mantissa_bits
    mask = 0xFFFFFFFF << bits_to_zero
    # 하위 비트를 0으로 마스킹하여 precision 감소
```

### Precision 레벨

| Precision | Exponent Bits | Mantissa Bits | Total Bits | 구현 방식 |
|-----------|---------------|---------------|------------|----------|
| BF16      | 8             | 7             | 16         | torch.bfloat16 |
| Custom-6  | 8             | 6             | 15         | Manual quantization |
| Custom-5  | 8             | 5             | 14         | Manual quantization |
| Custom-4  | 8             | 4             | 13         | Manual quantization |
| Custom-3  | 8             | 3             | 12         | Manual quantization |
| Custom-2  | 8             | 2             | 11         | Manual quantization |

### 측정 메트릭

1. **Central Tendency Errors**
   - Mean L1 Error
   - Median Error
   - Standard Deviation

2. **Tail Errors**
   - Max Error
   - 95th Percentile
   - 99th Percentile

3. **Relative Metrics**
   - Degradation vs BF16 baseline (%)
   - Per-link error analysis

4. **Performance**
   - Inference time per precision level

## 예상 결과

BF16 대비 낮은 mantissa precision의 영향:
- **7-bit (BF16)**: Baseline
- **6-bit**: ~0-5% degradation (negligible)
- **5-bit**: ~5-10% degradation (acceptable)
- **4-bit**: ~10-20% degradation (moderate)
- **3-bit**: ~20-40% degradation (significant)
- **2-bit**: >40% degradation (severe)

## 참고 사항

- 모델: `sdf_256x5_mesh_50000.pt` (epoch 1,996 checkpoint)
- Test samples: 5000 per precision level
- Joint space: Franka Panda 7-DOF + 3D query point (10D input)
- Output: 9 link distances (9D output)
- Ground truth: Point-to-mesh distance calculation

## 트러블슈팅

### PyTorch 없음 에러
```
ModuleNotFoundError: No module named 'torch'
```
→ 기존 validation_bf16.py 실행 환경 확인 필요

### 메모리 부족
→ `n_samples` 파라미터를 5000에서 1000으로 감소

### 실행 시간 너무 김
→ `mantissa_configs`에서 일부 precision level 제거
