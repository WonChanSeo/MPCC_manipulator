값이 완전히 동일하네요 — relu boundary는 원인이 아닙니다. 더 깊이 분석하겠습니다.

C++ golden 코드를 다시 확인해보니 batch version의 nerf_jac 정밀도 불일치를 발견했습니다.

Forward pass 입력 (batch, line 596-597):


mlp_.batch_input_nerf.middleRows(...) = mlp_.batch_input.array().sin().matrix();  // float sin
mlp_.batch_input_nerf.bottomRows(...)  = mlp_.batch_input.array().cos().matrix();  // float cos
→ sin(float(x)), cos(float(x)) — float precision

Jacobian nerf_jac (batch, line 653-654):


nerf_jac.middleRows(...).diagonal() = inputs.col(i).array().cos().cast<float>();  // double cos → cast
nerf_jac.bottomRows(...).diagonal()  = (-inputs.col(i).array().sin()).cast<float>(); // double sin → cast
→ (float)cos(double(x)), (float)(-sin(double(x))) — double precision 후 float 변환

inputs은 Eigen::MatrixXd(double)이므로, cos(double(x))를 float로 cast한 것과 cosf(float(x))는 다를 수 있습니다.

Golden INPUT에는 sin(float(x)), cos(float(x))가 저장되고, HW는 이 값을 input_buf에서 읽어 direction vector를 만듭니다. 하지만 golden JACOBIAN의 nerf_jac은 cos(double(x)) cast로 만들어졌습니다.

Joint 6만 영향받는 이유: x[6]의 특정 값에서 cos(double(x[6]))→float와 cosf(float(x[6]))의 FP32 결과가 다르고, joints 0-5에서는 우연히 같은 결과를 냅니다.

이건 C++ golden 모델의 버그입니다. HW는 정확합니다. C++ 수정:


// 기존 (line 653-654):
nerf_jac.middleRows(mlp_.n_input, mlp_.n_input).diagonal() = inputs.col(i).array().cos().cast<float>();
nerf_jac.bottomRows(mlp_.n_input).diagonal() = (-inputs.col(i).array().sin()).cast<float>();

// 수정: float input에서 sin/cos 계산 (forward pass와 동일)
Eigen::VectorXf input_f_i = inputs.col(i).cast<float>();
nerf_jac.middleRows(mlp_.n_input, mlp_.n_input).diagonal() = input_f_i.array().cos();
nerf_jac.bottomRows(mlp_.n_input).diagonal() = -input_f_i.array().sin();