#include "Constraints/EnvCollision/EnvCollisionModel.h"
#include <chrono>

#ifdef NN_USE_FLEXFLOAT
#include <flexfloat.h>
#endif

#ifndef NN_FF_exponent_bits
#define NN_FF_exponent_bits 8
#endif
#ifndef NN_FF_mantissa_bits
#define NN_FF_mantissa_bits 7
#endif
#define NN_FF_DESC ((flexfloat_desc_t){NN_FF_exponent_bits, NN_FF_mantissa_bits})

namespace mpcc
{
    EnvCollNNmodel::EnvCollNNmodel()
    {
        file_path_ = pkg_path + "NNmodel/env/parameter_scaled_bf16/";
    }

    EnvCollNNmodel::EnvCollNNmodel(const std::string & file_path)
    {
        file_path_ = file_path;
    }

    EnvCollNNmodel::~EnvCollNNmodel()
    {
        std::cout<<"NN model terminate" <<std::endl;
    }

    void EnvCollNNmodel::readWeightFile(int weight_num)
    {
        if (!mlp_.weight_files[weight_num].is_open())
        {
            std::cout << "Can not find the file: " << mlp_.w_path[weight_num] << std::endl;
        }
        for (int i = 0; i < mlp_.weight[weight_num].rows(); i++)
        {
            for (int j = 0; j < mlp_.weight[weight_num].cols(); j++)
            {
                float temp_value;
                mlp_.weight_files[weight_num] >> temp_value;
                mlp_.weight[weight_num](i, j) = temp_value;  // bf16 precision 값을 float으로 직접 저장
            }
        }
        mlp_.weight_files[weight_num].close();

        if (mlp_.loadweightfile_verbose == true)
        {
            std::cout << "weight_" << weight_num << ": \n"
                << mlp_.weight[weight_num] <<std::endl;
        }
    }

    void EnvCollNNmodel::readBiasFile(int bias_num)
    {
        if (!mlp_.bias_files[bias_num].is_open())
        {
            std::cout << "Can not find the file: " << mlp_.b_path[bias_num] << std::endl;
        }
        for (int i = 0; i < mlp_.bias[bias_num].rows(); i++)
        {
            float temp_value;
            mlp_.bias_files[bias_num] >> temp_value;
            mlp_.bias[bias_num](i) = temp_value;  // bf16 precision 값을 float으로 직접 저장
        }
        mlp_.bias_files[bias_num].close();

        if (mlp_.loadbiasfile_verbose == true)
        {
            std::cout << "bias_" << bias_num - mlp_.n_layer << ": \n"
                << mlp_.bias[bias_num] << std::endl;
        }
    }

    void EnvCollNNmodel::loadNetwork()
    {
        for (int i = 0; i < mlp_.n_layer; i++)
        {
            mlp_.w_path[i] = file_path_ + "weight_" + std::to_string(i) + ".txt";
            mlp_.b_path[i] = file_path_ + "bias_" + std::to_string(i) + ".txt";

            mlp_.weight_files[i].open(mlp_.w_path[i], std::ios::in);
            mlp_.bias_files[i].open(mlp_.b_path[i], std::ios::in);

            readWeightFile(i);
            readBiasFile(i);
        }
        // Note: Last layer weights/biases are pre-scaled by 0.01 in parameter_scaled/ folder
        // This absorbs the cm->m unit conversion into the network weights
    }

    // void EnvCollNNmodel::initializeNetwork(int n_input, int n_output, Eigen::VectorXd n_hidden, bool is_nerf)
    // {
    //     mlp_.is_nerf = is_nerf;
    //     mlp_.n_input = n_input;
    //     mlp_.n_output = n_output;
    //     mlp_.n_hidden = n_hidden;
    //     mlp_.n_layer = n_hidden.rows() + 1; // hiden layers + output layer

    //     mlp_.weight.resize(mlp_.n_layer);
    //     mlp_.bias.resize(mlp_.n_layer);
    //     mlp_.hidden.resize(mlp_.n_layer - 1);
    //     mlp_.hidden_derivative.resize(mlp_.n_layer - 1);

    //     mlp_.w_path.resize(mlp_.n_layer);
    //     mlp_.b_path.resize(mlp_.n_layer); 
    //     mlp_.weight_files.resize(mlp_.n_layer);
    //     mlp_.bias_files.resize(mlp_.n_layer); 

    //     //parameters resize
    //     for (int i = 0; i < mlp_.n_layer; i++)
    //     {
    //         if (i == 0)
    //         {
    //             if(mlp_.is_nerf) 
    //             {
    //                 mlp_.weight[i].setZero(mlp_.n_hidden(i), 3 * mlp_.n_input);
    //                 mlp_.hidden_derivative[i].setZero(mlp_.n_hidden(i), 3 * mlp_.n_input);
    //             }
    //             else
    //             {
    //                 mlp_.weight[i].setZero(mlp_.n_hidden(i), mlp_.n_input);
    //                 mlp_.hidden_derivative[i].setZero(mlp_.n_hidden(i), mlp_.n_input);
    //             }
    //             mlp_.bias[i].setZero(mlp_.n_hidden(i));
    //             mlp_.hidden[i].setZero(mlp_.n_hidden(i));
    //         }
    //         else if (i == mlp_.n_layer - 1)
    //         {
    //             mlp_.weight[i].setZero(mlp_.n_output, mlp_.n_hidden(i - 1));
    //             mlp_.bias[i].setZero(mlp_.n_output);
    //         }
    //         else
    //         {
    //             mlp_.weight[i].setZero(mlp_.n_hidden(i), mlp_.n_hidden(i - 1));
    //             mlp_.bias[i].setZero(mlp_.n_hidden(i));
    //             mlp_.hidden[i].setZero(mlp_.n_hidden(i));
    //             mlp_.hidden_derivative[i].setZero(mlp_.n_hidden(i), mlp_.n_hidden(i - 1));
    //         }
    //     }
    //     //input output resize
    //     mlp_.input.resize(mlp_.n_input);
    //     mlp_.input_nerf.resize(3 * mlp_.n_input);
    //     mlp_.output.resize(mlp_.n_output);
    //     mlp_.output_derivative.setZero(mlp_.n_output, mlp_.n_input);
    // }

    // --- initializeNetwork 함수 수정 ---
    void EnvCollNNmodel::initializeNetwork(int n_input, int n_output, Eigen::VectorXd n_hidden, bool is_nerf)
    {
        mlp_.is_nerf = is_nerf;
        mlp_.n_input = n_input;
        mlp_.n_output = n_output;
        mlp_.n_hidden = n_hidden;
        mlp_.n_layer = n_hidden.rows() + 1; // hiden layers + output layer

        mlp_.weight.resize(mlp_.n_layer);
        mlp_.bias.resize(mlp_.n_layer);
        mlp_.hidden.resize(mlp_.n_layer - 1);
        mlp_.hidden_derivative.resize(mlp_.n_layer - 1);

        // 배치 추론용 변수 크기 할당 (추가)
        mlp_.batch_hidden.resize(mlp_.n_layer - 1);

        mlp_.w_path.resize(mlp_.n_layer);
        mlp_.b_path.resize(mlp_.n_layer);
        mlp_.weight_files.resize(mlp_.n_layer);
        mlp_.bias_files.resize(mlp_.n_layer);

        // parameters resize (bfloat16 precision)
        for (int i = 0; i < mlp_.n_layer; i++)
        {
            if (i == 0)
            {
                if(mlp_.is_nerf)
                {
                    mlp_.weight[i].setZero(mlp_.n_hidden(i), 3 * mlp_.n_input);
                    mlp_.hidden_derivative[i].setZero(mlp_.n_hidden(i), 3 * mlp_.n_input);
                }
                else
                {
                    mlp_.weight[i].setZero(mlp_.n_hidden(i), mlp_.n_input);
                    mlp_.hidden_derivative[i].setZero(mlp_.n_hidden(i), mlp_.n_input);
                }
                mlp_.bias[i].setZero(mlp_.n_hidden(i));
                mlp_.hidden[i].setZero(mlp_.n_hidden(i));
            }
            else if (i == mlp_.n_layer - 1)
            {
                mlp_.weight[i].setZero(mlp_.n_output, mlp_.n_hidden(i - 1));
                mlp_.bias[i].setZero(mlp_.n_output);
            }
            else
            {
                mlp_.weight[i].setZero(mlp_.n_hidden(i), mlp_.n_hidden(i - 1));
                mlp_.bias[i].setZero(mlp_.n_hidden(i));
                mlp_.hidden[i].setZero(mlp_.n_hidden(i));
                mlp_.hidden_derivative[i].setZero(mlp_.n_hidden(i), mlp_.n_hidden(i - 1));
            }
        }
        // input output resize (bfloat16 precision)
        mlp_.input.resize(mlp_.n_input);
        mlp_.input_nerf.resize(3 * mlp_.n_input);
        mlp_.output.resize(mlp_.n_output);
        mlp_.output_derivative.setZero(mlp_.n_output, mlp_.n_input);

        // --- 배치 추론용 변수 사전 할당 (성능 최적화) ---
        mlp_.pre_activations.resize(mlp_.n_layer - 1);
        mlp_.final_min_output.resize(mlp_.n_output);
        mlp_.final_min_jacobian.resize(mlp_.n_output, mlp_.n_input);

        // --- 자코비안 계산용 임시 변수 사전 할당 (루프 내 동적 할당 제거) ---
        if (mlp_.is_nerf) {
            mlp_.nerf_jac.resize(3 * mlp_.n_input, mlp_.n_input);
            mlp_.temp_derivative.resize(static_cast<int>(mlp_.n_hidden(0)), mlp_.n_input);
        } else {
            mlp_.temp_derivative.resize(static_cast<int>(mlp_.n_hidden(0)), mlp_.n_input);
        }

        // 각 레이어별 ReLU 미분 벡터 사전 할당
        mlp_.relu_derivatives.resize(mlp_.n_layer - 1);
        for (int i = 0; i < mlp_.n_layer - 1; i++) {
            mlp_.relu_derivatives[i].resize(static_cast<int>(mlp_.n_hidden(i)));
        }
    }

    void EnvCollNNmodel::setNeuralNetwork(int n_input, int n_output, Eigen::VectorXd n_hidden, bool is_nerf)
    {
        initializeNetwork(n_input, n_output, n_hidden, is_nerf);
        loadNetwork();
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> EnvCollNNmodel::calculateMlpOutput(Eigen::VectorXd input, bool time_verbose)
    {
        // FP32 연산용 임시 변수들
        Eigen::VectorXf input_f = input.cast<float>();
        Eigen::VectorXf input_nerf_f;
        std::vector<Eigen::VectorXf> hidden_f(mlp_.n_layer - 1);
        std::vector<Eigen::MatrixXf> hidden_derivative_f(mlp_.n_layer - 1);
        Eigen::VectorXf output_f;
        Eigen::MatrixXf output_derivative_f;

        if (mlp_.is_nerf)
        {
            input_nerf_f.resize(3 * mlp_.n_input);
            Eigen::VectorXf sinInput = input_f.array().sin();
            Eigen::VectorXf cosInput = input_f.array().cos();

            input_nerf_f.segment(0 * mlp_.n_input, mlp_.n_input) = input_f;
            input_nerf_f.segment(1 * mlp_.n_input, mlp_.n_input) = sinInput;
            input_nerf_f.segment(2 * mlp_.n_input, mlp_.n_input) = cosInput;
        }

        // weight/bias는 이미 float 타입으로 저장됨 (bf16 precision 값)
        // 별도 캐스팅 불필요, 직접 참조

        Eigen::MatrixXf temp_derivative_f;
        for (int layer = 0; layer < mlp_.n_layer; layer++)
        {
            if (layer == 0) // input layer
            {
                hidden_f[0].resize(static_cast<int>(mlp_.n_hidden(0)));
                hidden_derivative_f[0].resize(static_cast<int>(mlp_.n_hidden(0)), mlp_.is_nerf ? 3 * mlp_.n_input : mlp_.n_input);

                if (mlp_.is_nerf) hidden_f[0] = mlp_.weight[0] * input_nerf_f + mlp_.bias[0];
                else              hidden_f[0] = mlp_.weight[0] * input_f + mlp_.bias[0];

                for (int h = 0; h < mlp_.n_hidden(layer); h++)
                {
                    float relu_deriv = (hidden_f[0](h) > 0.0f) ? 1.0f : 0.0f;
                    hidden_derivative_f[0].row(h) = relu_deriv * mlp_.weight[0].row(h);
                    hidden_f[0](h) = std::max(0.0f, hidden_f[0](h));
                }

                if (mlp_.is_nerf)
                {
                    Eigen::MatrixXf nerf_jac_f;
                    nerf_jac_f.setZero(3 * mlp_.n_input, mlp_.n_input);
                    nerf_jac_f.block(0 * mlp_.n_input, 0, mlp_.n_input, mlp_.n_input).setIdentity();
                    nerf_jac_f.block(1 * mlp_.n_input, 0, mlp_.n_input, mlp_.n_input).diagonal() = input_f.array().cos();
                    nerf_jac_f.block(2 * mlp_.n_input, 0, mlp_.n_input, mlp_.n_input).diagonal() = -input_f.array().sin();

                    temp_derivative_f = hidden_derivative_f[0] * nerf_jac_f;
                }
                else
                {
                    temp_derivative_f = hidden_derivative_f[0];
                }
            }
            else if (layer == mlp_.n_layer - 1) // output layer
            {
                output_f = mlp_.weight[layer] * hidden_f[layer - 1] + mlp_.bias[layer];
                output_derivative_f = mlp_.weight[layer] * temp_derivative_f;
            }
            else // hidden layers
            {
                hidden_f[layer].resize(static_cast<int>(mlp_.n_hidden(layer)));
                hidden_derivative_f[layer].resize(static_cast<int>(mlp_.n_hidden(layer)), static_cast<int>(mlp_.n_hidden(layer - 1)));

                hidden_f[layer] = mlp_.weight[layer] * hidden_f[layer - 1] + mlp_.bias[layer];

                for (int h = 0; h < mlp_.n_hidden(layer); h++)
                {
                    float relu_deriv = (hidden_f[layer](h) > 0.0f) ? 1.0f : 0.0f;
                    hidden_derivative_f[layer].row(h) = relu_deriv * mlp_.weight[layer].row(h);
                    hidden_f[layer](h) = std::max(0.0f, hidden_f[layer](h));
                }

                temp_derivative_f = hidden_derivative_f[layer] * temp_derivative_f;
            }
        }

        // Convert output from float to double for interface compatibility
        return std::make_pair(output_f.cast<double>(), output_derivative_f.cast<double>());
    }

    // =================================================================
    // =========== 배치 추론 함수 (FP32) ============
    // =================================================================

    // 행렬 전체에 ReLU를 적용하는 헬퍼 함수 (FP32)
    Eigen::MatrixXf batch_ReLU_f(const Eigen::MatrixXf& x) {
        return (x.array() > 0.0f).select(x, Eigen::MatrixXf::Zero(x.rows(), x.cols()));
    }

    // 행렬 전체에 ReLU의 미분을 적용하는 헬퍼 함수 (FP32)
    Eigen::MatrixXf batch_ReLU_derivative_f(const Eigen::MatrixXf& x) {
        return (x.array() > 0.0f).select(Eigen::MatrixXf::Ones(x.rows(), x.cols()), Eigen::MatrixXf::Zero(x.rows(), x.cols()));
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> EnvCollNNmodel::calculateMlpOutputBatch(const Eigen::MatrixXd& inputs, bool time_verbose)
    {
        const int batch_size = inputs.cols();
        if (batch_size == 0) {
            return {Eigen::VectorXd(), Eigen::MatrixXd()};
        }

        // --- 배치 크기가 변경된 경우에만 재할당 (성능 최적화) ---
        if (batch_size != mlp_.allocated_batch_size) {
            mlp_.batch_jacobian.resize(batch_size);
            for (int i = 0; i < batch_size; ++i) {
                mlp_.batch_jacobian[i].resize(mlp_.n_output, mlp_.n_input);
            }
            if (mlp_.is_nerf) {
                mlp_.batch_input_nerf.resize(3 * mlp_.n_input, batch_size);
            }
            mlp_.allocated_batch_size = batch_size;
        }

        // ===== 1. 모든 결과와 자코비안을 계산 (효율성을 위해 배치 연산 유지) =====
        // Convert input from double to float for inference
        mlp_.batch_input = inputs.cast<float>();

        const Eigen::MatrixXf* current_input_ptr;
        if (mlp_.is_nerf) {
            mlp_.batch_input_nerf.topRows(mlp_.n_input) = mlp_.batch_input;
            mlp_.batch_input_nerf.middleRows(mlp_.n_input, mlp_.n_input) = mlp_.batch_input.array().sin().matrix();
            mlp_.batch_input_nerf.bottomRows(mlp_.n_input) = mlp_.batch_input.array().cos().matrix();
            current_input_ptr = &mlp_.batch_input_nerf;
        } else {
            current_input_ptr = &mlp_.batch_input;
        }

        for (int layer = 0; layer < mlp_.n_layer; ++layer) {
            if (layer == 0) {
                mlp_.pre_activations[0] = (mlp_.weight[0] * (*current_input_ptr)).colwise() + mlp_.bias[0];
                mlp_.batch_hidden[0] = batch_ReLU_f(mlp_.pre_activations[0]);
            }
            else if (layer == mlp_.n_layer - 1) {
                mlp_.batch_output = (mlp_.weight[layer] * mlp_.batch_hidden[layer - 1]).colwise() + mlp_.bias[layer];
            }
            else {
                mlp_.pre_activations[layer] = (mlp_.weight[layer] * mlp_.batch_hidden[layer - 1]).colwise() + mlp_.bias[layer];
                mlp_.batch_hidden[layer] = batch_ReLU_f(mlp_.pre_activations[layer]);
            }
        }

        // --- 자코비안 계산 ---
        for (int i = 0; i < batch_size; ++i) {
            Eigen::MatrixXf temp_derivative;
            if (mlp_.is_nerf) {
                Eigen::MatrixXf nerf_jac(3 * mlp_.n_input, mlp_.n_input);
                nerf_jac.setZero();
                nerf_jac.topRows(mlp_.n_input).setIdentity();
                nerf_jac.middleRows(mlp_.n_input, mlp_.n_input).diagonal() = inputs.col(i).array().cos().cast<float>();
                nerf_jac.bottomRows(mlp_.n_input).diagonal() = (-inputs.col(i).array().sin()).cast<float>();
                Eigen::MatrixXf relu_deriv_0 = batch_ReLU_derivative_f(mlp_.pre_activations[0].col(i));
                temp_derivative = (relu_deriv_0.asDiagonal() * mlp_.weight[0]) * nerf_jac;
            } else {
                Eigen::MatrixXf relu_deriv_0 = batch_ReLU_derivative_f(mlp_.pre_activations[0].col(i));
                temp_derivative = relu_deriv_0.asDiagonal() * mlp_.weight[0];
            }

            for (int layer = 1; layer < mlp_.n_layer - 1; ++layer) {
                Eigen::MatrixXf relu_deriv = batch_ReLU_derivative_f(mlp_.pre_activations[layer].col(i));
                temp_derivative = (relu_deriv.asDiagonal() * mlp_.weight[layer]) * temp_derivative;
            }

            mlp_.batch_jacobian[i] = mlp_.weight.back() * temp_derivative;
        }

        // ===== 2. 각 행(링크)별 최소값 탐색 및 결과 재구성 =====
        // 사전 할당된 멤버 변수 사용
        for (int i = 0; i < mlp_.n_output; ++i)
        {
            Eigen::Index min_col_for_row;
            mlp_.batch_output.row(i).minCoeff(&min_col_for_row);

            mlp_.final_min_output(i) = mlp_.batch_output(i, min_col_for_row);
            mlp_.final_min_jacobian.row(i) = mlp_.batch_jacobian[min_col_for_row].row(i);
        }

        // Convert output from float to double for interface compatibility
        return std::make_pair(mlp_.final_min_output.cast<double>(), mlp_.final_min_jacobian.cast<double>());
    }

    // ▼▼▼▼▼ 파일 하단에 getter 함수 구현을 추가합니다. ▼▼▼▼▼
    const std::vector<double>& EnvCollNNmodel::getInferenceTimes() const
    {
        return mlp_.inference_times_ms;
    }

    // ▼▼▼▼▼ ReLU deactivation ratio를 계산하여 반환하는 함수 ▼▼▼▼▼
    std::vector<double> EnvCollNNmodel::getReluDeactivationRatios() const
    {
        std::vector<double> ratios;
        if (mlp_.total_sample_count == 0) {
            return ratios; // 아직 inference가 호출되지 않았으면 빈 벡터 반환
        }

        ratios.resize(mlp_.n_layer - 1);
        for (int i = 0; i < mlp_.n_layer - 1; ++i) {
            // 평균 deactivation 개수를 구한 후 비율로 변환
            double avg_deactivated = static_cast<double>(mlp_.deactivated_units_per_layer[i]) / mlp_.total_sample_count;
            double total_units = static_cast<double>(mlp_.total_units_per_layer[i]);
            ratios[i] = avg_deactivated / total_units;
        }
        return ratios;
    }

    // ▼▼▼▼▼ 각 레이어의 총 유닛 수를 반환하는 함수 ▼▼▼▼▼
    std::vector<int> EnvCollNNmodel::getReluTotalUnits() const
    {
        return mlp_.total_units_per_layer;
    }

    // ▼▼▼▼▼ 각 레이어의 평균 deactivated 유닛 수를 반환하는 함수 ▼▼▼▼▼
    std::vector<double> EnvCollNNmodel::getReluAvgDeactivatedCounts() const
    {
        std::vector<double> avg_counts;
        if (mlp_.total_sample_count == 0) {
            return avg_counts;
        }

        avg_counts.resize(mlp_.n_layer - 1);
        for (int i = 0; i < mlp_.n_layer - 1; ++i) {
            avg_counts[i] = static_cast<double>(mlp_.deactivated_units_per_layer[i]) / mlp_.total_sample_count;
        }
        return avg_counts;
    }

    // ▼▼▼▼▼ 각 레이어의 최소 deactivated 유닛 수를 반환하는 함수 ▼▼▼▼▼
    std::vector<int> EnvCollNNmodel::getReluMinDeactivatedCounts() const
    {
        return mlp_.min_deactivated_per_layer;
    }

    // ▼▼▼▼▼ 각 레이어의 최대 deactivated 유닛 수를 반환하는 함수 ▼▼▼▼▼
    std::vector<int> EnvCollNNmodel::getReluMaxDeactivatedCounts() const
    {
        return mlp_.max_deactivated_per_layer;
    }
}



