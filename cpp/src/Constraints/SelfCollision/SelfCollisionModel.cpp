#include "Constraints/SelfCollision/SelfCollisionModel.h"
#include <cstring>

#ifdef NN_USE_TRUNCATE
#include <cfenv>
#endif

namespace mpcc
{
    // ===================================================================
    // =========== Hardware-like Adder Tree Functions ====================
    // ===================================================================

    // Adder tree 합산 (임의 개수) - Self Collision용
    float self_adder_tree_sum(const float* data, int count) {
        if (count == 0) return 0.0f;
        if (count == 1) return data[0];

        std::vector<float> current(data, data + count);

        // 2의 거듭제곱으로 패딩
        size_t next_pow2 = 1;
        while (next_pow2 < static_cast<size_t>(count)) next_pow2 <<= 1;
        current.resize(next_pow2, 0.0f);

        // Adder tree 레벨별 수행
        while (current.size() > 1) {
            std::vector<float> next;
            next.reserve(current.size() / 2);
            for (size_t i = 0; i < current.size(); i += 2) {
                float sum = current[i] + current[i + 1];
#ifdef NN_USE_TRUNCATE
                volatile float truncated = sum;
                float truncated_val = truncated;
                next.push_back(truncated_val);
#else
                next.push_back(sum);
#endif
            }
            current = std::move(next);
        }

        return current[0];
    }

    // Matrix-vector multiplication using adder tree - Self Collision용
    float self_matvec_row_adder_tree(const Eigen::Ref<const Eigen::RowVectorXf>& weight_row,
                                      const Eigen::Ref<const Eigen::VectorXf>& input,
                                      float bias) {
        const int n = weight_row.size();
        const int chunk_size = 128;
        std::vector<float> products(n);

        // Stage 1: 모든 곱셈을 병렬로 수행
        for (int i = 0; i < n; ++i) {
            float prod = weight_row(i) * input(i);
#ifdef NN_USE_TRUNCATE
            volatile float truncated = prod;
            products[i] = truncated;
#else
            products[i] = prod;
#endif
        }

        // Stage 2: 128개씩 chunk로 adder tree 수행 (최대 2개 chunk)
        float chunk_sum_0 = self_adder_tree_sum(products.data(), std::min(n, chunk_size));
        float chunk_sum_1 = (n > chunk_size) ? self_adder_tree_sum(products.data() + chunk_size, n - chunk_size) : 0.0f;

        // Method A: bias + chunk0 -> truncate -> + chunk1 -> truncate
        float result = bias + chunk_sum_0;
#ifdef NN_USE_TRUNCATE
        volatile float truncated_1 = result;
        result = truncated_1;
#endif
        result = result + chunk_sum_1;
#ifdef NN_USE_TRUNCATE
        volatile float truncated_2 = result;
        result = truncated_2;
#endif

        return result;
    }

    SelCollNNmodel::SelCollNNmodel()
    {
        file_path_ = pkg_path + "NNmodel/self/parameter_scaled_bf16/";
    }

    SelCollNNmodel::SelCollNNmodel(const std::string & file_path)
    {
        file_path_ = file_path;
    }

    SelCollNNmodel::~SelCollNNmodel()
    {
        std::cout<<"NN model terminate" <<std::endl;
    }

    void SelCollNNmodel::readWeightFile(int weight_num)
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
                mlp_.weight[weight_num](i, j) = temp_value;
            }
        }
        mlp_.weight_files[weight_num].close();

        if (mlp_.loadweightfile_verbose == true)
        {
            std::cout << "weight_" << weight_num << ": \n"
                << mlp_.weight[weight_num] <<std::endl;
        }
    }

    void SelCollNNmodel::readBiasFile(int bias_num)
    {
        if (!mlp_.bias_files[bias_num].is_open())
        {
            std::cout << "Can not find the file: " << mlp_.b_path[bias_num] << std::endl;
        }
        for (int i = 0; i < mlp_.bias[bias_num].rows(); i++)
        {
            float temp_value;
            mlp_.bias_files[bias_num] >> temp_value;
            mlp_.bias[bias_num](i) = temp_value;
        }
        mlp_.bias_files[bias_num].close();

        if (mlp_.loadbiasfile_verbose == true)
        {
            std::cout << "bias_" << bias_num - mlp_.n_layer << ": \n"
                << mlp_.bias[bias_num] << std::endl;
        }
    }

    void SelCollNNmodel::loadNetwork()
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
    }

    void SelCollNNmodel::initializeNetwork(int n_input, int n_output, Eigen::VectorXd n_hidden, bool is_nerf)
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

        mlp_.w_path.resize(mlp_.n_layer);
        mlp_.b_path.resize(mlp_.n_layer);
        mlp_.weight_files.resize(mlp_.n_layer);
        mlp_.bias_files.resize(mlp_.n_layer);

        //parameters resize (FP32)
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
        //input output resize (FP32)
        mlp_.input.resize(mlp_.n_input);
        mlp_.input_nerf.resize(3 * mlp_.n_input);
        mlp_.output.resize(mlp_.n_output);
        mlp_.output_derivative.setZero(mlp_.n_output, mlp_.n_input);
    }

    void SelCollNNmodel::setNeuralNetwork(int n_input, int n_output, Eigen::VectorXd n_hidden, bool is_nerf)
    {
        initializeNetwork(n_input, n_output, n_hidden, is_nerf);
        loadNetwork();
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> SelCollNNmodel::calculateMlpOutput(Eigen::VectorXd input, bool time_verbose)
    {
#ifdef NN_USE_TRUNCATE
        int old_round = std::fegetround();
        std::fesetround(FE_TOWARDZERO);
#endif

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

        Eigen::MatrixXf temp_derivative_f;
        for (int layer = 0; layer < mlp_.n_layer; layer++)
        {
            if (layer == 0) // input layer
            {
                const Eigen::VectorXf& current_input = mlp_.is_nerf ? input_nerf_f : input_f;
                const int n_neurons = static_cast<int>(mlp_.n_hidden(0));
                const int n_inputs = current_input.size();

                hidden_f[0].resize(n_neurons);
                hidden_derivative_f[0].resize(n_neurons, n_inputs);

                // Adder tree를 사용한 행렬-벡터 곱셈
                for (int h = 0; h < n_neurons; ++h) {
                    hidden_f[0](h) = self_matvec_row_adder_tree(
                        mlp_.weight[0].row(h), current_input, mlp_.bias[0](h));
                }

                for (int h = 0; h < n_neurons; h++)
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

                    // Jacobian도 adder tree로 계산
                    const int jac_rows = hidden_derivative_f[0].rows();
                    const int jac_cols = nerf_jac_f.cols();
                    temp_derivative_f.resize(jac_rows, jac_cols);
                    for (int r = 0; r < jac_rows; ++r) {
                        for (int c = 0; c < jac_cols; ++c) {
                            temp_derivative_f(r, c) = self_matvec_row_adder_tree(
                                hidden_derivative_f[0].row(r), nerf_jac_f.col(c), 0.0f);
                        }
                    }
                }
                else
                {
                    temp_derivative_f = hidden_derivative_f[0];
                }
            }
            else if (layer == mlp_.n_layer - 1) // output layer
            {
                const int n_outputs = mlp_.n_output;
                output_f.resize(n_outputs);

                // Adder tree를 사용한 출력층 계산
                for (int o = 0; o < n_outputs; ++o) {
                    output_f(o) = self_matvec_row_adder_tree(
                        mlp_.weight[layer].row(o), hidden_f[layer - 1], mlp_.bias[layer](o));
                }

                // 출력층 Jacobian 계산 (adder tree 사용)
                const int deriv_rows = mlp_.weight[layer].rows();
                const int deriv_cols = temp_derivative_f.cols();
                output_derivative_f.resize(deriv_rows, deriv_cols);
                for (int r = 0; r < deriv_rows; ++r) {
                    for (int c = 0; c < deriv_cols; ++c) {
                        output_derivative_f(r, c) = self_matvec_row_adder_tree(
                            mlp_.weight[layer].row(r), temp_derivative_f.col(c), 0.0f);
                    }
                }
            }
            else // hidden layers
            {
                const int n_neurons = static_cast<int>(mlp_.n_hidden(layer));
                hidden_f[layer].resize(n_neurons);
                hidden_derivative_f[layer].resize(n_neurons, static_cast<int>(mlp_.n_hidden(layer - 1)));

                // Adder tree를 사용한 은닉층 계산
                for (int h = 0; h < n_neurons; ++h) {
                    hidden_f[layer](h) = self_matvec_row_adder_tree(
                        mlp_.weight[layer].row(h), hidden_f[layer - 1], mlp_.bias[layer](h));
                }

                for (int h = 0; h < n_neurons; h++)
                {
                    float relu_deriv = (hidden_f[layer](h) > 0.0f) ? 1.0f : 0.0f;
                    hidden_derivative_f[layer].row(h) = relu_deriv * mlp_.weight[layer].row(h);
                    hidden_f[layer](h) = std::max(0.0f, hidden_f[layer](h));
                }

                // Jacobian 계산 (adder tree 사용)
                const int jac_rows = hidden_derivative_f[layer].rows();
                const int jac_cols = temp_derivative_f.cols();
                Eigen::MatrixXf new_temp_derivative_f(jac_rows, jac_cols);
                for (int r = 0; r < jac_rows; ++r) {
                    for (int c = 0; c < jac_cols; ++c) {
                        new_temp_derivative_f(r, c) = self_matvec_row_adder_tree(
                            hidden_derivative_f[layer].row(r), temp_derivative_f.col(c), 0.0f);
                    }
                }
                temp_derivative_f = std::move(new_temp_derivative_f);
            }
        }

        // Convert output from float to double for interface compatibility
#ifdef NN_USE_TRUNCATE
        std::fesetround(old_round);
#endif
        return std::make_pair(output_f.cast<double>(), output_derivative_f.cast<double>());
    }
}
