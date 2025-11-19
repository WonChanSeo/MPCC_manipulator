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
        file_path_ = pkg_path + "NNmodel/env/parameter/";
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
            mlp_.bias[bias_num](i) = temp_value;
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
    }

    void EnvCollNNmodel::initializeNetwork(int n_input, int n_output, Eigen::VectorXd n_hidden, bool is_nerf)
    {
        mlp_.is_nerf = is_nerf;
        mlp_.n_input = n_input;
        mlp_.n_output = n_output;
        mlp_.n_hidden = n_hidden;
        mlp_.n_layer = n_hidden.rows() + 1;

        mlp_.weight.resize(mlp_.n_layer);
        mlp_.bias.resize(mlp_.n_layer);
        mlp_.hidden.resize(mlp_.n_layer - 1);
        mlp_.hidden_derivative.resize(mlp_.n_layer - 1);
        mlp_.batch_hidden.resize(mlp_.n_layer - 1);

        mlp_.w_path.resize(mlp_.n_layer);
        mlp_.b_path.resize(mlp_.n_layer);
        mlp_.weight_files.resize(mlp_.n_layer);
        mlp_.bias_files.resize(mlp_.n_layer);

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
        mlp_.input.resize(mlp_.n_input);
        mlp_.input_nerf.resize(3 * mlp_.n_input);
        mlp_.output.resize(mlp_.n_output);
        mlp_.output_derivative.setZero(mlp_.n_output, mlp_.n_input);
    }

    void EnvCollNNmodel::setNeuralNetwork(int n_input, int n_output, Eigen::VectorXd n_hidden, bool is_nerf)
    {
        initializeNetwork(n_input, n_output, n_hidden, is_nerf);
        loadNetwork();
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> EnvCollNNmodel::calculateMlpOutput(Eigen::VectorXd input, bool time_verbose)
    {
        mlp_.input = input.cast<float>();
        if (mlp_.is_nerf)
        {
            printf("mlp is nerf\n");
            VectorXf32 sinInput = input.array().sin().cast<float>();
            VectorXf32 cosInput = input.array().cos().cast<float>();

            mlp_.input_nerf.segment(0 * mlp_.n_input, mlp_.n_input) = mlp_.input;
            mlp_.input_nerf.segment(1 * mlp_.n_input, mlp_.n_input) = sinInput;
            mlp_.input_nerf.segment(2 * mlp_.n_input, mlp_.n_input) = cosInput;
        }

        std::vector<clock_t> start, finish;
        start.resize(3*mlp_.n_layer);
        finish.resize(3*mlp_.n_layer);

        start[3*mlp_.n_layer - 1] = clock();
        MatrixXf32 temp_derivative;
        for (int layer = 0; layer < mlp_.n_layer; layer++)
        {
            if (layer == 0)
            {
                start[0] = clock();
                if (mlp_.is_nerf) mlp_.hidden[0] = mlp_.weight[0] * mlp_.input_nerf + mlp_.bias[0];
                else                mlp_.hidden[0] = mlp_.weight[0] * mlp_.input + mlp_.bias[0];
                finish[0] = clock();

                start[1] = clock();
                for (int h = 0; h < mlp_.n_hidden(layer); h++)
                {
                    mlp_.hidden_derivative[0].row(h) = ReLU_derivative(mlp_.hidden[0](h)) * mlp_.weight[0].row(h);
                    mlp_.hidden[0](h) = ReLU(mlp_.hidden[0](h));
                }
                finish[1] = clock();

                if (mlp_.is_nerf)
                {
                    MatrixXf32 nerf_jac;
                    nerf_jac.setZero(3 * mlp_.n_input, mlp_.n_input);
                    nerf_jac.block(0 * mlp_.n_input, 0, mlp_.n_input, mlp_.n_input).setIdentity();
                    nerf_jac.block(1 * mlp_.n_input, 0, mlp_.n_input, mlp_.n_input).diagonal() = mlp_.input.array().cos();
                    nerf_jac.block(2 * mlp_.n_input, 0, mlp_.n_input, mlp_.n_input).diagonal() = -mlp_.input.array().sin();

                    start[2] = clock();
                    temp_derivative = mlp_.hidden_derivative[0] * nerf_jac;
                    finish[2] = clock();
                }
                else
                {
                    temp_derivative = mlp_.hidden_derivative[0];
                }
            }
            else if (layer == mlp_.n_layer - 1)
            {
                start[layer*3] = clock();
                mlp_.output = mlp_.weight[layer] * mlp_.hidden[layer - 1] + mlp_.bias[layer];
                finish[layer*3] = clock();

                start[layer*3+1] = clock();
                mlp_.output_derivative = mlp_.weight[layer] * temp_derivative;
                finish[layer*3+1] = clock();
            }
            else
            {
                start[layer*3] = clock();
                mlp_.hidden[layer] = mlp_.weight[layer] * mlp_.hidden[layer - 1] + mlp_.bias[layer];
                finish[layer*3] = clock();

                start[layer*3+1] = clock();
                for (int h = 0; h < mlp_.n_hidden(layer); h++)
                {
                    mlp_.hidden_derivative[layer].row(h) = ReLU_derivative(mlp_.hidden[layer](h)) * mlp_.weight[layer].row(h);
                    mlp_.hidden[layer](h) = ReLU(mlp_.hidden[layer](h));
                }
                finish[layer*3+1] = clock();

                start[3*layer+2] = clock();
                temp_derivative = mlp_.hidden_derivative[layer] * temp_derivative;
                finish[3*layer+2] = clock();
            }
        }

        finish[3*mlp_.n_layer - 1] = clock();

        return std::make_pair(mlp_.output.cast<double>(), mlp_.output_derivative.cast<double>());
    }

    // Helper functions for batch operations (FP32 precision)
    MatrixXf32 batch_ReLU(const MatrixXf32& x) {
#ifdef NN_USE_FLEXFLOAT
        MatrixXf32 result = x;
        for (int i = 0; i < x.rows(); ++i) {
            for (int j = 0; j < x.cols(); ++j) {
                flexfloat_t ff_val;
                ff_init_float(&ff_val, x(i, j), NN_FF_DESC);
                if (ff_get_float(&ff_val) < 0.0f) {
                    result(i, j) = 0.0f;
                } else {
                    result(i, j) = ff_get_float(&ff_val);
                }
            }
        }
        return result;
#else
        return (x.array() > 0.0f).select(x, MatrixXf32::Zero(x.rows(), x.cols()));
#endif
    }

    MatrixXf32 batch_ReLU_derivative(const MatrixXf32& x) {
        return (x.array() > 0.0f).select(MatrixXf32::Ones(x.rows(), x.cols()), MatrixXf32::Zero(x.rows(), x.cols()));
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> EnvCollNNmodel::calculateMlpOutputBatch(const Eigen::MatrixXd& inputs, bool time_verbose)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        const int batch_size = inputs.cols();
        if (batch_size == 0) {
            return {Eigen::VectorXd(), Eigen::MatrixXd()};
        }

        // Initialize ReLU deactivation statistics
        if (mlp_.total_inference_count == 0) {
            mlp_.total_units_per_layer.resize(mlp_.n_layer - 1);
            mlp_.deactivated_units_per_layer.resize(mlp_.n_layer - 1, 0);
            mlp_.min_deactivated_per_layer.resize(mlp_.n_layer - 1, std::numeric_limits<int>::max());
            mlp_.max_deactivated_per_layer.resize(mlp_.n_layer - 1, 0);
            for (int i = 0; i < mlp_.n_layer - 1; ++i) {
                mlp_.total_units_per_layer[i] = mlp_.n_hidden(i);
            }
        }
        mlp_.total_inference_count++;
        mlp_.total_sample_count += batch_size;

        // Convert input from double to float for inference
        mlp_.batch_input = inputs.cast<float>();

        const MatrixXf32* current_input_ptr;
        const Eigen::MatrixXd* current_input_ptr_double = &inputs;

        if (mlp_.is_nerf) {
            mlp_.batch_input_nerf.resize(3 * mlp_.n_input, batch_size);
            mlp_.batch_input_nerf.topRows(mlp_.n_input) = mlp_.batch_input;
            mlp_.batch_input_nerf.middleRows(mlp_.n_input, mlp_.n_input) = mlp_.batch_input.array().sin().matrix();
            mlp_.batch_input_nerf.bottomRows(mlp_.n_input) = mlp_.batch_input.array().cos().matrix();
            current_input_ptr = &mlp_.batch_input_nerf;
        } else {
            current_input_ptr = &mlp_.batch_input;
        }

        mlp_.pre_activations.resize(mlp_.n_layer - 1);

        for (int layer = 0; layer < mlp_.n_layer; ++layer) {
            if (layer == 0) {
#ifdef NN_USE_FLEXFLOAT
                const int m = mlp_.weight[0].rows();
                const int n = mlp_.weight[0].cols();
                const int batch = current_input_ptr->cols();
                mlp_.pre_activations[0].resize(m, batch);

                for (int i = 0; i < m; ++i) {
                    for (int j = 0; j < batch; ++j) {
                        flexfloat_t ff_sum, ff_w, ff_x, ff_bias;
                        ff_init_float(&ff_sum, 0.0f, NN_FF_DESC);

                        for (int k = 0; k < n; ++k) {
                            ff_init_float(&ff_w, mlp_.weight[0](i, k), NN_FF_DESC);
                            ff_init_float(&ff_x, (*current_input_ptr)(k, j), NN_FF_DESC);
                            ff_fma(&ff_sum, &ff_w, &ff_x, &ff_sum);
                        }

                        // Add bias
                        ff_init_float(&ff_bias, mlp_.bias[0](i), NN_FF_DESC);
                        flexfloat_t ff_one;
                        ff_init_float(&ff_one, 1.0f, NN_FF_DESC);
                        ff_fma(&ff_sum, &ff_bias, &ff_one, &ff_sum);

                        mlp_.pre_activations[0](i, j) = ff_get_float(&ff_sum);
                    }
                }
#else
                mlp_.pre_activations[0] = (mlp_.weight[0] * (*current_input_ptr)).colwise() + mlp_.bias[0];
#endif
                mlp_.batch_hidden[0] = batch_ReLU(mlp_.pre_activations[0]);
            }
            else if (layer == mlp_.n_layer - 1) {
#ifdef NN_USE_FLEXFLOAT
                const int m = mlp_.weight[layer].rows();
                const int n = mlp_.weight[layer].cols();
                const int batch = mlp_.batch_hidden[layer - 1].cols();
                mlp_.batch_output.resize(m, batch);

                for (int i = 0; i < m; ++i) {
                    for (int j = 0; j < batch; ++j) {
                        flexfloat_t ff_sum, ff_w, ff_x, ff_bias;
                        ff_init_float(&ff_sum, 0.0f, NN_FF_DESC);

                        for (int k = 0; k < n; ++k) {
                            ff_init_float(&ff_w, mlp_.weight[layer](i, k), NN_FF_DESC);
                            ff_init_float(&ff_x, mlp_.batch_hidden[layer - 1](k, j), NN_FF_DESC);
                            ff_fma(&ff_sum, &ff_w, &ff_x, &ff_sum);
                        }

                        // Add bias
                        ff_init_float(&ff_bias, mlp_.bias[layer](i), NN_FF_DESC);
                        flexfloat_t ff_one;
                        ff_init_float(&ff_one, 1.0f, NN_FF_DESC);
                        ff_fma(&ff_sum, &ff_bias, &ff_one, &ff_sum);

                        mlp_.batch_output(i, j) = ff_get_float(&ff_sum);
                    }
                }
#else
                mlp_.batch_output = (mlp_.weight[layer] * mlp_.batch_hidden[layer - 1]).colwise() + mlp_.bias[layer];
#endif
            }
            else {
#ifdef NN_USE_FLEXFLOAT
                const int m = mlp_.weight[layer].rows();
                const int n = mlp_.weight[layer].cols();
                const int batch = mlp_.batch_hidden[layer - 1].cols();
                mlp_.pre_activations[layer].resize(m, batch);

                for (int i = 0; i < m; ++i) {
                    for (int j = 0; j < batch; ++j) {
                        flexfloat_t ff_sum, ff_w, ff_x, ff_bias;
                        ff_init_float(&ff_sum, 0.0f, NN_FF_DESC);

                        for (int k = 0; k < n; ++k) {
                            ff_init_float(&ff_w, mlp_.weight[layer](i, k), NN_FF_DESC);
                            ff_init_float(&ff_x, mlp_.batch_hidden[layer - 1](k, j), NN_FF_DESC);
                            ff_fma(&ff_sum, &ff_w, &ff_x, &ff_sum);
                        }

                        // Add bias
                        ff_init_float(&ff_bias, mlp_.bias[layer](i), NN_FF_DESC);
                        flexfloat_t ff_one;
                        ff_init_float(&ff_one, 1.0f, NN_FF_DESC);
                        ff_fma(&ff_sum, &ff_bias, &ff_one, &ff_sum);

                        mlp_.pre_activations[layer](i, j) = ff_get_float(&ff_sum);
                    }
                }
#else
                mlp_.pre_activations[layer] = (mlp_.weight[layer] * mlp_.batch_hidden[layer - 1]).colwise() + mlp_.bias[layer];
#endif
                mlp_.batch_hidden[layer] = batch_ReLU(mlp_.pre_activations[layer]);
            }
        }

        mlp_.batch_jacobian.resize(batch_size);

        // ReLU deactivation tracking
        std::vector<std::vector<int>> thread_deactivated_counts(batch_size, std::vector<int>(mlp_.n_layer - 1, 0));

        #pragma omp parallel for
        for (int i = 0; i < batch_size; ++i) {
            MatrixXf32 temp_derivative;
            if (mlp_.is_nerf) {
                MatrixXf32 nerf_jac(3 * mlp_.n_input, mlp_.n_input);
                nerf_jac.setZero();
                nerf_jac.topRows(mlp_.n_input).setIdentity();
                nerf_jac.middleRows(mlp_.n_input, mlp_.n_input).diagonal() = inputs.col(i).array().cos().cast<float>();
                nerf_jac.bottomRows(mlp_.n_input).diagonal() = (-inputs.col(i).array().sin()).cast<float>();
                MatrixXf32 relu_deriv_0 = batch_ReLU_derivative(mlp_.pre_activations[0].col(i));

                int deactivated_count_0 = (relu_deriv_0.array() == 0.0f).count();
                thread_deactivated_counts[i][0] = deactivated_count_0;

#ifdef NN_USE_FLEXFLOAT
                MatrixXf32 DW(relu_deriv_0.rows(), mlp_.weight[0].cols());
                for (int r = 0; r < DW.rows(); ++r) {
                    for (int c = 0; c < DW.cols(); ++c) {
                        flexfloat_t ff_result, ff_diag, ff_w, ff_zero;
                        ff_init_float(&ff_zero, 0.0f, NN_FF_DESC);
                        ff_init_float(&ff_diag, relu_deriv_0(r), NN_FF_DESC);
                        ff_init_float(&ff_w, mlp_.weight[0](r, c), NN_FF_DESC);
                        ff_init_float(&ff_result, 0.0f, NN_FF_DESC);
                        ff_fma(&ff_result, &ff_diag, &ff_w, &ff_zero);
                        DW(r, c) = ff_get_float(&ff_result);
                    }
                }
                temp_derivative.resize(DW.rows(), nerf_jac.cols());
                for (int r = 0; r < temp_derivative.rows(); ++r) {
                    for (int c = 0; c < temp_derivative.cols(); ++c) {
                        flexfloat_t ff_sum, ff_a, ff_b;
                        ff_init_float(&ff_sum, 0.0f, NN_FF_DESC);
                        for (int k = 0; k < DW.cols(); ++k) {
                            ff_init_float(&ff_a, DW(r, k), NN_FF_DESC);
                            ff_init_float(&ff_b, nerf_jac(k, c), NN_FF_DESC);
                            ff_fma(&ff_sum, &ff_a, &ff_b, &ff_sum);
                        }
                        temp_derivative(r, c) = ff_get_float(&ff_sum);
                    }
                }
#else
                temp_derivative = (relu_deriv_0.asDiagonal() * mlp_.weight[0]) * nerf_jac;
#endif
            } else {
                MatrixXf32 relu_deriv_0 = batch_ReLU_derivative(mlp_.pre_activations[0].col(i));

                int deactivated_count_0 = (relu_deriv_0.array() == 0.0f).count();
                thread_deactivated_counts[i][0] = deactivated_count_0;

#ifdef NN_USE_FLEXFLOAT
                temp_derivative.resize(relu_deriv_0.rows(), mlp_.weight[0].cols());
                for (int r = 0; r < temp_derivative.rows(); ++r) {
                    for (int c = 0; c < temp_derivative.cols(); ++c) {
                        flexfloat_t ff_result, ff_diag, ff_w, ff_zero;
                        ff_init_float(&ff_zero, 0.0f, NN_FF_DESC);
                        ff_init_float(&ff_diag, relu_deriv_0(r), NN_FF_DESC);
                        ff_init_float(&ff_w, mlp_.weight[0](r, c), NN_FF_DESC);
                        ff_init_float(&ff_result, 0.0f, NN_FF_DESC);
                        ff_fma(&ff_result, &ff_diag, &ff_w, &ff_zero);
                        temp_derivative(r, c) = ff_get_float(&ff_result);
                    }
                }
#else
                temp_derivative = relu_deriv_0.asDiagonal() * mlp_.weight[0];
#endif
            }

            for (int layer = 1; layer < mlp_.n_layer - 1; ++layer) {
                MatrixXf32 relu_deriv = batch_ReLU_derivative(mlp_.pre_activations[layer].col(i));

                int deactivated_count = (relu_deriv.array() == 0.0f).count();
                thread_deactivated_counts[i][layer] = deactivated_count;

#ifdef NN_USE_FLEXFLOAT
                MatrixXf32 DW(relu_deriv.rows(), mlp_.weight[layer].cols());
                for (int r = 0; r < DW.rows(); ++r) {
                    for (int c = 0; c < DW.cols(); ++c) {
                        flexfloat_t ff_result, ff_diag, ff_w, ff_zero;
                        ff_init_float(&ff_zero, 0.0f, NN_FF_DESC);
                        ff_init_float(&ff_diag, relu_deriv(r), NN_FF_DESC);
                        ff_init_float(&ff_w, mlp_.weight[layer](r, c), NN_FF_DESC);
                        ff_init_float(&ff_result, 0.0f, NN_FF_DESC);
                        ff_fma(&ff_result, &ff_diag, &ff_w, &ff_zero);
                        DW(r, c) = ff_get_float(&ff_result);
                    }
                }
                MatrixXf32 new_derivative(DW.rows(), temp_derivative.cols());
                for (int r = 0; r < new_derivative.rows(); ++r) {
                    for (int c = 0; c < new_derivative.cols(); ++c) {
                        flexfloat_t ff_sum, ff_a, ff_b;
                        ff_init_float(&ff_sum, 0.0f, NN_FF_DESC);
                        for (int k = 0; k < DW.cols(); ++k) {
                            ff_init_float(&ff_a, DW(r, k), NN_FF_DESC);
                            ff_init_float(&ff_b, temp_derivative(k, c), NN_FF_DESC);
                            ff_fma(&ff_sum, &ff_a, &ff_b, &ff_sum);
                        }
                        new_derivative(r, c) = ff_get_float(&ff_sum);
                    }
                }
                temp_derivative = new_derivative;
#else
                temp_derivative = (relu_deriv.asDiagonal() * mlp_.weight[layer]) * temp_derivative;
#endif
            }

#ifdef NN_USE_FLEXFLOAT
            mlp_.batch_jacobian[i].resize(mlp_.weight.back().rows(), temp_derivative.cols());
            for (int r = 0; r < mlp_.batch_jacobian[i].rows(); ++r) {
                for (int c = 0; c < mlp_.batch_jacobian[i].cols(); ++c) {
                    flexfloat_t ff_sum, ff_a, ff_b;
                    ff_init_float(&ff_sum, 0.0f, NN_FF_DESC);
                    for (int k = 0; k < mlp_.weight.back().cols(); ++k) {
                        ff_init_float(&ff_a, mlp_.weight.back()(r, k), NN_FF_DESC);
                        ff_init_float(&ff_b, temp_derivative(k, c), NN_FF_DESC);
                        ff_fma(&ff_sum, &ff_a, &ff_b, &ff_sum);
                    }
                    mlp_.batch_jacobian[i](r, c) = ff_get_float(&ff_sum);
                }
            }
#else
            mlp_.batch_jacobian[i] = mlp_.weight.back() * temp_derivative;
#endif
        }

        // Accumulate deactivation statistics
        for (int i = 0; i < batch_size; ++i) {
            for (int layer = 0; layer < mlp_.n_layer - 1; ++layer) {
                int deactivated_count = thread_deactivated_counts[i][layer];
                mlp_.deactivated_units_per_layer[layer] += deactivated_count;

                if (deactivated_count < mlp_.min_deactivated_per_layer[layer]) {
                    mlp_.min_deactivated_per_layer[layer] = deactivated_count;
                }
                if (deactivated_count > mlp_.max_deactivated_per_layer[layer]) {
                    mlp_.max_deactivated_per_layer[layer] = deactivated_count;
                }
            }
        }

        // Find minimum per row and reconstruct result
        const int n_output = mlp_.n_output;
        const int n_input_total = inputs.rows();

        VectorXf32 final_min_output(n_output);
        MatrixXf32 final_min_jacobian(n_output, n_input_total);

        for (int i = 0; i < n_output; ++i)
        {
            Eigen::Index min_col_for_row;
            mlp_.batch_output.row(i).minCoeff(&min_col_for_row);

            final_min_output(i) = mlp_.batch_output(i, min_col_for_row);
            final_min_jacobian.row(i) = mlp_.batch_jacobian[min_col_for_row].row(i);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;
        mlp_.inference_times_ms.push_back(elapsed_ms.count());

        return std::make_pair(final_min_output.cast<double>(), final_min_jacobian.cast<double>());
    }

    const std::vector<double>& EnvCollNNmodel::getInferenceTimes() const
    {
        return mlp_.inference_times_ms;
    }

    std::vector<double> EnvCollNNmodel::getReluDeactivationRatios() const
    {
        std::vector<double> ratios;
        if (mlp_.total_sample_count == 0) {
            return ratios;
        }

        ratios.resize(mlp_.n_layer - 1);
        for (int i = 0; i < mlp_.n_layer - 1; ++i) {
            double avg_deactivated = static_cast<double>(mlp_.deactivated_units_per_layer[i]) / mlp_.total_sample_count;
            double total_units = static_cast<double>(mlp_.total_units_per_layer[i]);
            ratios[i] = avg_deactivated / total_units;
        }
        return ratios;
    }

    std::vector<int> EnvCollNNmodel::getReluTotalUnits() const
    {
        return mlp_.total_units_per_layer;
    }

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

    std::vector<int> EnvCollNNmodel::getReluMinDeactivatedCounts() const
    {
        return mlp_.min_deactivated_per_layer;
    }

    std::vector<int> EnvCollNNmodel::getReluMaxDeactivatedCounts() const
    {
        return mlp_.max_deactivated_per_layer;
    }

    void EnvCollNNmodel::saveInputOutputLog(const std::string& filename, const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& outputs) const
    {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
            return;
        }

        file << std::setprecision(10);
        file << "# Input/Output Log for Neural Network Validation" << std::endl;
        file << "# Inputs: " << inputs.rows() << " x " << inputs.cols() << std::endl;
        file << "# Outputs: " << outputs.rows() << " x " << outputs.cols() << std::endl;
        file << std::endl;

        file << "INPUTS:" << std::endl;
        for (int i = 0; i < inputs.cols(); ++i) {
            file << "Sample " << i << ": ";
            for (int j = 0; j < inputs.rows(); ++j) {
                file << inputs(j, i);
                if (j < inputs.rows() - 1) file << ", ";
            }
            file << std::endl;
        }

        file << std::endl << "OUTPUTS:" << std::endl;
        for (int i = 0; i < outputs.cols(); ++i) {
            file << "Sample " << i << ": ";
            for (int j = 0; j < outputs.rows(); ++j) {
                file << outputs(j, i);
                if (j < outputs.rows() - 1) file << ", ";
            }
            file << std::endl;
        }

        file.close();
    }

    void EnvCollNNmodel::enableLogging(bool enable)
    {
        logging_enabled_ = enable;
    }

    bool EnvCollNNmodel::isLoggingEnabled() const
    {
        return logging_enabled_;
    }

    Eigen::MatrixXd EnvCollNNmodel::getBatchOutput() const
    {
        return mlp_.batch_output.cast<double>();
    }
}
