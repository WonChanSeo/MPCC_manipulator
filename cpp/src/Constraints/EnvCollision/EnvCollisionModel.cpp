#include "Constraints/EnvCollision/EnvCollisionModel.h"
#include <chrono>

#ifdef NN_USE_FLEXFLOAT
#include <flexfloat.h>
#endif

#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error in " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
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
#ifdef USE_CUDA
        cleanupGPU();
#endif
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
                mlp_.weight[weight_num](i, j) = Eigen::bfloat16(temp_value);
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
            mlp_.bias[bias_num](i) = Eigen::bfloat16(temp_value);
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
    }

    void EnvCollNNmodel::setNeuralNetwork(int n_input, int n_output, Eigen::VectorXd n_hidden, bool is_nerf)
    {
        initializeNetwork(n_input, n_output, n_hidden, is_nerf);
        loadNetwork();

#ifdef USE_CUDA
        transferWeightsToGPU();
#endif
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> EnvCollNNmodel::calculateMlpOutput(Eigen::VectorXd input, bool time_verbose)
    {
        // Convert input from double to bfloat16 for inference
        mlp_.input = input.cast<Eigen::bfloat16>();
        if (mlp_.is_nerf)
        {
            printf("mlp is nerf\n");
            VectorXbf16 sinInput = input.array().sin().cast<Eigen::bfloat16>();
            VectorXbf16 cosInput = input.array().cos().cast<Eigen::bfloat16>();

            mlp_.input_nerf.segment(0 * mlp_.n_input, mlp_.n_input) = mlp_.input;
            mlp_.input_nerf.segment(1 * mlp_.n_input, mlp_.n_input) = sinInput;
            mlp_.input_nerf.segment(2 * mlp_.n_input, mlp_.n_input) = cosInput;
        }
        // std::cout<< "INPUT DATA:"<< std::endl <<mlp_.input.transpose() << std::endl;

        std::vector<clock_t> start, finish;
        start.resize(3*mlp_.n_layer);
        finish.resize(3*mlp_.n_layer);

        start[3*mlp_.n_layer - 1] = clock(); // Total
        MatrixXbf16 temp_derivative;
        for (int layer = 0; layer < mlp_.n_layer; layer++)
        {
            if (layer == 0) // input layer
            {
                start[0] = clock(); // Linear 
                if (mlp_.is_nerf) mlp_.hidden[0] = mlp_.weight[0] * mlp_.input_nerf + mlp_.bias[0];
                else                mlp_.hidden[0] = mlp_.weight[0] * mlp_.input + mlp_.bias[0];
                finish[0] = clock();

                start[1] = clock(); // ReLU
                for (int h = 0; h < mlp_.n_hidden(layer); h++)
                {
                    mlp_.hidden_derivative[0].row(h) = ReLU_derivative(mlp_.hidden[0](h)) * mlp_.weight[0].row(h); //derivative wrt input
                    mlp_.hidden[0](h) = ReLU(mlp_.hidden[0](h));                                                     //activation function
                }
                finish[1] = clock();

                if (mlp_.is_nerf)
                {
                    MatrixXbf16 nerf_jac;
                    nerf_jac.setZero(3 * mlp_.n_input, mlp_.n_input);
                    nerf_jac.block(0 * mlp_.n_input, 0, mlp_.n_input, mlp_.n_input).setIdentity();
                    nerf_jac.block(1 * mlp_.n_input, 0, mlp_.n_input, mlp_.n_input).diagonal() = mlp_.input.array().cos();
                    nerf_jac.block(2 * mlp_.n_input, 0, mlp_.n_input, mlp_.n_input).diagonal() = -mlp_.input.array().sin();

                    start[2] = clock(); // Multip
                    temp_derivative = mlp_.hidden_derivative[0] * nerf_jac;
                    finish[2] = clock();
                }
                else
                {
                    temp_derivative = mlp_.hidden_derivative[0];
                }
            }
            else if (layer == mlp_.n_layer - 1) // output layer
            {
                start[layer*3] = clock(); // Linear
                mlp_.output = mlp_.weight[layer] * mlp_.hidden[layer - 1] + mlp_.bias[layer];
                finish[layer*3] = clock(); 

                start[layer*3+1] = clock(); // Multip
                mlp_.output_derivative = mlp_.weight[layer] * temp_derivative;
                finish[layer*3+1] = clock();
            }
            else // hidden layers
            {
                start[layer*3] = clock(); // Linear
                mlp_.hidden[layer] = mlp_.weight[layer] * mlp_.hidden[layer - 1] + mlp_.bias[layer];
                finish[layer*3] = clock();
                
                start[layer*3+1] = clock(); // ReLU
                for (int h = 0; h < mlp_.n_hidden(layer); h++)
                {
                    mlp_.hidden_derivative[layer].row(h) = ReLU_derivative(mlp_.hidden[layer](h)) * mlp_.weight[layer].row(h); //derivative wrt input
                    mlp_.hidden[layer](h) = ReLU(mlp_.hidden[layer](h));                                                         //activation function
                }
                finish[layer*3+1] = clock();

                start[3*layer+2] = clock(); //Multip
                temp_derivative = mlp_.hidden_derivative[layer] * temp_derivative;
                finish[3*layer+2] = clock();
            }
        }

        finish[3*mlp_.n_layer - 1] = clock();

        // if(time_verbose)
        // {
        //     std::cout<<"------------------Time[1e-6]------------------"<<std::endl;
        //     for (int layer = 0; layer < mlp_.n_layer; layer++)
        //     {
        //         if(layer == mlp_.n_layer - 1)
        //         {
        //             std::cout<<"Layer "<<layer<<" -Linear: "<<double(finish[3*layer+0]-start[3*layer+0])<<std::endl;
        //             std::cout<<"Layer "<<layer<<" -Multip: "<<double(finish[3*layer+1]-start[3*layer+1])<<std::endl;
        //             std::cout<<"Total          : "          <<double(finish[3*layer+2]-start[3*layer+2])<<std::endl;
        //         }
        //         else
        //         {
        //             std::cout<<"Layer "<<layer<<" -Linear: "<<double(finish[3*layer+0]-start[3*layer+0])<<std::endl;
        //             std::cout<<"Layer "<<layer<<" -ReLU  : "<<double(finish[3*layer+1]-start[3*layer+1])<<std::endl;
        //             std::cout<<"Layer "<<layer<<" -Multip: "<<double(finish[3*layer+2]-start[3*layer+2])<<std::endl;
        //         }
        //     }
        //     std::cout<<"------------------------------------------------"<<std::endl;
        // }

        // Convert output from bfloat16 to double for interface compatibility
        return std::make_pair(mlp_.output.cast<double>(), mlp_.output_derivative.cast<double>());
        // std::cout<< "OUTPUT DATA:"<< std::endl <<mlp_.output.transpose() << std::endl;
        // std::cout<< "OUTPUT DATA:"<< std::endl <<mlp_.output_derivative << std::endl;
    }

    // =================================================================
    // =========== 여기에 새로운 배치 추론 함수를 추가합니다 ============
    // =================================================================

    // 행렬 전체에 ReLU를 적용하는 헬퍼 함수 (bfloat16 precision)
    MatrixXbf16 batch_ReLU(const MatrixXbf16& x) {
        return (x.array() > Eigen::bfloat16(0)).select(x, MatrixXbf16::Zero(x.rows(), x.cols()));
    }

    // 행렬 전체에 ReLU의 미분을 적용하는 헬퍼 함수 (bfloat16 precision)
    MatrixXbf16 batch_ReLU_derivative(const MatrixXbf16& x) {
        return (x.array() > Eigen::bfloat16(0)).select(MatrixXbf16::Ones(x.rows(), x.cols()), MatrixXbf16::Zero(x.rows(), x.cols()));
    }

    // in EnvCollisionModel.cpp

    // 함수의 시그니처(반환 타입)와 내용 전체를 아래 코드로 덮어씁니다.
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> EnvCollNNmodel::calculateMlpOutputBatch(const Eigen::MatrixXd& inputs, bool time_verbose)
    {
        // ▼▼▼▼▼ 타이머 시작 ▼▼▼▼▼
        auto start_time = std::chrono::high_resolution_clock::now();

        const int batch_size = inputs.cols();
        if (batch_size == 0) {
            return {Eigen::VectorXd(), Eigen::MatrixXd()};
        }

        // ▼▼▼▼▼ ReLU deactivation 통계 초기화 (첫 호출 시) ▼▼▼▼▼
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

        // ===== 1. 모든 결과와 자코비안을 계산 (효율성을 위해 배치 연산 유지) =====
        // Convert input from double to bfloat16 for inference
        mlp_.batch_input = inputs.cast<Eigen::bfloat16>();

        const MatrixXbf16* current_input_ptr;
        if (mlp_.is_nerf) {
            mlp_.batch_input_nerf.resize(3 * mlp_.n_input, batch_size);
            mlp_.batch_input_nerf.topRows(mlp_.n_input) = mlp_.batch_input;
            mlp_.batch_input_nerf.middleRows(mlp_.n_input, mlp_.n_input) = mlp_.batch_input.array().sin().matrix();
            mlp_.batch_input_nerf.bottomRows(mlp_.n_input) = mlp_.batch_input.array().cos().matrix();
            current_input_ptr = &mlp_.batch_input_nerf;
        } else {
            current_input_ptr = &mlp_.batch_input;
        }

        std::vector<MatrixXbf16> pre_activations(mlp_.n_layer - 1);

#ifdef USE_CUDA
        // ===== CUDA GPU Acceleration =====
        // GPU 초기화 및 가중치 전송 (첫 호출 시)
        if (!mlp_.gpu_initialized) {
            initializeGPU();
            transferWeightsToGPU();

            // Batch hidden layers 메모리 할당
            mlp_.d_batch_hidden.resize(mlp_.n_layer - 1);
        }

        const int batch = batch_size;
        const int input_dim = current_input_ptr->rows();
        const int batch_input_size = input_dim * batch;

        // GPU 메모리 할당/재할당 - 크기가 변경되면 재할당
        static int prev_batch_input_size = 0;
        if (mlp_.d_batch_input == nullptr || batch_input_size != prev_batch_input_size) {
            if (mlp_.d_batch_input != nullptr) {
                CUDA_CHECK(cudaFree(mlp_.d_batch_input));
            }
            CUDA_CHECK(cudaMalloc(&mlp_.d_batch_input, batch_input_size * sizeof(float)));
            prev_batch_input_size = batch_input_size;
        }

        // Convert input from bfloat16 to float and upload to GPU
        std::vector<float> input_float(batch_input_size);
        for (int i = 0; i < batch_input_size; ++i) {
            input_float[i] = static_cast<float>(current_input_ptr->data()[i]);
        }
        CUDA_CHECK(cudaMemcpyAsync(mlp_.d_batch_input, input_float.data(),
                                   batch_input_size * sizeof(float),
                                   cudaMemcpyHostToDevice, mlp_.stream));

        // Forward pass through all layers
        for (int layer = 0; layer < mlp_.n_layer; ++layer) {
            const int out_dim = mlp_.weight[layer].rows();
            const int in_dim = mlp_.weight[layer].cols();

            float* d_input_current;
            float* d_output_current;

            // Determine input pointer
            if (layer == 0) {
                d_input_current = mlp_.d_batch_input;
            } else {
                d_input_current = mlp_.d_batch_hidden[layer - 1];
            }

            // Determine output pointer and allocate/reallocate if needed
            if (layer == mlp_.n_layer - 1) {
                // Last layer - output layer
                const int output_size = out_dim * batch;
                static int prev_output_size = 0;
                if (mlp_.d_batch_output == nullptr || output_size != prev_output_size) {
                    if (mlp_.d_batch_output != nullptr) {
                        CUDA_CHECK(cudaFree(mlp_.d_batch_output));
                    }
                    CUDA_CHECK(cudaMalloc(&mlp_.d_batch_output, output_size * sizeof(float)));
                    prev_output_size = output_size;
                }
                d_output_current = mlp_.d_batch_output;
            } else {
                // Hidden layer
                const int hidden_size = out_dim * batch;
                static std::vector<int> prev_hidden_sizes(mlp_.n_layer - 1, 0);
                if (mlp_.d_batch_hidden[layer] == nullptr || hidden_size != prev_hidden_sizes[layer]) {
                    if (mlp_.d_batch_hidden[layer] != nullptr) {
                        CUDA_CHECK(cudaFree(mlp_.d_batch_hidden[layer]));
                    }
                    CUDA_CHECK(cudaMalloc(&mlp_.d_batch_hidden[layer], hidden_size * sizeof(float)));
                    prev_hidden_sizes[layer] = hidden_size;
                }
                d_output_current = mlp_.d_batch_hidden[layer];
            }

            // Temporary buffer for pre-activation (before ReLU)
            float* d_temp_output = nullptr;
            if (layer < mlp_.n_layer - 1) {
                CUDA_CHECK(cudaMalloc(&d_temp_output, out_dim * batch * sizeof(float)));
            }

            // Launch matrix multiplication kernel with custom precision FMA
            launch_matmul_fma(
                mlp_.d_weight[layer],      // weight[out_dim x in_dim]
                d_input_current,            // input[in_dim x batch]
                mlp_.d_bias[layer],        // bias[out_dim]
                (layer < mlp_.n_layer - 1) ? d_temp_output : d_output_current,  // output[out_dim x batch]
                out_dim,
                in_dim,
                batch,
                mlp_.cuda_exp_bits,
                mlp_.cuda_mant_bits,
                mlp_.stream
            );

            // Apply ReLU for hidden layers (not output layer)
            if (layer < mlp_.n_layer - 1) {
                pre_activations[layer].resize(out_dim, batch);

                // Launch ReLU kernel
                launch_relu(
                    d_temp_output,
                    d_output_current,
                    out_dim * batch,
                    mlp_.cuda_exp_bits,
                    mlp_.cuda_mant_bits,
                    mlp_.stream
                );

                // Download pre-activation for statistics (optional, can be removed for performance)
                std::vector<float> pre_act_float(out_dim * batch);
                CUDA_CHECK(cudaMemcpyAsync(pre_act_float.data(), d_temp_output,
                                          out_dim * batch * sizeof(float),
                                          cudaMemcpyDeviceToHost, mlp_.stream));
                CUDA_CHECK(cudaStreamSynchronize(mlp_.stream));

                for (int i = 0; i < out_dim * batch; ++i) {
                    pre_activations[layer].data()[i] = Eigen::bfloat16(pre_act_float[i]);
                }

                // Download hidden activation for later use
                mlp_.batch_hidden[layer].resize(out_dim, batch);
                std::vector<float> hidden_float(out_dim * batch);
                CUDA_CHECK(cudaMemcpyAsync(hidden_float.data(), d_output_current,
                                          out_dim * batch * sizeof(float),
                                          cudaMemcpyDeviceToHost, mlp_.stream));
                CUDA_CHECK(cudaStreamSynchronize(mlp_.stream));

                for (int i = 0; i < out_dim * batch; ++i) {
                    mlp_.batch_hidden[layer].data()[i] = Eigen::bfloat16(hidden_float[i]);
                }

                CUDA_CHECK(cudaFree(d_temp_output));
            }
        }

        // Download final output from GPU
        const int output_size = mlp_.weight[mlp_.n_layer - 1].rows() * batch;
        mlp_.batch_output.resize(mlp_.weight[mlp_.n_layer - 1].rows(), batch);
        std::vector<float> output_float(output_size);
        CUDA_CHECK(cudaMemcpyAsync(output_float.data(), mlp_.d_batch_output,
                                   output_size * sizeof(float),
                                   cudaMemcpyDeviceToHost, mlp_.stream));
        CUDA_CHECK(cudaStreamSynchronize(mlp_.stream));

        for (int i = 0; i < output_size; ++i) {
            mlp_.batch_output.data()[i] = Eigen::bfloat16(output_float[i]);
        }

#else
        // ===== CPU Fallback (no OpenMP, just Eigen) =====
        for (int layer = 0; layer < mlp_.n_layer; ++layer) {
            if (layer == 0) {
                pre_activations[0] = (mlp_.weight[0] * (*current_input_ptr)).colwise() + mlp_.bias[0];
                mlp_.batch_hidden[0] = batch_ReLU(pre_activations[0]);
            }
            else if (layer == mlp_.n_layer - 1) {
                mlp_.batch_output = (mlp_.weight[layer] * mlp_.batch_hidden[layer - 1]).colwise() + mlp_.bias[layer];
            }
            else {
                pre_activations[layer] = (mlp_.weight[layer] * mlp_.batch_hidden[layer - 1]).colwise() + mlp_.bias[layer];
                mlp_.batch_hidden[layer] = batch_ReLU(pre_activations[layer]);
            }
        }
#endif

        std::vector<MatrixXbf16> batch_jacobian(batch_size);

        // ▼▼▼▼▼ ReLU deactivation을 추적하기 위한 임시 변수 ▼▼▼▼▼
        std::vector<std::vector<int>> thread_deactivated_counts(batch_size, std::vector<int>(mlp_.n_layer - 1, 0));

        // Calculate Jacobian for each batch sample (CPU computation, no OpenMP)
        for (int i = 0; i < batch_size; ++i) {
            MatrixXbf16 temp_derivative;
            if (mlp_.is_nerf) {
                MatrixXbf16 nerf_jac(3 * mlp_.n_input, mlp_.n_input);
                nerf_jac.setZero();
                nerf_jac.topRows(mlp_.n_input).setIdentity();
                nerf_jac.middleRows(mlp_.n_input, mlp_.n_input).diagonal() = inputs.col(i).array().cos().cast<Eigen::bfloat16>();
                nerf_jac.bottomRows(mlp_.n_input).diagonal() = (-inputs.col(i).array().sin()).cast<Eigen::bfloat16>();
                MatrixXbf16 relu_deriv_0 = batch_ReLU_derivative(pre_activations[0].col(i));

                // ▼▼▼▼▼ Layer 0의 deactivation 카운트 ▼▼▼▼▼
                int deactivated_count_0 = (relu_deriv_0.array() == Eigen::bfloat16(0)).count();
                thread_deactivated_counts[i][0] = deactivated_count_0;

                temp_derivative = (relu_deriv_0.asDiagonal() * mlp_.weight[0]) * nerf_jac;
            } else {
                MatrixXbf16 relu_deriv_0 = batch_ReLU_derivative(pre_activations[0].col(i));

                // ▼▼▼▼▼ Layer 0의 deactivation 카운트 ▼▼▼▼▼
                int deactivated_count_0 = (relu_deriv_0.array() == Eigen::bfloat16(0)).count();
                thread_deactivated_counts[i][0] = deactivated_count_0;

                temp_derivative = relu_deriv_0.asDiagonal() * mlp_.weight[0];
            }

            for (int layer = 1; layer < mlp_.n_layer - 1; ++layer) {
                MatrixXbf16 relu_deriv = batch_ReLU_derivative(pre_activations[layer].col(i));

                // ▼▼▼▼▼ 각 hidden layer의 deactivation 카운트 ▼▼▼▼▼
                int deactivated_count = (relu_deriv.array() == Eigen::bfloat16(0)).count();
                thread_deactivated_counts[i][layer] = deactivated_count;

                temp_derivative = (relu_deriv.asDiagonal() * mlp_.weight[layer]) * temp_derivative;
            }

            batch_jacobian[i] = mlp_.weight.back() * temp_derivative;
        }

        // ▼▼▼▼▼ 모든 배치 샘플의 deactivation 수를 누적하고 min/max 업데이트 ▼▼▼▼▼
        for (int i = 0; i < batch_size; ++i) {
            for (int layer = 0; layer < mlp_.n_layer - 1; ++layer) {
                int deactivated_count = thread_deactivated_counts[i][layer];
                mlp_.deactivated_units_per_layer[layer] += deactivated_count;

                // min/max 업데이트
                if (deactivated_count < mlp_.min_deactivated_per_layer[layer]) {
                    mlp_.min_deactivated_per_layer[layer] = deactivated_count;
                }
                if (deactivated_count > mlp_.max_deactivated_per_layer[layer]) {
                    mlp_.max_deactivated_per_layer[layer] = deactivated_count;
                }
            }
        }

        // ===== 2. 각 행(링크)별 최소값 탐색 및 결과 재구성 =====

        const int n_output = mlp_.n_output;
        const int n_input_total = inputs.rows(); // 예: 7(dof) + 3(obs) = 10

        // 1. 최종 결과를 담을 새로운 벡터와 자코비안 행렬을 초기화합니다.
        VectorXbf16 final_min_output(n_output);
        MatrixXbf16 final_min_jacobian(n_output, n_input_total);

        // 2. 각 행(각 링크)을 순회하는 루프를 실행합니다.
        for (int i = 0; i < n_output; ++i)
        {
            // i번째 행에서 최소값과 그 값의 열(column) 인덱스를 찾습니다.
            // 이 min_col_for_row는 i번째 링크에 가장 가까운 장애물의 인덱스를 의미합니다.
            Eigen::Index min_col_for_row;
            mlp_.batch_output.row(i).minCoeff(&min_col_for_row);

            // 3. 최종 결과 벡터의 i번째 원소를 채웁니다.
            // i번째 링크와 가장 가까운 장애물과의 거리(최소값)를 저장합니다.
            final_min_output(i) = mlp_.batch_output(i, min_col_for_row);

            // 4. 최종 자코비안 행렬의 i번째 행을 채웁니다.
            // i번째 링크에 가장 위협적인 장애물(min_col_for_row)의 자코비안 행렬을 가져와서,
            // 그 행렬의 i번째 행(i번째 링크에 대한 미분값)만 복사합니다.
            final_min_jacobian.row(i) = batch_jacobian[min_col_for_row].row(i);
        }

        // ▼▼▼▼▼ 타이머 종료 및 시간 계산/저장 (기존과 동일) ▼▼▼▼▼
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;
        mlp_.inference_times_ms.push_back(elapsed_ms.count());
        // ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        // 재구성된 최종 결과 벡터와 자코비안 행렬을 반환합니다.
        // Convert output from bfloat16 to double for interface compatibility
        return std::make_pair(final_min_output.cast<double>(), final_min_jacobian.cast<double>());
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

#ifdef USE_CUDA
    // =================================================================
    // ===================== GPU 관련 함수 구현 =========================
    // =================================================================

    void EnvCollNNmodel::initializeGPU()
    {
        if (mlp_.gpu_initialized) return;

        std::cout << "[GPU] Initializing CUDA and cuBLAS..." << std::endl;

        // cuBLAS 핸들 생성
        CUBLAS_CHECK(cublasCreate(&mlp_.cublas_handle));

        // CUDA stream 생성
        CUDA_CHECK(cudaStreamCreate(&mlp_.stream));

        // Precision parameters 설정 (from compile-time defines)
        mlp_.cuda_exp_bits = NN_FF_exponent_bits;
        mlp_.cuda_mant_bits = NN_FF_mantissa_bits;

        mlp_.gpu_initialized = true;
        std::cout << "[GPU] Initialization complete!" << std::endl;
        std::cout << "[GPU] Precision: " << mlp_.cuda_exp_bits << " exp bits, "
                  << mlp_.cuda_mant_bits << " mantissa bits" << std::endl;
    }

    void EnvCollNNmodel::cleanupGPU()
    {
        if (!mlp_.gpu_initialized) return;

        std::cout << "[GPU] Cleaning up GPU resources..." << std::endl;

        // Weight 메모리 해제
        for (auto ptr : mlp_.d_weight) {
            if (ptr) CUDA_CHECK(cudaFree(ptr));
        }
        mlp_.d_weight.clear();

        // Bias 메모리 해제
        for (auto ptr : mlp_.d_bias) {
            if (ptr) CUDA_CHECK(cudaFree(ptr));
        }
        mlp_.d_bias.clear();

        // Hidden 메모리 해제
        for (auto ptr : mlp_.d_hidden) {
            if (ptr) CUDA_CHECK(cudaFree(ptr));
        }
        mlp_.d_hidden.clear();

        // Batch hidden 메모리 해제
        for (auto ptr : mlp_.d_batch_hidden) {
            if (ptr) CUDA_CHECK(cudaFree(ptr));
        }
        mlp_.d_batch_hidden.clear();

        // 기타 메모리 해제
        if (mlp_.d_input) CUDA_CHECK(cudaFree(mlp_.d_input));
        if (mlp_.d_output) CUDA_CHECK(cudaFree(mlp_.d_output));
        if (mlp_.d_batch_input) CUDA_CHECK(cudaFree(mlp_.d_batch_input));
        if (mlp_.d_batch_output) CUDA_CHECK(cudaFree(mlp_.d_batch_output));

        // CUDA stream 해제
        if (mlp_.stream) {
            CUDA_CHECK(cudaStreamDestroy(mlp_.stream));
            mlp_.stream = nullptr;
        }

        // cuBLAS 핸들 해제
        if (mlp_.cublas_handle) {
            CUBLAS_CHECK(cublasDestroy(mlp_.cublas_handle));
            mlp_.cublas_handle = nullptr;
        }

        mlp_.gpu_initialized = false;
        std::cout << "[GPU] Cleanup complete!" << std::endl;
    }

    void EnvCollNNmodel::transferWeightsToGPU()
    {
        if (!mlp_.gpu_initialized) {
            initializeGPU();
        }

        std::cout << "[GPU] Transferring weights and biases to GPU..." << std::endl;

        // GPU 메모리 벡터 크기 조정
        mlp_.d_weight.resize(mlp_.n_layer);
        mlp_.d_bias.resize(mlp_.n_layer);

        for (int i = 0; i < mlp_.n_layer; ++i) {
            // Weight 메모리 할당 및 전송
            const int weight_size = mlp_.weight[i].rows() * mlp_.weight[i].cols();
            CUDA_CHECK(cudaMalloc(&mlp_.d_weight[i], weight_size * sizeof(float)));

            // bfloat16 -> float32 변환 후 전송
            std::vector<float> weight_float(weight_size);
            for (int j = 0; j < weight_size; ++j) {
                weight_float[j] = static_cast<float>(mlp_.weight[i].data()[j]);
            }
            CUDA_CHECK(cudaMemcpy(mlp_.d_weight[i], weight_float.data(),
                                 weight_size * sizeof(float), cudaMemcpyHostToDevice));

            // Bias 메모리 할당 및 전송
            const int bias_size = mlp_.bias[i].size();
            CUDA_CHECK(cudaMalloc(&mlp_.d_bias[i], bias_size * sizeof(float)));

            std::vector<float> bias_float(bias_size);
            for (int j = 0; j < bias_size; ++j) {
                bias_float[j] = static_cast<float>(mlp_.bias[i](j));
            }
            CUDA_CHECK(cudaMemcpy(mlp_.d_bias[i], bias_float.data(),
                                 bias_size * sizeof(float), cudaMemcpyHostToDevice));

            std::cout << "[GPU] Layer " << i << ": Weight(" << mlp_.weight[i].rows()
                      << "x" << mlp_.weight[i].cols() << "), Bias(" << bias_size << ")" << std::endl;
        }

        std::cout << "[GPU] Transfer complete!" << std::endl;
    }
#endif
}



