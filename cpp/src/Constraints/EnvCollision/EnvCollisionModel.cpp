#include "Constraints/EnvCollision/EnvCollisionModel.h"
#include <chrono>
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
                mlp_.weight_files[weight_num] >> mlp_.weight[weight_num](i, j);
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
            mlp_.bias_files[bias_num] >> mlp_.bias[bias_num](i);
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

        // parameters resize (bfloat16 precision - using float)
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
        // input output resize (bfloat16 precision - using float)
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
        // Convert input from double to float for bfloat16 precision inference
        mlp_.input = input.cast<float>();
        if (mlp_.is_nerf)
        {
            printf("mlp is nerf\n");
            Eigen::VectorXf sinInput = input.array().sin().cast<float>();
            Eigen::VectorXf cosInput = input.array().cos().cast<float>();

            mlp_.input_nerf.segment(0 * mlp_.n_input, mlp_.n_input) = mlp_.input;
            mlp_.input_nerf.segment(1 * mlp_.n_input, mlp_.n_input) = sinInput;
            mlp_.input_nerf.segment(2 * mlp_.n_input, mlp_.n_input) = cosInput;
        }
        // std::cout<< "INPUT DATA:"<< std::endl <<mlp_.input.transpose() << std::endl;

        std::vector<clock_t> start, finish;
        start.resize(3*mlp_.n_layer);
        finish.resize(3*mlp_.n_layer);

        start[3*mlp_.n_layer - 1] = clock(); // Total
        Eigen::MatrixXf temp_derivative;
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
                    Eigen::MatrixXf nerf_jac;
                    nerf_jac.setZero(3 * mlp_.n_input, mlp_.n_input);
                    nerf_jac.block(0 * mlp_.n_input, 0, mlp_.n_input, mlp_.n_input) = Eigen::MatrixXf::Identity(mlp_.n_input, mlp_.n_input);
                    nerf_jac.block(1 * mlp_.n_input, 0, mlp_.n_input, mlp_.n_input).diagonal() <<   mlp_.input.array().cos();
                    nerf_jac.block(2 * mlp_.n_input, 0, mlp_.n_input, mlp_.n_input).diagonal() << - mlp_.input.array().sin();

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

        // Convert output from float to double for interface compatibility
        return std::make_pair(mlp_.output.cast<double>(), mlp_.output_derivative.cast<double>());
        // std::cout<< "OUTPUT DATA:"<< std::endl <<mlp_.output.transpose() << std::endl;
        // std::cout<< "OUTPUT DATA:"<< std::endl <<mlp_.output_derivative << std::endl;
    }

    // =================================================================
    // =========== 여기에 새로운 배치 추론 함수를 추가합니다 ============
    // =================================================================
    
    // 행렬 전체에 ReLU를 적용하는 헬퍼 함수 (bfloat16 precision - using float)
    Eigen::MatrixXf batch_ReLU(const Eigen::MatrixXf& x) {
        return (x.array() > 0).select(x, Eigen::MatrixXf::Zero(x.rows(), x.cols()));
    }

    // 행렬 전체에 ReLU의 미분을 적용하는 헬퍼 함수 (bfloat16 precision - using float)
    Eigen::MatrixXf batch_ReLU_derivative(const Eigen::MatrixXf& x) {
        return (x.array() > 0).select(Eigen::MatrixXf::Ones(x.rows(), x.cols()), Eigen::MatrixXf::Zero(x.rows(), x.cols()));
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

        // ===== 1. 모든 결과와 자코비안을 계산 (효율성을 위해 배치 연산 유지) =====
        // Convert input from double to float for bfloat16 precision inference
        mlp_.batch_input = inputs.cast<float>();

        const Eigen::MatrixXf* current_input_ptr;
        if (mlp_.is_nerf) {
            mlp_.batch_input_nerf.resize(3 * mlp_.n_input, batch_size);
            mlp_.batch_input_nerf.topRows(mlp_.n_input) = mlp_.batch_input;
            mlp_.batch_input_nerf.middleRows(mlp_.n_input, mlp_.n_input) = mlp_.batch_input.array().sin().matrix();
            mlp_.batch_input_nerf.bottomRows(mlp_.n_input) = mlp_.batch_input.array().cos().matrix();
            current_input_ptr = &mlp_.batch_input_nerf;
        } else {
            current_input_ptr = &mlp_.batch_input;
        }

        std::vector<Eigen::MatrixXf> pre_activations(mlp_.n_layer - 1);

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

        std::vector<Eigen::MatrixXf> batch_jacobian(batch_size);

        #pragma omp parallel for
        for (int i = 0; i < batch_size; ++i) {
            Eigen::MatrixXf temp_derivative;
            if (mlp_.is_nerf) {
                Eigen::MatrixXf nerf_jac(3 * mlp_.n_input, mlp_.n_input);
                nerf_jac.setZero();
                nerf_jac.topRows(mlp_.n_input) = Eigen::MatrixXf::Identity(mlp_.n_input, mlp_.n_input);
                nerf_jac.middleRows(mlp_.n_input, mlp_.n_input).diagonal() = inputs.col(i).array().cos().cast<float>();
                nerf_jac.bottomRows(mlp_.n_input).diagonal() = (-inputs.col(i).array().sin()).cast<float>();
                Eigen::MatrixXf relu_deriv_0 = batch_ReLU_derivative(pre_activations[0].col(i));
                temp_derivative = (relu_deriv_0.asDiagonal() * mlp_.weight[0]) * nerf_jac;
            } else {
                Eigen::MatrixXf relu_deriv_0 = batch_ReLU_derivative(pre_activations[0].col(i));
                temp_derivative = relu_deriv_0.asDiagonal() * mlp_.weight[0];
            }

            for (int layer = 1; layer < mlp_.n_layer - 1; ++layer) {
                Eigen::MatrixXf relu_deriv = batch_ReLU_derivative(pre_activations[layer].col(i));
                temp_derivative = (relu_deriv.asDiagonal() * mlp_.weight[layer]) * temp_derivative;
            }
            batch_jacobian[i] = mlp_.weight.back() * temp_derivative;
        }

        // ===== 2. 각 행(링크)별 최소값 탐색 및 결과 재구성 =====

        const int n_output = mlp_.n_output;
        const int n_input_total = inputs.rows(); // 예: 7(dof) + 3(obs) = 10

        // 1. 최종 결과를 담을 새로운 벡터와 자코비안 행렬을 초기화합니다.
        Eigen::VectorXf final_min_output(n_output);
        Eigen::MatrixXf final_min_jacobian(n_output, n_input_total);

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
        // Convert output from float to double for interface compatibility
        return std::make_pair(final_min_output.cast<double>(), final_min_jacobian.cast<double>());
    }

    // ▼▼▼▼▼ 파일 하단에 getter 함수 구현을 추가합니다. ▼▼▼▼▼
    const std::vector<double>& EnvCollNNmodel::getInferenceTimes() const
    {
        return mlp_.inference_times_ms;
    }
}



