#include "Constraints/EnvCollision/EnvCollisionModel.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>
#include <cstring>

#if defined(NN_USE_TRUNCATE) || defined(NN_USE_ROUND_TO_EVEN)
#include <cfenv>
#endif

#if defined(NN_USE_TRUNCATE)
  #define NN_ROUNDING_MODE FE_TOWARDZERO
#elif defined(NN_USE_ROUND_TO_EVEN)
  #define NN_ROUNDING_MODE FE_TONEAREST
#endif

namespace mpcc
{
    // ===================================================================
    // =========== ASIC Test Case Logging ================================
    // ===================================================================
    static int g_asic_sample_count = 0;
    static const int MLP_TESTCASE_SAVE_INTERVAL = 50;
    static const std::string g_asic_output_dir = "/home/mms-wonchan/git/MPCC_manipulator/result/asic_testcases/mlp/";
    static bool g_asic_dir_initialized = false;

    void init_asic_output_dir() {
        if (!g_asic_dir_initialized) {
            // 부모 디렉토리들을 순차적으로 생성
            std::string cmd = "mkdir -p " + g_asic_output_dir;
            system(cmd.c_str());
            g_asic_dir_initialized = true;
        }
    }

    // FP32를 16진수 비트 표현으로 변환하는 헬퍼 함수
    std::string float_to_hex(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(float));
        std::stringstream ss;
        ss << std::hex << std::setw(8) << std::setfill('0') << bits;
        return ss.str();
    }

    // 행렬을 파일에 저장하는 헬퍼 함수
    void save_matrix_to_file(const std::string& filepath, const Eigen::MatrixXf& mat, const std::string& name = "") {
        std::ofstream file(filepath, std::ios::app);
        if (file.is_open()) {
            file << std::scientific << std::setprecision(8);
            if (!name.empty()) {
                file << "# " << name << " (" << mat.rows() << " x " << mat.cols() << ")\n";
            }
            for (int r = 0; r < mat.rows(); ++r) {
                for (int c = 0; c < mat.cols(); ++c) {
                    file << mat(r, c);
                    if (c < mat.cols() - 1) file << ",";
                }
                file << "\n";
            }
            file << "\n";
            file.close();
        }
    }

    void save_vector_to_file(const std::string& filepath, const Eigen::VectorXf& vec, const std::string& name = "") {
        std::ofstream file(filepath, std::ios::app);
        if (file.is_open()) {
            file << std::scientific << std::setprecision(8);
            if (!name.empty()) {
                file << "# " << name << " (" << vec.size() << ")\n";
            }
            for (int i = 0; i < vec.size(); ++i) {
                file << vec(i);
                if (i < vec.size() - 1) file << ",";
            }
            file << "\n\n";
            file.close();
        }
    }
    // ===================================================================
    // =========== Hardware-like Adder Tree Functions ====================
    // ===================================================================
    // fesetround(FE_TOWARDZERO)가 설정된 상태에서 모든 연산이 자동으로 truncate됨

    // Adder tree 합산 (임의 개수)
    float adder_tree_sum(const float* data, int count) {
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
#ifdef NN_ROUNDING_MODE
                // volatile로 메모리에 강제 저장하여 truncate 적용
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

    // Matrix-vector multiplication using adder tree - Method A (하드웨어 방식)
    // bias + chunk0 -> truncate -> + chunk1 -> truncate
    float matvec_row_adder_tree(const Eigen::Ref<const Eigen::RowVectorXf>& weight_row,
                                 const Eigen::Ref<const Eigen::VectorXf>& input,
                                 float bias) {
        const int n = weight_row.size();
        const int chunk_size = 128;
        std::vector<float> products(n);

        // Stage 1: 모든 곱셈을 병렬로 수행
        for (int i = 0; i < n; ++i) {
            float prod = weight_row(i) * input(i);
#ifdef NN_ROUNDING_MODE
            volatile float truncated = prod;
            products[i] = truncated;
#else
            products[i] = prod;
#endif
        }

        // Stage 2: 128개씩 chunk로 adder tree 수행 (최대 2개 chunk)
        float chunk_sum_0 = adder_tree_sum(products.data(), std::min(n, chunk_size));
        float chunk_sum_1 = (n > chunk_size) ? adder_tree_sum(products.data() + chunk_size, n - chunk_size) : 0.0f;

        // Method A: bias + chunk0 -> truncate -> + chunk1 -> truncate
        float result = bias + chunk_sum_0;
#ifdef NN_ROUNDING_MODE
        volatile float truncated_1 = result;
        result = truncated_1;
#endif
        result = result + chunk_sum_1;
#ifdef NN_ROUNDING_MODE
        volatile float truncated_2 = result;
        result = truncated_2;
#endif

        return result;
    }
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

    void EnvCollNNmodel::beginBatchLog(int solve_count, int batch_size) {
        log_solve_count_ = solve_count;
        log_batch_size_ = batch_size;
        log_horizon_idx_ = 0;
        log_entries_.resize(batch_size);
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> EnvCollNNmodel::calculateMlpOutput(Eigen::VectorXd input, bool time_verbose)
    {
#ifdef NN_ROUNDING_MODE
        int old_round = std::fegetround();
        std::fesetround(NN_ROUNDING_MODE);
#endif

        // FP32 연산용 임시 변수들
        Eigen::VectorXf input_f = input.cast<float>();
        Eigen::VectorXf input_nerf_f;
        std::vector<Eigen::VectorXf> hidden_f(mlp_.n_layer - 1);
        std::vector<Eigen::VectorXf> pre_activation_f(mlp_.n_layer - 1);  // ReLU 이전 값 저장
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
        Eigen::MatrixXf nerf_jac_f;  // NeRF Jacobian (로깅용)

        for (int layer = 0; layer < mlp_.n_layer; layer++)
        {
            if (layer == 0) // input layer
            {
                const Eigen::VectorXf& current_input = mlp_.is_nerf ? input_nerf_f : input_f;
                const int n_neurons = static_cast<int>(mlp_.n_hidden(0));
                const int n_inputs = current_input.size();

                hidden_f[0].resize(n_neurons);
                pre_activation_f[0].resize(n_neurons);
                hidden_derivative_f[0].resize(n_neurons, n_inputs);

                // Adder tree를 사용한 행렬-벡터 곱셈
                for (int h = 0; h < n_neurons; ++h) {
                    pre_activation_f[0](h) = matvec_row_adder_tree(
                        mlp_.weight[0].row(h), current_input, mlp_.bias[0](h));
                }

                for (int h = 0; h < n_neurons; h++)
                {
                    float relu_deriv = (pre_activation_f[0](h) > 0.0f) ? 1.0f : 0.0f;
                    hidden_derivative_f[0].row(h) = relu_deriv * mlp_.weight[0].row(h);
                    hidden_f[0](h) = std::max(0.0f, pre_activation_f[0](h));
                }

                if (mlp_.is_nerf)
                {
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
                            temp_derivative_f(r, c) = matvec_row_adder_tree(
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

                // Adder tree를 사용한 출력층 계산 - Method A 사용
                for (int o = 0; o < n_outputs; ++o) {
                    output_f(o) = matvec_row_adder_tree(
                        mlp_.weight[layer].row(o), hidden_f[layer - 1], mlp_.bias[layer](o));
                }

                // 출력층 Jacobian 계산 (adder tree 사용)
                const int deriv_rows = mlp_.weight[layer].rows();
                const int deriv_cols = temp_derivative_f.cols();
                output_derivative_f.resize(deriv_rows, deriv_cols);
                for (int r = 0; r < deriv_rows; ++r) {
                    for (int c = 0; c < deriv_cols; ++c) {
                        output_derivative_f(r, c) = matvec_row_adder_tree(
                            mlp_.weight[layer].row(r), temp_derivative_f.col(c), 0.0f);
                    }
                }
            }
            else // hidden layers
            {
                const int n_neurons = static_cast<int>(mlp_.n_hidden(layer));
                hidden_f[layer].resize(n_neurons);
                pre_activation_f[layer].resize(n_neurons);
                hidden_derivative_f[layer].resize(n_neurons, static_cast<int>(mlp_.n_hidden(layer - 1)));

                // Adder tree를 사용한 은닉층 계산
                for (int h = 0; h < n_neurons; ++h) {
                    pre_activation_f[layer](h) = matvec_row_adder_tree(
                        mlp_.weight[layer].row(h), hidden_f[layer - 1], mlp_.bias[layer](h));
                }

                for (int h = 0; h < n_neurons; h++)
                {
                    float relu_deriv = (pre_activation_f[layer](h) > 0.0f) ? 1.0f : 0.0f;
                    hidden_derivative_f[layer].row(h) = relu_deriv * mlp_.weight[layer].row(h);
                    hidden_f[layer](h) = std::max(0.0f, pre_activation_f[layer](h));
                }

                // Jacobian 계산 (adder tree 사용)
                const int jac_rows = hidden_derivative_f[layer].rows();
                const int jac_cols = temp_derivative_f.cols();
                Eigen::MatrixXf new_temp_derivative_f(jac_rows, jac_cols);
                for (int r = 0; r < jac_rows; ++r) {
                    for (int c = 0; c < jac_cols; ++c) {
                        new_temp_derivative_f(r, c) = matvec_row_adder_tree(
                            hidden_derivative_f[layer].row(r), temp_derivative_f.col(c), 0.0f);
                    }
                }
                temp_derivative_f = std::move(new_temp_derivative_f);
            }
        }

        // ===== Batch logging: 데이터 누적 (OSQP solve_count와 동기화) =====
        if (log_solve_count_ >= 0 && log_horizon_idx_ < log_batch_size_) {
            const Eigen::VectorXf& current_input_log = mlp_.is_nerf ? input_nerf_f : input_f;
            auto& entry = log_entries_[log_horizon_idx_];
            entry.nerf_input = current_input_log;
            entry.output = output_f;
            entry.jacobian = output_derivative_f;
            if (mlp_.is_nerf) entry.nerf_jac = nerf_jac_f;
            entry.pre_activations.resize(mlp_.n_layer - 1);
            entry.post_activations.resize(mlp_.n_layer - 1);
            for (int ll = 0; ll < mlp_.n_layer - 1; ++ll) {
                entry.pre_activations[ll] = pre_activation_f[ll];
                entry.post_activations[ll] = hidden_f[ll];
            }
            log_horizon_idx_++;

            // 배치 완료 시 파일 저장
            if (log_horizon_idx_ == log_batch_size_) {
                writeBatchLog();
            }
        }

        // Convert output from float to double for interface compatibility
#ifdef NN_ROUNDING_MODE
        std::fesetround(old_round);
#endif
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
#ifdef NN_ROUNDING_MODE
        int old_round_batch = std::fegetround();
        std::fesetround(NN_ROUNDING_MODE);
#endif

        const int batch_size = inputs.cols();
        if (batch_size == 0) {
#ifdef NN_ROUNDING_MODE
            std::fesetround(old_round_batch);
#endif
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

        // ===== 1. 입력 준비 =====
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

        // ===== 2. Forward pass with adder tree =====
        for (int layer = 0; layer < mlp_.n_layer; ++layer) {
            if (layer == 0) {
                const int n_neurons = static_cast<int>(mlp_.n_hidden(0));
                const int n_inputs = current_input_ptr->rows();
                mlp_.pre_activations[0].resize(n_neurons, batch_size);

                // Adder tree를 사용한 행렬-행렬 곱셈
                for (int b = 0; b < batch_size; ++b) {
                    for (int h = 0; h < n_neurons; ++h) {
                        mlp_.pre_activations[0](h, b) = matvec_row_adder_tree(
                            mlp_.weight[0].row(h), current_input_ptr->col(b), mlp_.bias[0](h));
                    }
                }
                mlp_.batch_hidden[0] = batch_ReLU_f(mlp_.pre_activations[0]);
            }
            else if (layer == mlp_.n_layer - 1) {
                const int n_outputs = mlp_.n_output;
                mlp_.batch_output.resize(n_outputs, batch_size);

                // Adder tree를 사용한 출력층 계산
                for (int b = 0; b < batch_size; ++b) {
                    for (int o = 0; o < n_outputs; ++o) {
                        mlp_.batch_output(o, b) = matvec_row_adder_tree(
                            mlp_.weight[layer].row(o), mlp_.batch_hidden[layer - 1].col(b), mlp_.bias[layer](o));
                    }
                }
            }
            else {
                const int n_neurons = static_cast<int>(mlp_.n_hidden(layer));
                mlp_.pre_activations[layer].resize(n_neurons, batch_size);

                // Adder tree를 사용한 은닉층 계산
                for (int b = 0; b < batch_size; ++b) {
                    for (int h = 0; h < n_neurons; ++h) {
                        mlp_.pre_activations[layer](h, b) = matvec_row_adder_tree(
                            mlp_.weight[layer].row(h), mlp_.batch_hidden[layer - 1].col(b), mlp_.bias[layer](h));
                    }
                }
                mlp_.batch_hidden[layer] = batch_ReLU_f(mlp_.pre_activations[layer]);
            }
        }

        // ===== 3. 자코비안 계산 (adder tree 사용) =====
        for (int i = 0; i < batch_size; ++i) {
            Eigen::MatrixXf temp_derivative;
            if (mlp_.is_nerf) {
                Eigen::MatrixXf nerf_jac(3 * mlp_.n_input, mlp_.n_input);
                nerf_jac.setZero();
                nerf_jac.topRows(mlp_.n_input).setIdentity();
                nerf_jac.middleRows(mlp_.n_input, mlp_.n_input).diagonal() = inputs.col(i).array().cos().cast<float>();
                nerf_jac.bottomRows(mlp_.n_input).diagonal() = (-inputs.col(i).array().sin()).cast<float>();
                Eigen::MatrixXf relu_deriv_0 = batch_ReLU_derivative_f(mlp_.pre_activations[0].col(i));
                Eigen::MatrixXf weight_scaled = relu_deriv_0.asDiagonal() * mlp_.weight[0];

                // Adder tree로 행렬-행렬 곱셈
                const int jac_rows = weight_scaled.rows();
                const int jac_cols = nerf_jac.cols();
                temp_derivative.resize(jac_rows, jac_cols);
                for (int r = 0; r < jac_rows; ++r) {
                    for (int c = 0; c < jac_cols; ++c) {
                        temp_derivative(r, c) = matvec_row_adder_tree(
                            weight_scaled.row(r), nerf_jac.col(c), 0.0f);
                    }
                }
            } else {
                Eigen::MatrixXf relu_deriv_0 = batch_ReLU_derivative_f(mlp_.pre_activations[0].col(i));
                temp_derivative = relu_deriv_0.asDiagonal() * mlp_.weight[0];
            }

            for (int layer = 1; layer < mlp_.n_layer - 1; ++layer) {
                Eigen::MatrixXf relu_deriv = batch_ReLU_derivative_f(mlp_.pre_activations[layer].col(i));
                Eigen::MatrixXf weight_scaled = relu_deriv.asDiagonal() * mlp_.weight[layer];

                // Adder tree로 행렬-행렬 곱셈
                const int jac_rows = weight_scaled.rows();
                const int jac_cols = temp_derivative.cols();
                Eigen::MatrixXf new_temp(jac_rows, jac_cols);
                for (int r = 0; r < jac_rows; ++r) {
                    for (int c = 0; c < jac_cols; ++c) {
                        new_temp(r, c) = matvec_row_adder_tree(
                            weight_scaled.row(r), temp_derivative.col(c), 0.0f);
                    }
                }
                temp_derivative = std::move(new_temp);
            }

            // 출력층 Jacobian
            const int out_rows = mlp_.weight.back().rows();
            const int out_cols = temp_derivative.cols();
            mlp_.batch_jacobian[i].resize(out_rows, out_cols);
            for (int r = 0; r < out_rows; ++r) {
                for (int c = 0; c < out_cols; ++c) {
                    mlp_.batch_jacobian[i](r, c) = matvec_row_adder_tree(
                        mlp_.weight.back().row(r), temp_derivative.col(c), 0.0f);
                }
            }
        }

        // ===== 4. 각 행(링크)별 최소값 탐색 및 결과 재구성 =====
        for (int i = 0; i < mlp_.n_output; ++i)
        {
            // 최소값 인덱스 찾기
            Eigen::Index min_col;
            mlp_.batch_output.row(i).minCoeff(&min_col);

            // 최소값과 해당 jacobian 저장
            mlp_.final_min_output(i) = mlp_.batch_output(i, min_col);
            mlp_.final_min_jacobian.row(i) = mlp_.batch_jacobian[min_col].row(i);
        }

        // ===== 5. ASIC 테스트 케이스 저장 (주석처리됨) =====
        /*
        init_asic_output_dir();

        // 각 배치 sample에 대해 개별 파일로 저장
        for (int b = 0; b < batch_size; ++b) {
            int sample_id = g_asic_sample_count++;
            bool is_detailed = (sample_id % 100 == 0);  // 100번째 sample마다 상세 저장

            std::string filename = g_asic_output_dir + "sample_" + std::to_string(sample_id) + ".csv";
            std::ofstream file(filename);

            if (file.is_open()) {
                file << std::scientific << std::setprecision(8);

                // 헤더 정보
                file << "# ASIC Test Case - Sample " << sample_id << "\n";
                file << "# Detailed: " << (is_detailed ? "YES" : "NO") << "\n";
                file << "# n_input: " << mlp_.n_input << ", n_output: " << mlp_.n_output << "\n";
                file << "# n_hidden_layers: " << (mlp_.n_layer - 1) << "\n\n";

                // 1. 입력 (nerf 적용된 값: [x, sin(x), cos(x)])
                file << "# INPUT (nerf_input) - " << current_input_ptr->rows() << " values\n";
                for (int i = 0; i < current_input_ptr->rows(); ++i) {
                    file << (*current_input_ptr)(i, b);
                    if (i < current_input_ptr->rows() - 1) file << ",";
                }
                file << "\n\n";

                // 2. 최종 출력 (output layer)
                file << "# OUTPUT - " << mlp_.n_output << " values\n";
                for (int i = 0; i < mlp_.n_output; ++i) {
                    file << mlp_.batch_output(i, b);
                    if (i < mlp_.n_output - 1) file << ",";
                }
                file << "\n\n";

                // 3. Jacobian (output x input)
                file << "# JACOBIAN - " << mlp_.n_output << " x " << mlp_.n_input << "\n";
                for (int r = 0; r < mlp_.batch_jacobian[b].rows(); ++r) {
                    for (int c = 0; c < mlp_.batch_jacobian[b].cols(); ++c) {
                        file << mlp_.batch_jacobian[b](r, c);
                        if (c < mlp_.batch_jacobian[b].cols() - 1) file << ",";
                    }
                    file << "\n";
                }
                file << "\n";

                // 4. 100번째 sample마다 상세 정보 저장
                if (is_detailed) {
                    // 각 hidden layer의 pre_activation과 post_activation (ReLU 후)
                    for (int layer = 0; layer < mlp_.n_layer - 1; ++layer) {
                        // Pre-activation (ReLU 전)
                        file << "# LAYER_" << layer << "_PRE_ACTIVATION - " << mlp_.pre_activations[layer].rows() << " values\n";
                        for (int h = 0; h < mlp_.pre_activations[layer].rows(); ++h) {
                            file << mlp_.pre_activations[layer](h, b);
                            if (h < mlp_.pre_activations[layer].rows() - 1) file << ",";
                        }
                        file << "\n\n";

                        // Post-activation (ReLU 후)
                        file << "# LAYER_" << layer << "_POST_ACTIVATION - " << mlp_.batch_hidden[layer].rows() << " values\n";
                        for (int h = 0; h < mlp_.batch_hidden[layer].rows(); ++h) {
                            file << mlp_.batch_hidden[layer](h, b);
                            if (h < mlp_.batch_hidden[layer].rows() - 1) file << ",";
                        }
                        file << "\n\n";
                    }

                    // 각 layer별 intermediate jacobian 계산 및 저장
                    file << "# === INTERMEDIATE JACOBIANS ===\n\n";

                    Eigen::MatrixXf temp_derivative_log;
                    if (mlp_.is_nerf) {
                        Eigen::MatrixXf nerf_jac(3 * mlp_.n_input, mlp_.n_input);
                        nerf_jac.setZero();
                        nerf_jac.topRows(mlp_.n_input).setIdentity();
                        nerf_jac.middleRows(mlp_.n_input, mlp_.n_input).diagonal() = inputs.col(b).array().cos().cast<float>();
                        nerf_jac.bottomRows(mlp_.n_input).diagonal() = (-inputs.col(b).array().sin()).cast<float>();

                        file << "# NERF_JACOBIAN - " << nerf_jac.rows() << " x " << nerf_jac.cols() << "\n";
                        for (int r = 0; r < nerf_jac.rows(); ++r) {
                            for (int c = 0; c < nerf_jac.cols(); ++c) {
                                file << nerf_jac(r, c);
                                if (c < nerf_jac.cols() - 1) file << ",";
                            }
                            file << "\n";
                        }
                        file << "\n";

                        Eigen::MatrixXf relu_deriv_0 = batch_ReLU_derivative_f(mlp_.pre_activations[0].col(b));
                        Eigen::MatrixXf weight_scaled = relu_deriv_0.asDiagonal() * mlp_.weight[0];

                        const int jac_rows = weight_scaled.rows();
                        const int jac_cols = nerf_jac.cols();
                        temp_derivative_log.resize(jac_rows, jac_cols);
                        for (int r = 0; r < jac_rows; ++r) {
                            for (int c = 0; c < jac_cols; ++c) {
                                temp_derivative_log(r, c) = matvec_row_adder_tree(
                                    weight_scaled.row(r), nerf_jac.col(c), 0.0f);
                            }
                        }
                    } else {
                        Eigen::MatrixXf relu_deriv_0 = batch_ReLU_derivative_f(mlp_.pre_activations[0].col(b));
                        temp_derivative_log = relu_deriv_0.asDiagonal() * mlp_.weight[0];
                    }

                    file << "# JACOBIAN_AFTER_LAYER_0 - " << temp_derivative_log.rows() << " x " << temp_derivative_log.cols() << "\n";
                    for (int r = 0; r < temp_derivative_log.rows(); ++r) {
                        for (int c = 0; c < temp_derivative_log.cols(); ++c) {
                            file << temp_derivative_log(r, c);
                            if (c < temp_derivative_log.cols() - 1) file << ",";
                        }
                        file << "\n";
                    }
                    file << "\n";

                    for (int layer = 1; layer < mlp_.n_layer - 1; ++layer) {
                        Eigen::MatrixXf relu_deriv = batch_ReLU_derivative_f(mlp_.pre_activations[layer].col(b));
                        Eigen::MatrixXf weight_scaled = relu_deriv.asDiagonal() * mlp_.weight[layer];

                        const int jac_rows = weight_scaled.rows();
                        const int jac_cols = temp_derivative_log.cols();
                        Eigen::MatrixXf new_temp(jac_rows, jac_cols);
                        for (int r = 0; r < jac_rows; ++r) {
                            for (int c = 0; c < jac_cols; ++c) {
                                new_temp(r, c) = matvec_row_adder_tree(
                                    weight_scaled.row(r), temp_derivative_log.col(c), 0.0f);
                            }
                        }
                        temp_derivative_log = std::move(new_temp);

                        file << "# JACOBIAN_AFTER_LAYER_" << layer << " - " << temp_derivative_log.rows() << " x " << temp_derivative_log.cols() << "\n";
                        for (int r = 0; r < temp_derivative_log.rows(); ++r) {
                            for (int c = 0; c < temp_derivative_log.cols(); ++c) {
                                file << temp_derivative_log(r, c);
                                if (c < temp_derivative_log.cols() - 1) file << ",";
                            }
                            file << "\n";
                        }
                        file << "\n";
                    }
                }

                file.close();
            }
        }
        */

        // Convert output from float to double for interface compatibility
#ifdef NN_ROUNDING_MODE
        std::fesetround(old_round_batch);
#endif
        return std::make_pair(mlp_.final_min_output.cast<double>(), mlp_.final_min_jacobian.cast<double>());
    }

    // ===================================================================
    // =========== Batch Log 저장 (OSQP solve_count 동기화) ==============
    // ===================================================================
    void EnvCollNNmodel::writeBatchLog() {
        try {
            // 첫 번째 호출 시 로그 출력
            if (log_solve_count_ == 0) {
                init_asic_output_dir();
                std::cout << "[MLP Batch Logging] Starting batch logging to: " << g_asic_output_dir << std::endl;
            }

            // MLP_TESTCASE_SAVE_INTERVAL 기준으로 저장 여부 결정 (solve_count 기준)
            if (log_solve_count_ % MLP_TESTCASE_SAVE_INTERVAL != 0) return;

            init_asic_output_dir();
            bool is_detailed = (log_solve_count_ % 100 == 0);

            std::string filename = g_asic_output_dir + "sample_" + std::to_string(log_solve_count_) + ".csv";
            std::string filename_bits = g_asic_output_dir + "sample_" + std::to_string(log_solve_count_) + "_bits.csv";

            std::ofstream file(filename);
            std::ofstream file_bits(filename_bits);

            if (!file.is_open() || !file_bits.is_open()) {
                if (log_solve_count_ <= 1) {
                    std::cerr << "[MLP Batch Logging ERROR] Failed to open file: " << filename << std::endl;
                }
                return;
            }

            // ===== 실수 값 파일 헤더 =====
            file << std::scientific << std::setprecision(8);
            file << "# MLP Batch Test Case - Solve Count " << log_solve_count_ << "\n";
            file << "# Batch Size (Horizon): " << log_batch_size_ << "\n";
            file << "# Detailed: " << (is_detailed ? "YES" : "NO") << "\n";
            file << "# n_input: " << mlp_.n_input << ", n_output: " << mlp_.n_output << "\n";
            file << "# n_hidden_layers: " << (mlp_.n_layer - 1) << "\n\n";

            // ===== FP32 비트 파일 헤더 =====
            file_bits << "# MLP Batch Test Case (FP32 Bits) - Solve Count " << log_solve_count_ << "\n";
            file_bits << "# Batch Size (Horizon): " << log_batch_size_ << "\n";
            file_bits << "# Detailed: " << (is_detailed ? "YES" : "NO") << "\n";
            file_bits << "# n_input: " << mlp_.n_input << ", n_output: " << mlp_.n_output << "\n";
            file_bits << "# n_hidden_layers: " << (mlp_.n_layer - 1) << "\n";
            file_bits << "# Format: 8-digit hexadecimal (32-bit IEEE 754)\n\n";

            // ===== 각 Horizon point별 데이터 저장 =====
            for (int h = 0; h < log_batch_size_; ++h) {
                const auto& entry = log_entries_[h];

                // --- INPUT ---
                file << "# HORIZON_" << h << "_INPUT - " << entry.nerf_input.size() << " values\n";
                for (int i = 0; i < entry.nerf_input.size(); ++i) {
                    file << entry.nerf_input(i);
                    if (i < entry.nerf_input.size() - 1) file << ",";
                }
                file << "\n\n";

                file_bits << "# HORIZON_" << h << "_INPUT - " << entry.nerf_input.size() << " values\n";
                for (int i = 0; i < entry.nerf_input.size(); ++i) {
                    file_bits << float_to_hex(entry.nerf_input(i));
                    if (i < entry.nerf_input.size() - 1) file_bits << ",";
                }
                file_bits << "\n\n";

                // --- OUTPUT ---
                file << "# HORIZON_" << h << "_OUTPUT - " << entry.output.size() << " values\n";
                for (int i = 0; i < entry.output.size(); ++i) {
                    file << entry.output(i);
                    if (i < entry.output.size() - 1) file << ",";
                }
                file << "\n\n";

                file_bits << "# HORIZON_" << h << "_OUTPUT - " << entry.output.size() << " values\n";
                for (int i = 0; i < entry.output.size(); ++i) {
                    file_bits << float_to_hex(entry.output(i));
                    if (i < entry.output.size() - 1) file_bits << ",";
                }
                file_bits << "\n\n";

                // --- JACOBIAN ---
                file << "# HORIZON_" << h << "_JACOBIAN - " << entry.jacobian.rows() << " x " << entry.jacobian.cols() << "\n";
                for (int r = 0; r < entry.jacobian.rows(); ++r) {
                    for (int c = 0; c < entry.jacobian.cols(); ++c) {
                        file << entry.jacobian(r, c);
                        if (c < entry.jacobian.cols() - 1) file << ",";
                    }
                    file << "\n";
                }
                file << "\n";

                file_bits << "# HORIZON_" << h << "_JACOBIAN - " << entry.jacobian.rows() << " x " << entry.jacobian.cols() << "\n";
                for (int r = 0; r < entry.jacobian.rows(); ++r) {
                    for (int c = 0; c < entry.jacobian.cols(); ++c) {
                        file_bits << float_to_hex(entry.jacobian(r, c));
                        if (c < entry.jacobian.cols() - 1) file_bits << ",";
                    }
                    file_bits << "\n";
                }
                file_bits << "\n";

                // --- NERF_JACOBIAN ---
                if (mlp_.is_nerf && entry.nerf_jac.size() > 0) {
                    file << "# HORIZON_" << h << "_NERF_JACOBIAN - " << entry.nerf_jac.rows() << " x " << entry.nerf_jac.cols() << "\n";
                    for (int r = 0; r < entry.nerf_jac.rows(); ++r) {
                        for (int c = 0; c < entry.nerf_jac.cols(); ++c) {
                            file << entry.nerf_jac(r, c);
                            if (c < entry.nerf_jac.cols() - 1) file << ",";
                        }
                        file << "\n";
                    }
                    file << "\n";

                    file_bits << "# HORIZON_" << h << "_NERF_JACOBIAN - " << entry.nerf_jac.rows() << " x " << entry.nerf_jac.cols() << "\n";
                    for (int r = 0; r < entry.nerf_jac.rows(); ++r) {
                        for (int c = 0; c < entry.nerf_jac.cols(); ++c) {
                            file_bits << float_to_hex(entry.nerf_jac(r, c));
                            if (c < entry.nerf_jac.cols() - 1) file_bits << ",";
                        }
                        file_bits << "\n";
                    }
                    file_bits << "\n";
                }

                // --- DETAILED: Pre/Post Activation ---
                if (is_detailed) {
                    for (int layer = 0; layer < static_cast<int>(entry.pre_activations.size()); ++layer) {
                        // Pre-activation
                        file << "# HORIZON_" << h << "_LAYER_" << layer << "_PRE_ACTIVATION - " << entry.pre_activations[layer].size() << " values\n";
                        for (int i = 0; i < entry.pre_activations[layer].size(); ++i) {
                            file << entry.pre_activations[layer](i);
                            if (i < entry.pre_activations[layer].size() - 1) file << ",";
                        }
                        file << "\n\n";

                        file_bits << "# HORIZON_" << h << "_LAYER_" << layer << "_PRE_ACTIVATION - " << entry.pre_activations[layer].size() << " values\n";
                        for (int i = 0; i < entry.pre_activations[layer].size(); ++i) {
                            file_bits << float_to_hex(entry.pre_activations[layer](i));
                            if (i < entry.pre_activations[layer].size() - 1) file_bits << ",";
                        }
                        file_bits << "\n\n";

                        // Post-activation
                        file << "# HORIZON_" << h << "_LAYER_" << layer << "_POST_ACTIVATION - " << entry.post_activations[layer].size() << " values\n";
                        for (int i = 0; i < entry.post_activations[layer].size(); ++i) {
                            file << entry.post_activations[layer](i);
                            if (i < entry.post_activations[layer].size() - 1) file << ",";
                        }
                        file << "\n\n";

                        file_bits << "# HORIZON_" << h << "_LAYER_" << layer << "_POST_ACTIVATION - " << entry.post_activations[layer].size() << " values\n";
                        for (int i = 0; i < entry.post_activations[layer].size(); ++i) {
                            file_bits << float_to_hex(entry.post_activations[layer](i));
                            if (i < entry.post_activations[layer].size() - 1) file_bits << ",";
                        }
                        file_bits << "\n\n";
                    }
                }
            }

            file.close();
            file_bits.close();

            // ===== Per-layer detail files for sample_0 =====
            if (log_solve_count_ == 0) {
                // Note: rounding mode is still active (called from calculateMlpOutput)
                std::vector<std::ofstream> lf(mlp_.n_layer);
                std::vector<std::ofstream> lf_bits(mlp_.n_layer);

                for (int layer = 0; layer < mlp_.n_layer; ++layer) {
                    lf[layer].open(g_asic_output_dir + "sample_0_layer_" + std::to_string(layer) + ".csv");
                    lf_bits[layer].open(g_asic_output_dir + "sample_0_layer_" + std::to_string(layer) + "_bits.csv");
                    if (!lf[layer].is_open() || !lf_bits[layer].is_open()) {
                        std::cerr << "[MLP Batch Logging ERROR] Failed to open layer file for layer " << layer << std::endl;
                        continue;
                    }
                    lf[layer] << std::scientific << std::setprecision(8);

                    lf[layer] << "# MLP Layer " << layer << " Detail - Sample 0\n";
                    lf[layer] << "# Batch Size (Horizon): " << log_batch_size_ << "\n";
                    if (layer < mlp_.n_layer - 1) {
                        lf[layer] << "# Layer type: hidden, neurons: " << static_cast<int>(mlp_.n_hidden(layer)) << "\n\n";
                    } else {
                        lf[layer] << "# Layer type: output, neurons: " << mlp_.n_output << "\n\n";
                    }

                    lf_bits[layer] << "# MLP Layer " << layer << " Detail (FP32 Bits) - Sample 0\n";
                    lf_bits[layer] << "# Batch Size (Horizon): " << log_batch_size_ << "\n";
                    lf_bits[layer] << "# Format: 8-digit hexadecimal (32-bit IEEE 754)\n\n";
                }

                for (int h = 0; h < log_batch_size_; ++h) {
                    const auto& entry = log_entries_[h];
                    Eigen::MatrixXf temp_derivative_log;

                    for (int layer = 0; layer < mlp_.n_layer; ++layer) {
                        auto& f = lf[layer];
                        auto& fb = lf_bits[layer];
                        if (!f.is_open()) continue;

                        if (layer < mlp_.n_layer - 1) {
                            // Hidden layer: pre/post activation
                            const auto& pre = entry.pre_activations[layer];
                            const auto& post = entry.post_activations[layer];

                            f << "# HORIZON_" << h << "_PRE_ACTIVATION - " << pre.size() << " values\n";
                            fb << "# HORIZON_" << h << "_PRE_ACTIVATION - " << pre.size() << " values\n";
                            for (int i = 0; i < pre.size(); ++i) {
                                f << pre(i); fb << float_to_hex(pre(i));
                                if (i < pre.size() - 1) { f << ","; fb << ","; }
                            }
                            f << "\n\n"; fb << "\n\n";

                            f << "# HORIZON_" << h << "_POST_ACTIVATION - " << post.size() << " values\n";
                            fb << "# HORIZON_" << h << "_POST_ACTIVATION - " << post.size() << " values\n";
                            for (int i = 0; i < post.size(); ++i) {
                                f << post(i); fb << float_to_hex(post(i));
                                if (i < post.size() - 1) { f << ","; fb << ","; }
                            }
                            f << "\n\n"; fb << "\n\n";

                            // Recompute intermediate Jacobian up to this layer
                            if (layer == 0) {
                                Eigen::VectorXf relu_deriv(pre.size());
                                for (int i = 0; i < relu_deriv.size(); ++i)
                                    relu_deriv(i) = (pre(i) > 0.0f) ? 1.0f : 0.0f;
                                Eigen::MatrixXf weight_scaled = relu_deriv.asDiagonal() * mlp_.weight[0];

                                if (mlp_.is_nerf && entry.nerf_jac.size() > 0) {
                                    const int jr = weight_scaled.rows(), jc = entry.nerf_jac.cols();
                                    temp_derivative_log.resize(jr, jc);
                                    for (int r = 0; r < jr; ++r)
                                        for (int c = 0; c < jc; ++c)
                                            temp_derivative_log(r, c) = matvec_row_adder_tree(
                                                weight_scaled.row(r), entry.nerf_jac.col(c), 0.0f);
                                } else {
                                    temp_derivative_log = weight_scaled;
                                }
                            } else {
                                Eigen::VectorXf relu_deriv(pre.size());
                                for (int i = 0; i < relu_deriv.size(); ++i)
                                    relu_deriv(i) = (pre(i) > 0.0f) ? 1.0f : 0.0f;
                                Eigen::MatrixXf weight_scaled = relu_deriv.asDiagonal() * mlp_.weight[layer];

                                const int jr = weight_scaled.rows(), jc = temp_derivative_log.cols();
                                Eigen::MatrixXf new_temp(jr, jc);
                                for (int r = 0; r < jr; ++r)
                                    for (int c = 0; c < jc; ++c)
                                        new_temp(r, c) = matvec_row_adder_tree(
                                            weight_scaled.row(r), temp_derivative_log.col(c), 0.0f);
                                temp_derivative_log = std::move(new_temp);
                            }
                        } else {
                            // Output layer: output values
                            f << "# HORIZON_" << h << "_OUTPUT - " << entry.output.size() << " values\n";
                            fb << "# HORIZON_" << h << "_OUTPUT - " << entry.output.size() << " values\n";
                            for (int i = 0; i < entry.output.size(); ++i) {
                                f << entry.output(i); fb << float_to_hex(entry.output(i));
                                if (i < entry.output.size() - 1) { f << ","; fb << ","; }
                            }
                            f << "\n\n"; fb << "\n\n";

                            // Output Jacobian from stored entry (exact inference result)
                            temp_derivative_log = entry.jacobian;
                        }

                        // Write intermediate Jacobian for this layer
                        f << "# HORIZON_" << h << "_JACOBIAN - " << temp_derivative_log.rows() << " x " << temp_derivative_log.cols() << "\n";
                        fb << "# HORIZON_" << h << "_JACOBIAN - " << temp_derivative_log.rows() << " x " << temp_derivative_log.cols() << "\n";
                        for (int r = 0; r < temp_derivative_log.rows(); ++r) {
                            for (int c = 0; c < temp_derivative_log.cols(); ++c) {
                                f << temp_derivative_log(r, c); fb << float_to_hex(temp_derivative_log(r, c));
                                if (c < temp_derivative_log.cols() - 1) { f << ","; fb << ","; }
                            }
                            f << "\n"; fb << "\n";
                        }
                        f << "\n"; fb << "\n";
                    }
                }

                for (int layer = 0; layer < mlp_.n_layer; ++layer) {
                    if (lf[layer].is_open()) lf[layer].close();
                    if (lf_bits[layer].is_open()) lf_bits[layer].close();
                }
                std::cout << "[MLP Batch Logging] Per-layer detail files saved for sample_0 ("
                          << mlp_.n_layer << " layers)" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[MLP Batch Logging ERROR] Exception: " << e.what() << std::endl;
        }
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



