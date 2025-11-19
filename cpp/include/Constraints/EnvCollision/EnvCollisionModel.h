#ifndef MPCC_ENV_COLLISION_H
#define MPCC_ENV_COLLISION_H

#include <config.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <Eigen/Dense>

namespace mpcc
{
    // Type aliases for float matrices and vectors (FP32)
    using MatrixXf32 = Eigen::MatrixXf;
    using VectorXf32 = Eigen::VectorXf;
    class EnvCollNNmodel
    {
        struct MLP
        {
            ~MLP() { std::cout << "MLP terminate" << std::endl; }
            // --- 신경망 파라미터 (FP32 precision) ---
            std::vector<MatrixXf32> weight;
            std::vector<VectorXf32> bias;
            std::vector<std::string> w_path;
            std::vector<std::string> b_path;
            std::vector<std::ifstream> weight_files;
            std::vector<std::ifstream> bias_files;

            // --- 신경망 구조 ---
            int n_input;
            int n_output;
            Eigen::VectorXd n_hidden;
            int n_layer;

            // --- 단일 추론(Single Inference)용 변수 (FP32 precision) ---
            std::vector<VectorXf32> hidden;
            std::vector<MatrixXf32> hidden_derivative;
            VectorXf32 input;
            VectorXf32 output;
            MatrixXf32 output_derivative;

            // --- NeRF 관련 ---
            bool is_nerf;
            VectorXf32 input_nerf;

            // --- 배치 추론(Batch Inference)용 변수 (FP32 precision) ---
            std::vector<MatrixXf32> batch_hidden;
            MatrixXf32 batch_input;
            MatrixXf32 batch_input_nerf;
            MatrixXf32 batch_output;
            std::vector<MatrixXf32> pre_activations;
            std::vector<MatrixXf32> batch_jacobian;

            // ▼▼▼▼▼ 여기에 시간 저장을 위한 변수를 추가합니다. ▼▼▼▼▼
            std::vector<double> inference_times_ms;

            // ▼▼▼▼▼ 여기에 이전 인덱스 저장 변수를 추가합니다. ▼▼▼▼▼
            Eigen::Index previous_min_col = -1; // 초기값 -1

            // ▼▼▼▼▼ ReLU deactivation 통계를 저장하는 변수들 ▼▼▼▼▼
            // 각 레이어별 총 유닛 수를 저장 (n_layer-1 크기, output layer 제외)
            std::vector<int> total_units_per_layer;
            // 각 레이어별 deactivated 유닛 수의 합계 (누적)
            std::vector<int> deactivated_units_per_layer;
            // 각 레이어별 최소 deactivation 개수
            std::vector<int> min_deactivated_per_layer;
            // 각 레이어별 최대 deactivation 개수
            std::vector<int> max_deactivated_per_layer;
            // 총 inference 호출 횟수 (통계 계산용)
            int total_inference_count = 0;
            // 총 처리된 샘플 수 (batch_size의 합계)
            int total_sample_count = 0;
            

            // --- 기타 설정 ---
            bool loadweightfile_verbose = false;
            bool loadbiasfile_verbose = false;
        };
        
        public:
            EnvCollNNmodel();
            EnvCollNNmodel(const std::string & file_path);
            ~EnvCollNNmodel();

            void setNeuralNetwork(int n_input, int n_output, Eigen::VectorXd n_hidden, bool is_nerf);
            
            // 기존: 단일 입력을 처리하는 함수
            std::pair<Eigen::VectorXd, Eigen::MatrixXd> calculateMlpOutput(Eigen::VectorXd input, bool time_verbose = false);
            
            // 추가: 여러 입력을 한 번에 처리하는 배치 추론 함수
            std::pair<Eigen::VectorXd, Eigen::MatrixXd> calculateMlpOutputBatch(const Eigen::MatrixXd& inputs, bool time_verbose = false);

            // ▼▼▼▼▼ 여기에 상태 플래그를 추가합니다. ▼▼▼▼▼
            bool obstacle_switched = false;
            
            // ▼▼▼▼▼ 저장된 시간들을 반환하는 getter 함수를 선언합니다. ▼▼▼▼▼
            const std::vector<double>& getInferenceTimes() const;

            // ▼▼▼▼▼ ReLU deactivation 통계를 반환하는 getter 함수들 ▼▼▼▼▼
            std::vector<double> getReluDeactivationRatios() const;
            std::vector<int> getReluTotalUnits() const;
            std::vector<double> getReluAvgDeactivatedCounts() const;
            std::vector<int> getReluMinDeactivatedCounts() const;
            std::vector<int> getReluMaxDeactivatedCounts() const;

            // ▼▼▼▼▼ Input/Output logging functions ▼▼▼▼▼
            void saveInputOutputLog(const std::string& filename, const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& outputs) const;
            void enableLogging(bool enable);
            bool isLoggingEnabled() const;
            Eigen::MatrixXd getBatchOutput() const;  // Get full batch output (before min selection)
        private:
            std::string file_path_;
            MLP mlp_;
            bool logging_enabled_ = false;

            void readWeightFile(int weight_num);
            void readBiasFile(int bias_num);
            void loadNetwork();
            void initializeNetwork(int n_input, int n_output, Eigen::VectorXd n_hidden, bool is_nerf);

            // ReLU는 private 멤버로 유지 (FP32 precision)
            float ReLU(float input)
            {
                return (input > 0.0f) ? input : 0.0f;
            }
            float ReLU_derivative(float input)
            {
                return (input > 0.0f) ? 1.0f : 0.0f;
            }
    };
}

#endif