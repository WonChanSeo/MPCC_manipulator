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
#include <Eigen/src/Core/arch/Default/BFloat16.h>

namespace mpcc
{
    // Type aliases for bfloat16 matrices and vectors
    using MatrixXbf16 = Eigen::Matrix<Eigen::bfloat16, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXbf16 = Eigen::Matrix<Eigen::bfloat16, Eigen::Dynamic, 1>;
    class EnvCollNNmodel
    {
        struct MLP
        {
            ~MLP() { std::cout << "MLP terminate" << std::endl; }
            // --- 신경망 파라미터 (bfloat16 precision) ---
            std::vector<MatrixXbf16> weight;
            std::vector<VectorXbf16> bias;
            std::vector<std::string> w_path;
            std::vector<std::string> b_path;
            std::vector<std::ifstream> weight_files;
            std::vector<std::ifstream> bias_files;

            // --- 신경망 구조 ---
            int n_input;
            int n_output;
            Eigen::VectorXd n_hidden;
            int n_layer;

            // --- 단일 추론(Single Inference)용 변수 (bfloat16 precision) ---
            std::vector<VectorXbf16> hidden;
            std::vector<MatrixXbf16> hidden_derivative;
            VectorXbf16 input;
            VectorXbf16 output;
            MatrixXbf16 output_derivative;

            // --- NeRF 관련 ---
            bool is_nerf;
            VectorXbf16 input_nerf;

            // --- 배치 추론(Batch Inference)용 변수 (bfloat16 precision) ---
            std::vector<MatrixXbf16> batch_hidden;
            MatrixXbf16 batch_input;
            MatrixXbf16 batch_input_nerf;
            MatrixXbf16 batch_output;

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
        private:
            std::string file_path_;
            MLP mlp_;

            void readWeightFile(int weight_num);
            void readBiasFile(int bias_num);
            void loadNetwork();
            void initializeNetwork(int n_input, int n_output, Eigen::VectorXd n_hidden, bool is_nerf);

            // ReLU는 private 멤버로 유지 (bfloat16 precision)
            Eigen::bfloat16 ReLU(Eigen::bfloat16 input)
            {
                return (input > Eigen::bfloat16(0)) ? input : Eigen::bfloat16(0);
            }
            Eigen::bfloat16 ReLU_derivative(Eigen::bfloat16 input)
            {
                return (input > Eigen::bfloat16(0)) ? Eigen::bfloat16(1) : Eigen::bfloat16(0);
            }
    };
}

#endif