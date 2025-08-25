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
    class EnvCollNNmodel
    {
        struct MLP
        {
            ~MLP() { std::cout << "MLP terminate" << std::endl; }
            // --- 신경망 파라미터 ---
            std::vector<Eigen::MatrixXd> weight;
            std::vector<Eigen::VectorXd> bias;
            std::vector<std::string> w_path;
            std::vector<std::string> b_path;
            std::vector<std::ifstream> weight_files;
            std::vector<std::ifstream> bias_files;

            // --- 신경망 구조 ---
            int n_input;
            int n_output;
            Eigen::VectorXd n_hidden;
            int n_layer;

            // --- 단일 추론(Single Inference)용 변수 ---
            std::vector<Eigen::VectorXd> hidden;
            std::vector<Eigen::MatrixXd> hidden_derivative;
            Eigen::VectorXd input;
            Eigen::VectorXd output;
            Eigen::MatrixXd output_derivative;

            // --- NeRF 관련 ---
            bool is_nerf;
            Eigen::VectorXd input_nerf;

            // --- 배치 추론(Batch Inference)용 변수 (추가) ---
            std::vector<Eigen::MatrixXd> batch_hidden;
            Eigen::MatrixXd batch_input;
            Eigen::MatrixXd batch_input_nerf;
            Eigen::MatrixXd batch_output;

            // ▼▼▼▼▼ 여기에 시간 저장을 위한 변수를 추가합니다. ▼▼▼▼▼
            std::vector<double> inference_times_ms;

            // ▼▼▼▼▼ 여기에 이전 인덱스 저장 변수를 추가합니다. ▼▼▼▼▼
            Eigen::Index previous_min_col = -1; // 초기값 -1
            

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
        private:
            std::string file_path_;
            MLP mlp_;

            void readWeightFile(int weight_num);
            void readBiasFile(int bias_num);
            void loadNetwork();
            void initializeNetwork(int n_input, int n_output, Eigen::VectorXd n_hidden, bool is_nerf);

            // ReLU는 private 멤버로 유지
            double ReLU(double input)
            {
                return std::max(0.0, input);
            }
            double ReLU_derivative(double input)
            {
                return (input > 0)? 1.0 : 0.0;
            }
    };
}

#endif