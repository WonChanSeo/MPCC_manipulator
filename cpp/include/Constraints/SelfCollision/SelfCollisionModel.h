#ifndef MPCC_SELF_COLLISION_H
#define MPCC_SELF_COLLISION_H

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
    class SelCollNNmodel
    {
        struct MLP
        {
            ~MLP() { std::cout << "MLP terminate" << std::endl; }
            // --- 신경망 파라미터 (FP32) ---
            std::vector<Eigen::MatrixXf> weight;
            std::vector<Eigen::VectorXf> bias;
            std::vector<Eigen::VectorXf> hidden;
            std::vector<Eigen::MatrixXf> hidden_derivative;

            std::vector<std::string> w_path;
            std::vector<std::string> b_path;

            std::vector<std::ifstream> weight_files;
            std::vector<std::ifstream> bias_files;

            int n_input;
            int n_output;
            Eigen::VectorXd n_hidden;
            int n_layer;

            Eigen::VectorXf input;
            Eigen::VectorXf output;
            Eigen::MatrixXf output_derivative;

            bool is_nerf;
            Eigen::VectorXf input_nerf;

            bool loadweightfile_verbose = false;
            bool loadbiasfile_verbose = false;
        };

        public:
            SelCollNNmodel();
            SelCollNNmodel(const std::string & file_path);
            ~SelCollNNmodel();
            void setNeuralNetwork(int n_input, int n_output, Eigen::VectorXd n_hidden, bool is_nerf);
            std::pair<Eigen::VectorXd, Eigen::MatrixXd> calculateMlpOutput(Eigen::VectorXd input, bool time_verbose);
        private:
            std::string file_path_;
            MLP mlp_;

            void readWeightFile(int weight_num);
            void readBiasFile(int bias_num);
            void loadNetwork();
            void initializeNetwork(int n_input, int n_output, Eigen::VectorXd n_hidden, bool is_nerf);

            float ReLU(float input)
            {
                return std::max(0.0f, input);
            }
            float ReLU_derivative(float input)
            {
                return (input > 0.0f) ? 1.0f : 0.0f;
            }
    };
}

#endif
