
// Auto-generated C++ test program for validation
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include "Constraints/EnvCollision/EnvCollisionModel.h"

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "C++ INFERENCE VALIDATION TEST" << std::endl;
    std::cout << "======================================================================" << std::endl;

    // Load input data
    std::ifstream input_file("cpp_validation_input.txt");
    if (!input_file.is_open()) {
        std::cerr << "Failed to open input file!" << std::endl;
        return 1;
    }

    const int n_input = 10;
    const int n_samples = 100;
    Eigen::MatrixXd inputs(n_input, n_samples);

    for (int i = 0; i < n_input; ++i) {
        for (int j = 0; j < n_samples; ++j) {
            input_file >> inputs(i, j);
        }
    }
    input_file.close();

    std::cout << "✓ Loaded " << n_samples << " input samples" << std::endl;
    std::cout << "  Input shape: " << inputs.rows() << " × " << inputs.cols() << std::endl;

    // Initialize Neural Network model
    mpcc::EnvCollNNmodel nn_model;

    // Set network architecture (match Python model)
    Eigen::VectorXd n_hidden(4);
    n_hidden << 256, 256, 256, 256;
    nn_model.setNeuralNetwork(10, 9, n_hidden, true);  // is_nerf = true

    std::cout << "✓ Neural network initialized" << std::endl;
    std::cout << "  Architecture: 10 -> [256, 256, 256, 256] -> 9" << std::endl;
    std::cout << "  NERF encoding: enabled" << std::endl;

    // Run inference
    std::cout << "\nRunning batch inference..." << std::endl;
    auto result = nn_model.calculateMlpOutputBatch(inputs, false);

    const Eigen::VectorXd& min_outputs = result.first;
    const Eigen::MatrixXd& jacobian = result.second;

    // Get full batch output (9 x n_samples)
    Eigen::MatrixXd batch_output = nn_model.getBatchOutput();

    std::cout << "✓ Inference complete" << std::endl;
    std::cout << "  Full output shape: " << batch_output.rows() << " × " << batch_output.cols() << std::endl;

    // Save outputs
    nn_model.saveInputOutputLog("cpp_validation_log.txt", inputs, batch_output);

    // Save results for Python comparison (n_samples x 9 format)
    std::ofstream output_file("cpp_validation_output.txt");
    output_file << std::fixed << std::setprecision(10);
    for (int j = 0; j < batch_output.cols(); ++j) {  // For each sample
        for (int i = 0; i < batch_output.rows(); ++i) {  // For each output
            output_file << batch_output(i, j);
            if (i < batch_output.rows() - 1) output_file << " ";
        }
        output_file << std::endl;
    }
    output_file.close();

    std::cout << "✓ Saved output to: cpp_validation_output.txt" << std::endl;

    // Print statistics
    std::cout << "\nOutput statistics:" << std::endl;
    std::cout << "  Min: " << batch_output.minCoeff() << std::endl;
    std::cout << "  Max: " << batch_output.maxCoeff() << std::endl;
    std::cout << "  Mean: " << batch_output.mean() << std::endl;

    std::cout << "\n======================================================================" << std::endl;
    std::cout << "Test completed successfully!" << std::endl;
    std::cout << "Run compare_cpp_python.py to compare results" << std::endl;
    std::cout << "======================================================================" << std::endl;

    return 0;
}
