"""
Validate C++ inference against Python using Neural-JSDF validation dataset
This script:
1. Loads the validation dataset used for quantization tests
2. Exports it for C++ to process
3. Runs C++ inference
4. Compares C++ output with Python output
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*80)
print("C++ vs PYTHON INFERENCE VALIDATION")
print("Using Neural-JSDF validation dataset")
print("="*80)

# ============================================================================
# 1. Load the Neural-JSDF model (same as validation scripts)
# ============================================================================

has_model = False
try:
    # Try to import from validation scripts
    sys.path.insert(0, os.path.abspath('../..'))
    from validation_fma_quant import FMAQuantValidation
    has_model = True
    print("✓ Imported validation model successfully")
except ImportError as e:
    print(f"Warning: Cannot import validation model: {e}")
    print("Will only generate test data.")

# ============================================================================
# 2. Generate validation dataset (same parameters as validation scripts)
# ============================================================================

print("\n" + "-"*80)
print("Generating validation dataset...")
print("-"*80)

device = torch.device('cpu')
params = {'device': device, 'dtype': torch.float32}

# Same range as validation scripts
q_min = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -1, -1, -0.2]).to(**params)
q_max = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1, 1, 1.3]).to(**params)
q_span = q_max - q_min

# Generate test samples
n_samples = 100  # Match validation scripts
torch.manual_seed(42)  # For reproducibility
input_data = q_min + q_span * torch.rand(n_samples, 10, **params)

print(f"Generated {n_samples} samples")
print(f"Input shape: {input_data.shape}")
print(f"Input range: [{torch.min(input_data).item():.4f}, {torch.max(input_data).item():.4f}]")

# ============================================================================
# 3. Run Python inference (if model available)
# ============================================================================

python_output = None
if has_model:
    print("\n" + "-"*80)
    print("Running Python inference...")
    print("-"*80)

    try:
        # Initialize FMA validation model (E8M7 = BF16, matching C++ default)
        validator = FMAQuantValidation(exponent_bits=8, mantissa_bits=7)

        print(f"✓ Model initialized")
        print(f"  Architecture: 10 -> [256, 256, 256, 256] -> 9")
        print(f"  NERF encoding: enabled")
        print(f"  Quantization: E8M7 (BF16, FMA-level quantization)")

        # Run inference
        python_output = validator.calc_nn_pred(input_data)

        print(f"✓ Python inference complete")
        print(f"  Output shape: {python_output.shape}")
        print(f"  Output range: [{torch.min(python_output).item():.4f}, {torch.max(python_output).item():.4f}]")

    except Exception as e:
        print(f"Error running Python inference: {e}")
        import traceback
        traceback.print_exc()
        python_output = None

# ============================================================================
# 4. Export data for C++
# ============================================================================

print("\n" + "-"*80)
print("Exporting data for C++...")
print("-"*80)

# Save input data
cpp_input_file = 'cpp_validation_input.txt'
np.savetxt(cpp_input_file, input_data.numpy().T, fmt='%.10f')  # Transpose: C++ expects column-major
print(f"✓ Saved input to: {cpp_input_file}")
print(f"  Format: {input_data.shape[1]} rows × {input_data.shape[0]} columns (column-major)")

# Save Python output (if available)
if python_output is not None:
    python_output_file = 'python_validation_output.txt'
    np.savetxt(python_output_file, python_output.numpy().T, fmt='%.10f')
    print(f"✓ Saved Python output to: {python_output_file}")
    print(f"  Format: {python_output.shape[1]} rows × {python_output.shape[0]} columns (column-major)")

    # Also save as .npz for easier loading
    np.savez('python_validation_reference.npz',
             input=input_data.numpy(),
             output=python_output.numpy())
    print(f"✓ Saved reference data to: python_validation_reference.npz")

# ============================================================================
# 5. Generate C++ test program
# ============================================================================

print("\n" + "-"*80)
print("Generating C++ test program...")
print("-"*80)

cpp_test_code = f"""
// Auto-generated C++ test program for validation
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include "Constraints/EnvCollision/EnvCollisionModel.h"

int main() {{
    std::cout << "======================================================================" << std::endl;
    std::cout << "C++ INFERENCE VALIDATION TEST" << std::endl;
    std::cout << "======================================================================" << std::endl;

    // Load input data
    std::ifstream input_file("cpp_validation_input.txt");
    if (!input_file.is_open()) {{
        std::cerr << "Failed to open input file!" << std::endl;
        return 1;
    }}

    const int n_input = 10;
    const int n_samples = {n_samples};
    Eigen::MatrixXd inputs(n_input, n_samples);

    for (int i = 0; i < n_input; ++i) {{
        for (int j = 0; j < n_samples; ++j) {{
            input_file >> inputs(i, j);
        }}
    }}
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
    std::cout << "\\nRunning batch inference..." << std::endl;
    auto result = nn_model.calculateMlpOutputBatch(inputs, false);

    const Eigen::VectorXd& outputs = result.first;
    const Eigen::MatrixXd& jacobian = result.second;

    std::cout << "✓ Inference complete" << std::endl;
    std::cout << "  Output shape: " << outputs.size() << " × 1 (minimum per row)" << std::endl;

    // Save outputs
    nn_model.saveInputOutputLog("cpp_validation_log.txt", inputs,
                                 Eigen::MatrixXd::Zero(9, n_samples));  // TODO: save full batch output

    // Save results for Python comparison
    std::ofstream output_file("cpp_validation_output.txt");
    output_file << std::fixed << std::setprecision(10);
    for (int i = 0; i < outputs.size(); ++i) {{
        output_file << outputs(i) << std::endl;
    }}
    output_file.close();

    std::cout << "✓ Saved output to: cpp_validation_output.txt" << std::endl;

    // Print statistics
    std::cout << "\\nOutput statistics:" << std::endl;
    std::cout << "  Min: " << outputs.minCoeff() << std::endl;
    std::cout << "  Max: " << outputs.maxCoeff() << std::endl;
    std::cout << "  Mean: " << outputs.mean() << std::endl;

    std::cout << "\\n======================================================================" << std::endl;
    std::cout << "Test completed successfully!" << std::endl;
    std::cout << "Run compare_cpp_python.py to compare results" << std::endl;
    std::cout << "======================================================================" << std::endl;

    return 0;
}}
"""

cpp_test_file = '../../../cpp/test_validation_inference.cpp'
with open(cpp_test_file, 'w') as f:
    f.write(cpp_test_code)

print(f"✓ Generated C++ test program: {cpp_test_file}")

# ============================================================================
# 6. Instructions for user
# ============================================================================

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("1. Compile the C++ test program:")
print("   cd ../../../cpp/build")
print("   make")
print()
print("2. Run the C++ test:")
print("   cd ../../Neural-JSDF/learning/nn_validation")
print("   ../../../cpp/build/test_validation_inference")
print()
print("3. Compare results:")
print("   python3 compare_cpp_python.py")
print("="*80)
