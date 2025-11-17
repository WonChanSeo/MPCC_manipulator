#!/bin/bash

# Neural-JSDF Mantissa Precision Comparison Test Runner
# This script runs the precision validation and generates visualizations

echo "========================================================================"
echo "Neural-JSDF Mantissa Precision Comparison"
echo "Testing precision levels: 23 (fp32), 16, 12, 10, 8, 7 (bf16), 6, 5, 4, 3, 2 bit mantissa"
echo "========================================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if Python is available
if command -v python &> /dev/null; then
    PYTHON_CMD=python
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo "Error: Python not found"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version
echo ""

# Check if required modules are available
echo "Checking dependencies..."
$PYTHON_CMD -c "import torch; import numpy; import scipy; import matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Required Python modules not found"
    echo "Please install: torch, numpy, scipy, matplotlib"
    exit 1
fi
echo "✓ All dependencies available"
echo ""

# Step 1: Run precision validation
echo "========================================================================"
echo "Step 1: Running precision validation tests..."
echo "This will test 11 different mantissa bit configurations (5000 samples each)"
echo "Estimated time: 10-15 minutes"
echo "========================================================================"
echo ""

$PYTHON_CMD validation_precision.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Validation test failed"
    exit 1
fi

echo ""
echo "✓ Validation tests completed successfully"
echo ""

# Step 2: Generate visualizations
echo "========================================================================"
echo "Step 2: Generating visualizations..."
echo "========================================================================"
echo ""

$PYTHON_CMD visualize_precision_comparison.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Visualization generation failed"
    exit 1
fi

echo ""
echo "✓ Visualizations generated successfully"
echo ""

# Show generated files
echo "========================================================================"
echo "GENERATED FILES"
echo "========================================================================"
echo ""
echo "Data files:"
for bits in 23 16 12 10 8 7 6 5 4 3 2; do
    if [ -f "data_mantissa_${bits}bit.npz" ]; then
        size=$(du -h "data_mantissa_${bits}bit.npz" | cut -f1)
        echo "  ✓ data_mantissa_${bits}bit.npz ($size)"
    fi
done

if [ -f "precision_comparison_summary.npz" ]; then
    size=$(du -h "precision_comparison_summary.npz" | cut -f1)
    echo "  ✓ precision_comparison_summary.npz ($size)"
fi

echo ""
echo "Visualization files:"
if [ -f "precision_comparison_analysis.png" ]; then
    size=$(du -h "precision_comparison_analysis.png" | cut -f1)
    echo "  ✓ precision_comparison_analysis.png ($size)"
fi

if [ -f "precision_comparison_summary.png" ]; then
    size=$(du -h "precision_comparison_summary.png" | cut -f1)
    echo "  ✓ precision_comparison_summary.png ($size)"
fi

echo ""
echo "========================================================================"
echo "TEST COMPLETED SUCCESSFULLY!"
echo "========================================================================"
echo ""
echo "View the results:"
echo "  • Open precision_comparison_analysis.png for detailed analysis"
echo "  • Open precision_comparison_summary.png for summary metrics"
echo ""
echo "To view images from terminal:"
echo "  eog precision_comparison_analysis.png &"
echo "  eog precision_comparison_summary.png &"
echo ""
echo "========================================================================"
