#!/bin/bash
# Setup script for CarRacing-v3 RL Training
# This script sets up the Python environment and installs all dependencies

echo "=========================================="
echo "CarRacing-v3 RL Training Setup"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Check for SWIG (needed for Box2D)
if ! command -v swig &> /dev/null; then
    echo "⚠️  SWIG not found. Box2D requires SWIG."
    echo "   Install with: brew install swig (macOS) or sudo apt-get install swig (Linux)"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start training, run:"
echo "  python car_racing_trainer.py"
echo ""
echo "=========================================="
