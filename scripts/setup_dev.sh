#!/bin/bash
set -e

echo "=========================================="
echo "🚀 Bootstrapping APEX Quant Environment"
echo "=========================================="

echo "0. Validating Environment Requirements..."
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [ "$PYTHON_VERSION" != "3.10" ]; then
    echo "⚠️  WARNING: Python 3.10 is recommended for APEX. You are using $PYTHON_VERSION."
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "⚠️  WARNING: Docker is not installed. Local integration tests will fail."
else
    if ! docker info >/dev/null 2>&1; then
        echo "⚠️  WARNING: Docker daemon is not running."
    fi
fi

if [ "$(uname)" = "Darwin" ]; then
    if command -v brew >/dev/null 2>&1; then
        if ! brew --prefix libomp >/dev/null 2>&1; then
            echo "⚠️  WARNING: libomp is not installed. Run 'brew install libomp' for real LightGBM training."
        fi
    else
        echo "⚠️  WARNING: Homebrew is not installed. Install libomp manually for real LightGBM training."
    fi
fi

echo "1. Creating Python Virtual Environment..."
python3 -m venv venv
source venv/bin/activate

echo "2. Upgrading Pip & Installing Dependencies..."
pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt

echo "3. Installing Pre-Commit Hooks..."
pre-commit install

echo "4. Initializing Local Testing Architecture..."
if [ ! -d "htmlcov" ]; then
    mkdir -p htmlcov
fi

echo "5. Validating Makefile & Pytest Installation..."
if command -v make >/dev/null 2>&1; then
    echo "✅ Makefile runtime found."
else
    echo "❌ ERROR: 'make' is not installed on your system."
    exit 1
fi

if pytest --version >/dev/null 2>&1; then
    echo "✅ Pytest successfully installed."
else
    echo "❌ ERROR: Pytest installation failed."
    exit 1
fi

echo "=========================================="
echo "🎯 Setup Complete!"
echo "Run 'source venv/bin/activate' to enter the environment."
echo "Run 'make ci-local' to validate the repository."
echo "=========================================="
