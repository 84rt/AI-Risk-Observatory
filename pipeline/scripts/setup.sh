#!/bin/bash
# Setup script for AIRO pipeline

set -e

echo "=========================================="
echo "AIRO Pipeline Setup"
echo "=========================================="
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
python3 --version || { echo "Error: Python 3 not found. Please install Python 3.8+"; exit 1; }

# Create virtual environment
echo "[2/5] Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"

# Activate virtual environment
echo "[3/5] Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo "[4/5] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Download NLTK data
echo "[5/5] Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True)"
echo "✓ NLTK data downloaded"

# Create directories
mkdir -p data output/pdfs logs

# Check for .env file
if [ ! -f .env ] && [ ! -f .env.local ]; then
    echo ""
    echo "⚠️  No .env or .env.local file found!"
    echo "   Creating .env from template..."
    cp .env.template .env
    echo "   Please edit .env and add your API keys:"
    echo "   - GEMINI_API_KEY"
    echo "   - COMPANIES_HOUSE_API_KEY"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Edit .env and add your API keys"
echo ""
echo "3. Edit data/companies_template.csv with your companies"
echo ""
echo "4. Run the pipeline:"
echo "   python run_pipeline.py --companies data/companies_template.csv"
echo ""
