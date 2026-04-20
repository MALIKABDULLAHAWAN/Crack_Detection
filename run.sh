#!/bin/bash
# ============================================================
# Crack Detection System - Start Script
# Just run this file to start your web app!
# ============================================================

echo ""
echo "============================================"
echo "   Crack Detection System - FYP Project"
echo "============================================"
echo ""

# Check Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found. Please install Python first."
    exit 1
fi

# Install required packages if not already installed
echo "Checking required packages..."
pip3 install flask torch torchvision pillow --quiet --break-system-packages 2>/dev/null || \
pip3 install flask torch torchvision pillow --quiet 2>/dev/null

echo ""
echo "Starting the web app..."
echo ""
echo ">>> Open your browser and go to:  http://127.0.0.1:5000"
echo ">>> To stop the app, press Ctrl+C"
echo ""

# Start Flask
python3 app.py
