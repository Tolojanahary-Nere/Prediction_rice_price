#!/bin/bash
# Script to run the Rice Price Prediction App

# Check if venv exists, if not create it
if [ ! -d "venv_app" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv_app
fi

# Activate venv
source venv_app/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the app
echo "Starting Streamlit app..."
streamlit run app.py
