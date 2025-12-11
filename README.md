# 🌾 Mada Rice Analytics - Prediction & Analysis

## Overview
A comprehensive data analytics and forecasting platform for rice prices in Madagascar. This project demonstrates a hybrid architecture combining a **FastAPI backend** for data management and a **Streamlit frontend** for interactive visualization and AI-powered forecasting.

## ✨ Key Features

### 📊 Interactive Dashboard
- **Market Overview**: Real-time tracking of rice prices across major regions (Antananarivo, Toamasina, etc.).
- **Regional Analysis**: Comparative charts and heatmaps.
- **Filtering**: Drill down by Rice Type (Vary Gasy, Makalioka, etc.) and Region.

### 🤖 AI Forecasting
- **Machine Learning**: Uses **Random Forest Regressor** to predict future prices.
- **Feature Engineering**: Incorporates seasonality, inflation trends, and simulated external factors (Rainfall, Fuel Price).
- **Scenario Simulation**: "What-if" analysis to estimate the impact of economic shocks.

### 🏗️ Technical Architecture
- **Backend**: FastAPI + MongoDB (Code available in `backend/`).
- **Frontend**: Streamlit + Plotly.
- **Data**: Realistic synthetic data generation engine reflecting Malagasy market dynamics.

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Tolojanahary-Nere/Prediction_rice_price.git
   cd Prediction_rice_price
   ```

2. **Run the Application**
   ```bash
   chmod +x run.sh
   ./run.sh
   ```

## 📂 Project Structure
```
Prediction_rice_price/
├── app.py               # Streamlit Dashboard Entry Point
├── src/
│   ├── data_generator.py # Synthetic Data Engine
│   └── model.py         # ML Forecasting Logic
├── backend/             # Original FastAPI Backend
│   └── fastapi_app/     # API Code
├── data/                # Data storage
├── requirements.txt     # Python Dependencies
└── run.sh               # Startup Script
```

## 👤 Author
**TOLOJANAHARY Josia Marie Néré**
*Full Stack Developer & Data Scientist*
