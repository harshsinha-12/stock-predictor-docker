# Stock Price Prediction App

This is a machine learning-based application that predicts stock price movements for the next trading day. It uses historical stock data to train a Random Forest classifier model and provides predictions for whether a stock's price will go up or down. The application is available in two implementations:

1. **Flask API**: Backend service with REST endpoints for model training and prediction
2. **Streamlit UI**: Interactive web application with data visualization

## Features

- Predict next day price movement for any publicly traded stock
- Train custom models for specific stocks
- Interactive data visualization (Streamlit version)
- Simple, user-friendly web interface 
- Containerized with Docker for easy deployment

## Prerequisites

- Docker and Docker Compose installed on your system (for containerized deployment)
- Python 3.9+ (for local development)
- Internet connection (for fetching stock data)

## Quick Start

### Live Demo

Try the Streamlit application online: [Stock Predictor App](https://stock-predictor-docker.streamlit.app/)

### Local Setup

1. Clone or download this repository
2. Navigate to the project directory
3. Choose your preferred method to run the application:

### Using Docker Compose (Recommended)

```bash
# Run both Flask and Streamlit applications
docker-compose up --build
```

- Flask API will be available at: `http://localhost:5001`
- Streamlit UI will be available at: `http://localhost:8501`

### Running Locally

#### Flask Application
```bash
# Install requirements
pip install -r requirements.txt

# Run the Flask app
python app.py
```

#### Streamlit Application
```bash
# Run the script (installs dependencies and starts app)
./run_streamlit.sh

# Or manually:
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Using the Application

### Flask UI
1. Open your web browser and navigate to `http://localhost:5001`
2. Enter a stock ticker symbol (e.g., AAPL for Apple) and click "Predict"

### Streamlit UI
1. Open your web browser and navigate to `http://localhost:8501` (or visit the [live demo](https://stock-predictor-docker.streamlit.app/))
2. Choose between "Predict Stock Movement" and "Train Model" modes
3. Enter a stock ticker and use the interactive controls

## Training a Model

The application needs to train a model before it can make predictions.

1. Enter a stock ticker symbol in the input field
2. Click the "Train New Model" button
3. Wait for the model to be trained (this may take a few seconds)
4. Once trained, you can predict the next day's movement

## Tech Stack

- Flask & Streamlit: Web application frameworks
- scikit-learn: Machine learning library for the prediction model
- yfinance: Yahoo Finance API wrapper for fetching stock data
- NumPy & Pandas: Data manipulation and analysis
- Matplotlib & Plotly: Data visualization
- Docker: Containerization

## Docker Commands

### Run using pre-built images:

#### Flask Application
```bash
docker run -p 5001:5001 harshsinha12/stock-predictor-flask
```

## Disclaimer

This application is for educational purposes only and should not be used for financial decisions. Stock market investments involve risk, and past performance does not guarantee future results.
