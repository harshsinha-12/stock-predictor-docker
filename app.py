from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import datetime
import yfinance as yf

app = Flask(__name__)

model_path = 'stock_model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    scaler = pickle.load(open('scaler.pkl', 'rb'))
else:
    model = None
    scaler = None

def fetch_stock_data(ticker, period='1y'):
    """Fetch historical stock data using yfinance"""
    try:
        end_date = datetime.datetime.now()
        if period == '1y':
            start_date = (end_date - datetime.timedelta(days=365))
        elif period == '3mo':
            start_date = (end_date - datetime.timedelta(days=90))
        elif period == '5y':
            start_date = (end_date - datetime.timedelta(days=365*5))
        else:
            start_date = (end_date - datetime.timedelta(days=365))
        data = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
        if data.empty:
            print(f"Failed to fetch data for {ticker}. The ticker might be incorrect or delisted.")
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def create_features(data):
    """Create features from stock data for prediction"""
    df = data.copy()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Return'] = df['Close'].pct_change()
    df['MA5_Return'] = df['MA5'].pct_change()
    df['MA20_Return'] = df['MA20'].pct_change()
    df = df.dropna()
    return df

def predict_next_day(ticker):
    """Predict the next day's stock price movement"""
    if model is None:
        return {"error": "Model not trained. Please train the model first."}
    stock_data = fetch_stock_data(ticker, period='3mo')
    if stock_data is None or stock_data.empty:
        return {"error": f"Unable to fetch data for {ticker}"}
    features_df = create_features(stock_data)
    if features_df.empty:
        return {"error": "Not enough data to create features"}
    latest_features = features_df.iloc[-1][['MA5', 'MA20', 'Return', 'MA5_Return', 'MA20_Return']].values.reshape(1, -1)
    scaled_features = scaler.transform(latest_features)
    prediction = model.predict(scaled_features)[0]
    last_close = stock_data['Close'].iloc[-1]
    if prediction == 1:
        movement = "UP"
        confidence = float(model.predict_proba(scaled_features)[0][1])
    else:
        movement = "DOWN"
        confidence = float(model.predict_proba(scaled_features)[0][0])
    return {
        "ticker": ticker,
        "last_close": float(last_close),
        "predicted_movement": movement,
        "confidence": confidence,
        "last_update": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ticker = data.get('ticker', 'AAPL')
    
    result = predict_next_day(ticker)
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health_check():
    return 'OK', 200

@app.route('/train', methods=['POST'])
def train_model():
    """Train the model using historical data"""
    data = request.json
    ticker = data.get('ticker', 'AAPL')
    stock_data = fetch_stock_data(ticker, period='5y')
    if stock_data is None or stock_data.empty:
        return jsonify({"error": f"Unable to fetch data for {ticker}"})
    features_df = create_features(stock_data)
    if features_df.empty:
        return jsonify({"error": "Not enough data to create features"})
    features_df['Target'] = (features_df['Close'].shift(-1) > features_df['Close']).astype(int)
    features_df = features_df.dropna()
    X = features_df[['MA5', 'MA20', 'Return', 'MA5_Return', 'MA20_Return']].values
    y = features_df['Target'].values
    global scaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    from sklearn.ensemble import RandomForestClassifier
    global model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return jsonify({
        "message": "Model trained successfully",
        "ticker": ticker,
        "data_points": len(features_df)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)