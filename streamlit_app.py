import streamlit as st
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
import pickle
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64

# Set page config
st.set_page_config(
    page_title="Stock Price Movement Predictor",
    page_icon="📈",
    layout="wide"
)

# Define global variables at the top level
model = None
scaler = None
model_trained = False
model_path = 'stock_model.pkl'
scaler_path = 'scaler.pkl'

# Apply custom CSS
def load_css():
    """Load and apply custom CSS styling"""
    if os.path.exists('static/style.css'):
        with open('static/style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning("Custom styling file not found. Using default styling.")

# Check if model exists
if os.path.exists(model_path) and os.path.exists(scaler_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    model_trained = True

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
            st.error(f"Failed to fetch data for {ticker}. The ticker might be incorrect or delisted.")
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
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
    global model, scaler
    
    if model is None:
        st.error("Model not trained. Please train the model first.")
        return None
    
    stock_data = fetch_stock_data(ticker, period='3mo')
    if stock_data is None or stock_data.empty:
        st.error(f"Unable to fetch data for {ticker}")
        return None
    
    features_df = create_features(stock_data)
    if features_df.empty:
        st.error("Not enough data to create features")
        return None
    
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

def train_model(ticker, n_estimators=100, max_depth=None, min_samples_split=2):
    """Train the model using historical data"""
    global model, scaler, model_trained
    
    stock_data = fetch_stock_data(ticker, period='5y')
    if stock_data is None or stock_data.empty:
        st.error(f"Unable to fetch data for {ticker}")
        return None
    
    features_df = create_features(stock_data)
    if features_df.empty:
        st.error("Not enough data to create features")
        return None
    
    features_df['Target'] = (features_df['Close'].shift(-1) > features_df['Close']).astype(int)
    features_df = features_df.dropna()
    
    X = features_df[['MA5', 'MA20', 'Return', 'MA5_Return', 'MA20_Return']].values
    y = features_df['Target'].values
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Use the parameters passed from the UI
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X_scaled, y)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Use 5-fold cross validation to get model performance metrics
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    
    # Set model_trained flag
    model_trained = True
    
    return {
        "message": "Model trained successfully",
        "ticker": ticker,
        "data_points": len(features_df),
        "cv_accuracy": np.mean(cv_scores),
        "model_params": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split
        }
    }

def plot_stock_data(data, ticker):
    """Plot stock data with moving averages using Plotly"""
    # Create subplots: price and volume
    fig = make_subplots(
        rows=2, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{ticker} Stock Price', 'Trading Volume'),
        row_width=[0.7, 0.3]
    )
    
    # Add price traces
    fig.add_trace(
        go.Scatter(
            x=data['Date'],
            y=data['Close'],
            name='Close Price',
            line=dict(color='royalblue')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data['Date'],
            y=data['MA5'],
            name='5-Day MA',
            line=dict(color='orange')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data['Date'],
            y=data['MA20'],
            name='20-Day MA',
            line=dict(color='green')
        ),
        row=1, col=1
    )
    
    # Add volume trace
    fig.add_trace(
        go.Bar(
            x=data['Date'],
            y=data['Volume'],
            name='Volume',
            marker=dict(color='rgba(0, 0, 200, 0.3)')
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )
    
    fig.update_yaxes(title_text="Price: ", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

# Apply the CSS
load_css()

# Header with custom styling
st.markdown("<h1 class='main-header'>Stock Price Movement Predictor</h1>", unsafe_allow_html=True)
st.markdown("### Predict tomorrow's stock movement with machine learning")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/stock-market.png", width=80)
    st.title('Navigation')
    app_mode = st.radio('Choose Mode', ['Predict Stock Movement', 'Train Model'])
    
    st.markdown("---")
    st.markdown("### About")
    st.info("This application predicts stock price movements using historical data and machine learning.")
    
    # Model status
    st.markdown("### Model Status")
    if model_trained:
        st.success("Model is trained and ready for predictions")
    else:
        st.warning("Model not trained. Please train the model first.")

# Main content based on selected mode
if app_mode == 'Predict Stock Movement':
    # Create a container with a card-like appearance
    main_container = st.container()
    with main_container:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # User input in a form
            with st.form("stock_input_form"):
                ticker = st.text_input('Enter Stock Ticker (e.g., AAPL, MSFT, GOOG)', 'AAPL')
                period = st.select_slider('Select Period for Analysis', options=['3mo', '1y', '5y'], value='1y')
                
                col1_form, col2_form = st.columns(2)
                with col1_form:
                    analyze_button = st.form_submit_button('Analyze Stock')
                with col2_form:
                    predict_button = st.form_submit_button('Predict Movement')
        
        with col2:
            st.markdown("### Popular Tickers")
            popular_tickers = {
                "Technology": ["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
                "Finance": ["JPM", "BAC", "GS", "V", "MA"],
                "Energy": ["XOM", "CVX", "BP", "COP", "SLB"]
            }
            
            sector = st.selectbox("Sector", list(popular_tickers.keys()))
            ticker_buttons = st.columns(5)
            
            for i, tick in enumerate(popular_tickers[sector]):
                with ticker_buttons[i % 5]:
                    if st.button(tick, key=f"tick_{tick}"):
                        ticker = tick
                        st.experimental_rerun()
    
    # Analysis section
    analysis_container = st.container()
    with analysis_container:
        if analyze_button or 'ticker' in locals():
            # Fetch and display stock data
            with st.spinner('Fetching stock data...'):
                ticker_to_use = ticker if 'ticker' in locals() else 'AAPL'
                stock_data = fetch_stock_data(ticker_to_use, period)
                if stock_data is not None:
                    st.success(f'Successfully fetched data for {ticker_to_use}')
                    
                    # Create features and plot
                    features_df = create_features(stock_data)
                    
                    # Display the interactive plot
                    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                    fig = plot_stock_data(features_df, ticker_to_use)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Display recent data in an expandable section
                    with st.expander("View Recent Data"):
                        st.dataframe(stock_data.tail(), use_container_width=True)
                        
                        # Add summary statistics
                        st.markdown("### Summary Statistics")
                        stats_cols = st.columns(4)
                        with stats_cols[0]:
                            st.metric("Current Price:", f"{stock_data['Close'].iloc[-1]:.2f}", 
                                     f"{stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]:.2f}")
                        with stats_cols[1]:
                            st.metric("52w High: ", f"{stock_data['High'].max():.2f}")
                        with stats_cols[2]:
                            st.metric("52w Low: ", f"{stock_data['Low'].min():.2f}")
                        with stats_cols[3]:
                            st.metric("Volume: ", f"{stock_data['Volume'].iloc[-1]:,.0f}")

    # Prediction section
    prediction_container = st.container()
    with prediction_container:
        if predict_button:
            if not model_trained:
                st.error('Model not trained. Please go to Train Model tab and train the model first.')
            else:
                with st.spinner('Predicting...'):
                    result = predict_next_day(ticker)
                    if result:
                        # Create a nice card for prediction result
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.subheader('Prediction Result')
                        
                        # Display prediction with nice formatting
                        movement_class = "prediction-up" if result['predicted_movement'] == 'UP' else "prediction-down"
                        confidence_pct = f"{result['confidence']*100:.2f}%"
                        
                        col1_pred, col2_pred = st.columns([1, 1])
                        
                        with col1_pred:
                            st.markdown(f"""
                            ### Ticker: {result['ticker']}
                            - Last Close Price: **{result['last_close']:.2f}**
                            - Predicted Movement: <span class='{movement_class}'>{result['predicted_movement']}</span>
                            - Confidence: **{confidence_pct}**
                            - Last Updated: {result['last_update']}
                            """, unsafe_allow_html=True)
                        
                        with col2_pred:
                            # Gauge chart for confidence
                            st.subheader('Prediction Confidence')
                            st.progress(result['confidence'])
                            
                            # Add emojis for better visual cues
                            if result['predicted_movement'] == 'UP':
                                st.markdown("### 📈 Bullish Signal")
                            else:
                                st.markdown("### 📉 Bearish Signal")
                        
                        st.markdown("</div>", unsafe_allow_html=True)

elif app_mode == 'Train Model':
    st.markdown("<h2 class='main-header'>Train Stock Prediction Model</h2>", unsafe_allow_html=True)
    
    # Create two columns layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Model Training Process
        
        This section allows you to train a new machine learning model to predict stock price movements.
        
        The training process:
        1. Fetches 5 years of historical data for the selected ticker
        2. Creates technical indicators as features (moving averages, returns)
        3. Labels data points as UP (1) or DOWN (0) based on next day's movement
        4. Trains a Random Forest classifier model on the data
        5. Saves the model for future predictions
        """)
        
        # Model parameters in expander
        with st.expander("Advanced Model Configuration"):
            n_estimators = st.slider("Number of Trees in Forest", 50, 500, 100, 10)
            max_depth = st.slider("Maximum Depth of Trees", 2, 30, 10, 1)
            min_samples_split = st.slider("Minimum Samples to Split", 2, 20, 2, 1)
        
        # Input form
        with st.form("train_model_form"):
            ticker = st.text_input('Enter Stock Ticker for Training', 'AAPL')
            train_button = st.form_submit_button('Train New Model')
        
        if train_button:
            # Progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate stages of training with progress updates
            status_text.text("Fetching historical data...")
            progress_bar.progress(10)
            
            # Start training
            with st.spinner('Training model... This may take a moment.'):
                # Override the default model parameters
                if 'n_estimators' in locals():
                    # We'll modify the train_model function slightly to use these parameters
                    result = train_model(ticker, n_estimators, max_depth, min_samples_split)
                else:
                    result = train_model(ticker)
                
                if result:
                    # Update progress through the process
                    status_text.text("Processing features...")
                    progress_bar.progress(40)
                    status_text.text("Training model...")
                    progress_bar.progress(70)
                    status_text.text("Saving model...")
                    progress_bar.progress(90)
                    status_text.text("Complete!")
                    progress_bar.progress(100)
                    
                    # Success message
                    st.success(f"Model trained successfully on {result['ticker']} with {result['data_points']} data points")
                    st.balloons()
    
    with col2:
        st.markdown("### Model Information")
        
        # Display model status
        if model_trained:
            st.success("Model Status: Trained")
            # Display feature importance if model exists
            if model is not None:
                st.markdown("#### Feature Importance")
                feature_names = ['MA5', 'MA20', 'Return', 'MA5_Return', 'MA20_Return']
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Create a horizontal bar chart for feature importance
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h',
                    marker=dict(color='royalblue')
                ))
                fig.update_layout(
                    title='Feature Importance',
                    xaxis_title='Importance',
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Model Status: Not Trained")
            st.info("Train a model to see feature importance and metrics.")

# Footer
st.sidebar.markdown('---')
st.sidebar.markdown("""
### Technologies Used
- 🐍 Python
- 📊 Streamlit
- 🧠 scikit-learn
- 📈 yfinance
- 📊 Plotly & Matplotlib
""")
st.sidebar.info('Stock Price Movement Predictor © 2025')

# Add a disclaimer at the bottom
st.markdown('---')
st.caption("""
**Disclaimer**: This application is for educational purposes only. 
Stock market investments involve risk, and past performance does not guarantee future results.
No financial decisions should be made based solely on the predictions from this application.
""")
