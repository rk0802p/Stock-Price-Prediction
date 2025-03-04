---

# Stock Price Prediction with LSTM

A Streamlit app that fetches stock/crypto data from Alpha Vantage, trains an LSTM model, and predicts future prices.

---

## **Features**
- Real-time and historical data for stocks (e.g., TSLA) and crypto (e.g., BTC).
- LSTM-based price predictions.
- Interactive UI with plots and forecasts.

---

## **Technologies**
- Python 3.7+
- Streamlit, Pandas, Numpy, Scikit-learn, TensorFlow, Matplotlib, Alpha Vantage API

---

## **Setup**
1. Get an Alpha Vantage API key [here](https://www.alphavantage.co/).
2. Replace `"API"` in `app.py` with your key.
3. Install dependencies: `pip install streamlit pandas numpy scikit-learn tensorflow matplotlib alpha-vantage`
4. Run: `streamlit run app.py`

---

## **Usage**
- Enter a ticker (e.g., `AAPL`, `BTC`).
- Set historical days (100–1000) and prediction days (1–10).
- Click **Predict** to see current price, historical fit, and future forecasts.

---

## **Limitations**
- Real-time data is approximated.
- Free API has rate limits (5 requests/min).

---

## **How It Works**
Fetches data → Preprocesses → Trains LSTM → Predicts → Visualizes.

---