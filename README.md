# 📈 Cryptocurrency Price Prediction

This project utilizes historical cryptocurrency data to predict future price movements using machine learning. It fetches data from the CoinGecko API, calculates trading metrics, trains a Random Forest regression model, and makes predictions on price changes.

## 🚀 Features

- **Data Fetching**: Retrieve historical cryptocurrency price data using the CoinGecko API.
- **Metrics Calculation**: Calculate key trading metrics to analyze price movements.
- **Machine Learning Model**: Train a Random Forest regressor to predict future high and low price differences.
- **Prediction**: Make predictions based on trained models.

## 📊 Technologies Used

- **Python**: Main programming language.
- **Libraries**:
  - `requests`: For making API calls.
  - `pandas`: For data manipulation and analysis.
  - `scikit-learn`: For building and evaluating machine learning models.
  - `joblib`: For saving and loading models.

## 📦 Installation

1. Install required packages:

```bash
pip install -r requirements.txt
```
2. Add your CoinGecko API key in the script where indicated.

## 📋 Usage
Run the main script to fetch data, calculate metrics, train the model, and make predictions:

```bash
python main.py
```
Replace the input_features in the script with actual values to test predictions.

## 📊 Output
The script saves:

Raw cryptocurrency data and calculated metrics to an Excel file named ```crypto_data.xlsx.```
