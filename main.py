import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Step 1: Fetch Historical Data from API
def fetch_crypto_data(crypto_pair, api_key):
    base_url = "https://api.coingecko.com/api/v3/coins/{}/market_chart"
    crypto_id = crypto_pair.split('/')[0].lower()
    url = base_url.format(crypto_id)
    params = {
        'vs_currency': crypto_pair.split('/')[1].lower(),
        'days': '365',  # Limit to the last 365 days
        'interval': 'daily'
    }
    headers = {
        'x_cg_demo_api_key': api_key  # Include your API key here
    }
    
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    
    # Check if 'prices' is in the response
    if 'prices' not in data:
        print("Error: 'prices' key not found in the response.")
        print("Full response:", data)
        return pd.DataFrame()  # Return an empty DataFrame or handle accordingly
    
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['Date', 'Close'])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df['Open'] = df['Close'].shift(1)
    df['High'] = df['Close'].rolling(window=1).max()
    df['Low'] = df['Close'].rolling(window=1).min()
    df = df.dropna()
    
    return df

# Step 2: Calculate Trading Metrics
def calculate_metrics(data, variable1, variable2):
    # Ensure the data is sorted by date
    data = data.sort_values(by='Date')
    
    # Calculate rolling metrics for High and Low
    data[f'High_Last_{variable1}_Days'] = data['High'].rolling(window=variable1).max()
    data[f'Days_Since_High_Last_{variable1}_Days'] = (data['Date'] - data['Date'].shift(variable1)).dt.days
    data[f'%_Diff_From_High_Last_{variable1}_Days'] = (data['Close'] - data[f'High_Last_{variable1}_Days']) / data[f'High_Last_{variable1}_Days'] * 100
    
    data[f'Low_Last_{variable1}_Days'] = data['Low'].rolling(window=variable1).min()
    data[f'Days_Since_Low_Last_{variable1}_Days'] = (data['Date'] - data['Date'].shift(variable1)).dt.days
    data[f'%_Diff_From_Low_Last_{variable1}_Days'] = (data['Close'] - data[f'Low_Last_{variable1}_Days']) / data[f'Low_Last_{variable1}_Days'] * 100
    
    # Calculate future metrics
    data[f'High_Next_{variable2}_Days'] = data['High'].shift(-variable2).rolling(window=variable2).max()
    data[f'%_Diff_From_High_Next_{variable2}_Days'] = (data['Close'] - data[f'High_Next_{variable2}_Days']) / data[f'High_Next_{variable2}_Days'] * 100
    
    data[f'Low_Next_{variable2}_Days'] = data['Low'].shift(-variable2).rolling(window=variable2).min()
    data[f'%_Diff_From_Low_Next_{variable2}_Days'] = (data['Close'] - data[f'Low_Next_{variable2}_Days']) / data[f'Low_Next_{variable2}_Days'] * 100
    
    # Drop rows with NaN values
    data = data.dropna()
    
    return data

# Step 3: Train Machine Learning Model
def train_model(X, y_high, y_low):
    # Split the data into training and testing sets
    X_train, X_test, y_high_train, y_high_test, y_low_train, y_low_test = train_test_split(
        X, y_high, y_low, test_size=0.2, random_state=42
    )
    
    # Initialize the model
    model_high = RandomForestRegressor(n_estimators=100, random_state=42)
    model_low = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train the model
    model_high.fit(X_train, y_high_train)
    model_low.fit(X_train, y_low_train)
    
    # Evaluate the model
    y_high_pred = model_high.predict(X_test)
    y_low_pred = model_low.predict(X_test)
    
    high_mae = mean_absolute_error(y_high_test, y_high_pred)
    low_mae = mean_absolute_error(y_low_test, y_low_pred)
    
    print(f'Mean Absolute Error for High: {high_mae}')
    print(f'Mean Absolute Error for Low: {low_mae}')
    
    # Save the model
    joblib.dump(model_high, 'model_high.pkl')
    joblib.dump(model_low, 'model_low.pkl')
    
    return model_high, model_low

# Step 4: Predict Outcomes
def predict_outcomes(model_high, model_low, input_features):
    # Load the models
    model_high = joblib.load('model_high.pkl')
    model_low = joblib.load('model_low.pkl')
    
    # Make predictions
    high_pred = model_high.predict([input_features])
    low_pred = model_low.predict([input_features])
    
    return high_pred[0], low_pred[0]

# Main script
if __name__ == "__main__":
    api_key = ''COINGECKO_API_KEY''  # Replace with your actual API key
    crypto_pair = "bitcoin/usd"
    
    # Fetch data
    df = fetch_crypto_data(crypto_pair, api_key)
    
    if df.empty:
        print("Failed to fetch data.")
    else:
        print("Fetched Data:")
        print(df.head())  # Print first few rows for verification
        
        # Define metrics
        variable1 = 7
        variable2 = 5
        
        # Calculate metrics
        metrics_df = calculate_metrics(df, variable1, variable2)
        
        print("Metrics DataFrame Columns:")
        print(metrics_df.columns)
        
        # Save to Excel
        with pd.ExcelWriter("C:/py pg/Assesments/crypto_data.xlsx") as writer:
              df.to_excel(writer, sheet_name='Raw Data', index=False)
              metrics_df.to_excel(writer, sheet_name='Metrics Data', index=False)

        print("Data and metrics saved to crypto_data.xlsx.")
        
        # Prepare the data for training
        features = [
            f'Days_Since_High_Last_{variable1}_Days',
            f'%_Diff_From_High_Last_{variable1}_Days',
            f'Days_Since_Low_Last_{variable1}_Days',
            f'%_Diff_From_Low_Last_{variable1}_Days'
        ]
        
        target_high = f'%_Diff_From_High_Next_{variable2}_Days'
        target_low = f'%_Diff_From_Low_Next_{variable2}_Days'

        # Ensure the DataFrame isn't empty and contains valid targets
        if metrics_df.empty or target_high not in metrics_df.columns or target_low not in metrics_df.columns:
            print("Not enough data to proceed with model training.")
        else:
            X = metrics_df[features]
            y_high = metrics_df[target_high]
            y_low = metrics_df[target_low]

            # Check for NaN values in y
            if y_high.isna().any() or y_low.isna().any():
                print("Error: Target variables contain NaN values.")
            else:
                # Train the model
                model_high, model_low = train_model(X, y_high, y_low)

                # Example usage for prediction
                input_features = [1, -0.90, 7, 4.76]  # Replace with actual input values
                high_pred, low_pred = predict_outcomes(model_high, model_low, input_features)
                print(f'Predicted % Diff From High Next {variable2} Days: {high_pred}')
                print(f'Predicted % Diff From Low Next {variable2} Days: {low_pred}')
