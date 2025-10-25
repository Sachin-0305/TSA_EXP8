# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 25/10/2025

### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
```
Read the dataset:
```
data = pd.read_csv('GoogleStockPrices.csv')

data['Date'] = pd.to_datetime(data['Date'])

data.set_index('Date', inplace=True)

price_data = data[['Close']]

print("Shape of the dataset:", price_data.shape)
print("First 10 rows of the dataset:")
print(price_data.head(10))
```
Plot Original Data:
```
plt.figure(figsize=(12, 6))
plt.plot(price_data['Close'], label='Original Google Stock Price', color='blue')
plt.title('Original Google Stock Price Data')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()
```
Data Transformation:
```
data_monthly = price_data.resample('MS').mean()

scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),
    index=data_monthly.index
)
scaled_data = scaled_data + 1
```
Exponential Smoothing:
```
x = int(len(scaled_data) * 0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]

model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
test_predictions_add = model_add.forecast(steps=len(test_data))
```
Plot train/test/predictions:
```
ax = train_data.plot(figsize=(12, 6))
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["Train Data", "Predictions", "Test Data"])
ax.set_title('Visual Evaluation of Forecast')
plt.grid()
plt.show()

# RMSE
rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print("Root Mean Square Error (RMSE):", rmse)

# Variance and Mean:

print("Variance:", np.sqrt(scaled_data.var()), "Mean:", scaled_data.mean())
```
 Future Predictions:
```
model_full = ExponentialSmoothing(scaled_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
predictions = model_full.forecast(steps=int(len(data_monthly) / 4)) 

ax = scaled_data.plot(figsize=(12, 6))
predictions.plot(ax=ax)
ax.legend(["Monthly Data", "Future Predictions"])
ax.set_xlabel('Date')
ax.set_ylabel('Scaled Stock Price')
ax.set_title('Future Prediction of Google Stock Prices')
plt.grid()
plt.show()
```
### OUTPUT:
Original Data:

<img width="432" height="308" alt="image" src="https://github.com/user-attachments/assets/012f64ce-83c2-4476-98fb-6237c022dab3" />

<img width="881" height="479" alt="image" src="https://github.com/user-attachments/assets/769dcf08-e2f8-4709-82a3-1e13a9ecd3d3" />

Moving Average:

<img width="443" height="832" alt="image" src="https://github.com/user-attachments/assets/2772bf06-e5d1-466a-8957-f2a00e909d3b" />

Plot Transform Dataset:

<img width="901" height="498" alt="image" src="https://github.com/user-attachments/assets/7e45f293-3d9b-4ac3-8bd6-6ec11c0e284a" />

Exponential Smoothing:

<img width="886" height="503" alt="image" src="https://github.com/user-attachments/assets/a4285928-aab9-447f-959a-46178ffae2ed" />

Performance metrics:

<img width="455" height="27" alt="image" src="https://github.com/user-attachments/assets/4745863f-bf95-421b-b399-1443ff1ecd48" />


### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
