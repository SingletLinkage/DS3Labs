import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('asianpaint.csv', parse_dates=['Date'], index_col='Date')

# --------------------------- PART 1 --------------------
train_size = int(len(df) * 0.65)
train, test = df[:train_size], df[train_size:]

plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Open'], label='Train')
plt.plot(test.index, test['Open'], label='Test')
plt.title('Asian Paints Stock Price - Train vs Test')
plt.xlabel('Date')
plt.ylabel('Opening Price')
plt.legend()
plt.show()

# ----------------------- PART 2 --------------------------
def create_dataset(X, lag=1):
    df = pd.DataFrame(X)
    columns = [df.shift(i) for i in range(1, lag+1)]
    df = pd.concat(columns, axis=1)
    df.columns = ['t-'+str(i) for i in range(1, lag+1)]
    df['t'] = X
    df = df.dropna()
    return df

#  lagged dataset
lagged_data = create_dataset(train['Open'])


X = lagged_data['t-1']
y = lagged_data['t']
X = np.column_stack((np.ones(len(X)), X))

coeffs = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

print("AR(1) Model Coefficients:")
print(f"w0 (intercept): {coeffs[0]:.4f}")
print(f"w1: {coeffs[1]:.4f}")

test_lag = create_dataset(test['Open'])
X_test = np.column_stack((np.ones(len(test_lag)), test_lag['t-1']))
predictions = X_test.dot(coeffs)

plt.figure(figsize=(12, 6))
plt.plot(test.index[1:], test_lag['t'], label='Actual')
plt.plot(test.index[1:], predictions, label='Predicted')
plt.title('Asian Paints Stock Price - Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Opening Price')
plt.legend()
plt.show()

# Calculate RMSE and MAPE
def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted)**2))

def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

rmse_value = rmse(test_lag['t'], predictions)
mape_value = mape(test_lag['t'], predictions)

print(f"RMSE (%): {rmse_value/np.mean(test_lag['t'])*100:.2f}%")
print(f"MAPE (%): {mape_value:.2f}%")