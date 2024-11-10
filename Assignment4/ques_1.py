import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ---------- Part I -----------------
df = pd.read_csv('abalone.csv')
train, test = train_test_split(df, random_state=42)

train.to_csv('abalone_train.csv', index=False)
test.to_csv('abalone_test.csv', index=False)

# ------------------ Part 2 -------------------------
correlations = df.corr()['Rings'].abs().sort_values(ascending=False)
best_feature = correlations.index[1]

# Simple Linear Regression
X_train = train[best_feature].values.reshape(-1, 1)
y_train = train['Rings'].values
X_test = test[best_feature].values.reshape(-1, 1)
y_test = test['Rings'].values

def linear_regression(X, y):
    X = np.column_stack((np.ones(X.shape[0]), X))
    coeffs = np.linalg.inv(X.T @ X) @ X.T @ y
    return coeffs

coeffs = linear_regression(X_train, y_train)

# plotting
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.5)
plt.plot(X_train, coeffs[0] + coeffs[1] * X_train, color='r', label='Best-fit line')
plt.xlabel(best_feature)
plt.ylabel('Rings')
plt.title(f'Linear Regression: {best_feature} vs Rings')
plt.legend()
plt.show()

# RMSE
def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted)**2))

# error
y_train_pred = coeffs[0] + coeffs[1] * X_train
y_test_pred = coeffs[0] + coeffs[1] * X_test

train_rmse = rmse(y_train, y_train_pred)
test_rmse = rmse(y_test, y_test_pred)

print(f"Linear Regression Train RMSE: {train_rmse:.4f}")
print(f"Linear Regression Test RMSE: {test_rmse:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Rings')
plt.ylabel('Predicted Rings')
plt.title('Actual vs Predicted Rings (Test Data)')
plt.show()

# ------------------------ Part 3 -------------------
def polynomial_regression(X, y, degree):
    X_poly = np.column_stack([X**i for i in range(degree+1)])
    coeffs = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
    return coeffs

train_rmse_list = []
test_rmse_list = []

for degree in range(2, 6):
    coeffs = polynomial_regression(X_train, y_train, degree)
    
    X_train_poly = np.column_stack([X_train**i for i in range(degree+1)])
    X_test_poly = np.column_stack([X_test**i for i in range(degree+1)])
    
    y_train_pred = X_train_poly @ coeffs
    y_test_pred = X_test_poly @ coeffs
    
    train_rmse = rmse(y_train, y_train_pred)
    test_rmse = rmse(y_test, y_test_pred)
    
    train_rmse_list.append(train_rmse)
    test_rmse_list.append(test_rmse)

# Plot RMSE vs degree
plt.figure(figsize=(10, 6))
plt.bar(range(2, 6), train_rmse_list, alpha=0.5, label='Train RMSE')
plt.xlabel('Degree of Polynomial')
plt.ylabel('RMSE')
plt.title('RMSE vs Degree of Polynomial')
plt.legend()
plt.show()

# Find best degree based on test RMSE
best_degree = test_rmse_list.index(min(test_rmse_list)) + 2

# Plot best-fit curve
best_coeffs = polynomial_regression(X_train, y_train, best_degree)
X_plot = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
y_plot = sum([best_coeffs[i] * X_plot**i for i in range(best_degree+1)])

plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.5)
plt.plot(X_plot, y_plot, color='r', label=f'Best-fit curve (degree={best_degree})')
plt.xlabel(best_feature)
plt.ylabel('Rings')
plt.title(f'Polynomial Regression: {best_feature} vs Rings')
plt.legend()
plt.show()

print(f"Best polynomial degree: {best_degree}")
print(f"Best polynomial Test RMSE: {min(test_rmse_list):.4f}")