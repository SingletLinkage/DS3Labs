import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

database = pd.read_csv('Iris.csv')

y = database['Species']
x = database.drop('Species', axis=1)

# replace outliers with median
for i in x.columns:
    q1 = x[i].quantile(0.25)
    q3 = x[i].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    x[i] = x[i].apply(lambda t: t if t > lower_bound and t < upper_bound else x[i].median())


x_mean_sub = x - x.mean()
corr_matrix = x_mean_sub.T @ x_mean_sub

print("Shape: mean_subtracted: ", x_mean_sub.shape)
print("Shape: correlation matrix: ", corr_matrix.shape)

eigen_values, eigen_vectors = np.linalg.eig(corr_matrix)

idx = np.argsort(eigen_values)[::-1][:2]
Q = eigen_vectors[:, idx].T

print("Q: \n", Q)
print(f'{Q.shape=}')

x_new = x_mean_sub @ Q.T

print("Shape: x_new: ", x_new.shape)
print(x_new) 
x_new.to_csv('PCA.csv')

plt.scatter(x_new.iloc[:, 0], x_new.iloc[:, 1], c=y.map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}))

# eigen directions
Q_reduced = Q @ Q.T
print(f'{Q_reduced=}')

# superimpose scaled eigendirections
for i in range(2):
    plt.quiver(0, 0, Q_reduced[0, i], Q_reduced[1, i], angles='xy', scale_units='xy', scale=1)

plt.show()


# Reconstruction

x_reconstructed = x_new @ Q
print("Shape: x_reconstructed: ", x_reconstructed.shape)

def rmse(y_true, y_pred):
    rmse = 0
    for i in y_pred.index:
        rmse += (y_true[i] - y_pred[i])**2
    return (rmse/len(y_true))**0.5


print('\n\nRMSE values for each feature:')
for i in range(len(x.columns)):
    rmseval = rmse(x_mean_sub.iloc[:, i], x_reconstructed.iloc[:, i])
    print(f'{x.columns[i]} {rmseval}')