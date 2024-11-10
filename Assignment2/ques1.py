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

# normalize the data
mean = x.mean()
std = x.std()
x_ = (x - mean)/std


cov_matrix = x_.T.dot(x_) / (x_.shape[0] - 1)

eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

idx = np.argsort(eigen_values)[::-1]
eigen_values = eigen_values[idx]

new_vals = eigen_values[:2]
new_vecs = eigen_vectors[:, idx[:2]]

PCA_vals = x_.dot(new_vecs)
PCA_vals.to_csv('PCA.csv')

plt.scatter(PCA_vals.iloc[:, 0], PCA_vals.iloc[:, 1], c=y.map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}))
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA')
plt.grid()

# reduce dimension of new_vecs to 2 using PCA
print(new_vecs)
C = new_vecs.dot(new_vecs.T).T/(new_vecs.shape[0]-1)

eigen_values, eigen_vectors = np.linalg.eig(C)
idx = np.argsort(eigen_values)[::-1]
_vecs = eigen_vectors[:, idx[:2]]
C=new_vecs.T.dot(_vecs)
print(f'{_vecs=}\n{C=}')
# superimpose scaled eigendirections
for i in range(2):
    plt.quiver(0, 0, C[i][0]*new_vals[i], C[i][1]*new_vals[i], angles='xy', scale_units='xy', scale=1)

plt.show()

# Reconstruction
x_re = PCA_vals.dot(new_vecs.T)
x_re = x_re*std.values + mean.values

# find rmse for each feature
x.columns = [0, 1, 2, 3]

def rmse(y_true, y_pred):
    rmse = 0
    for i in y_pred.index:
        rmse += (y_true[i] - y_pred[i])**2
    return (rmse/len(y_true))**0.5


rmseval = 0
print('RMSE values for each feature:')
for i in x.columns:
    rmseval = rmse(x[i], x_re[i])
    print(rmseval)