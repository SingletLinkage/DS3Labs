import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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


pca = PCA()
X_pca = pca.fit_transform(x)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c='b', marker='o')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA: First vs Second Principal Component')
plt.grid()

eigen_vectors = pca.components_
origin = 0,0
plt.quiver(
    origin[0], origin[1],
    eigen_vectors[0, 0], eigen_vectors[0, 1],
    angles='xy', scale_units='xy', scale=1, color='r', label='Eigenvector 1'
)
plt.quiver(
    origin[0], origin[1],
    eigen_vectors[1, 0], eigen_vectors[1, 1],
    angles='xy', scale_units='xy', scale=1, color='g', label='Eigenvector 2'
)
plt.legend()

plt.show()