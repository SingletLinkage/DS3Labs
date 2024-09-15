import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

X = pd.read_csv('PCA.csv', index_col=0)
y = pd.read_csv('Iris.csv')['Species']

def ED(a, b):
    return np.sqrt(np.sum((a - b)**2))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=104, shuffle=True)

y_pred = []

for x in X_test.index:
    distances = []
    for x_ in X_train.index:
        distances.append(ED(X_test.loc[x], X_train.loc[x_]))
    
    sorted_indices = np.argsort(distances)
    y_vals = y_train.iloc[sorted_indices]

    # print("Computed: ", y_vals[:5].mode().values[0])
    # print("Actual: ", y_test[x])
    y_pred.append(y_vals[:5].mode().values[0])

# Confusion Matrix
cf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
import matplotlib.pyplot as plt

plt.imshow(cf_matrix, cmap='Blues', interpolation='nearest')

# Add text on image
for i in range(len(cf_matrix)):
    for j in range(len(cf_matrix)):
        plt.text(j, i, str(cf_matrix[i, j]), ha='center', va='center', color='black')

plt.colorbar()
plt.xticks(np.arange(len(cf_matrix)), labels=np.unique(y_test), rotation=45)
plt.yticks(np.arange(len(cf_matrix)), labels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
