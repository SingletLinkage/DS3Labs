import pandas as pd, numpy as np

def get_gaussian_prob(x, mu, cov):
    return np.exp(-0.5 * (x - mu) @ np.linalg.inv(cov) @ (x - mu).T) / np.sqrt(np.linalg.det(cov))

x_train = pd.read_csv('iris_train.csv')
y_train = x_train['Species']
x_train.drop(['Species', 'Unnamed: 0'], axis=1, inplace=True)

x_test = pd.read_csv('iris_test.csv')
y_test = x_test['Species']
x_test.drop(labels=['Species', 'Unnamed: 0'], axis=1, inplace=True)
    
classes = list(set(y_train))
means, covs, class_probs = [], [], np.zeros_like(classes, dtype=float)

for i, c in enumerate(classes):
    x_train_c = x_train[y_train == c]
    mu_train_c = pd.Series([sum(x_train_c[i]) / len(x_train_c[i]) for i in x_train_c.columns], index=x_train_c.columns)
    cov_train_c = (x_train_c - mu_train_c).T @ (x_train_c - mu_train_c) / len(x_train_c)

    means.append(mu_train_c)
    covs.append(cov_train_c)
    class_probs[i] = len(x_train_c) / len(x_train)

for _x, _y in [(x_train, y_train), (x_test, y_test)]:
    y_pred = []
    for idx, test_sample in _x.iterrows():
        probs = []
        for i, c in enumerate(classes):
            probs.append(get_gaussian_prob(test_sample, means[i], covs[i]) * class_probs[i])
        probs = probs / sum(probs)
        y_pred.append(classes[np.argmax(probs)])

    confusion = pd.DataFrame(0, columns=classes, index=classes)
    for i in range(len(y_pred)):
        confusion.at[y_pred[i], _y[i]] += 1

    print('Accuracy: ', np.sum(y_pred == _y)/len(_y)*100, '%')
    print("Confusion matrix: \n", confusion)