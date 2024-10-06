import pandas as pd
import numpy as np

mean, eigen_values, eigen_vectors, Q = None, None, None, None

def get_gaussian_prob(x, mu, cov):
    # return np.exp(-0.5 * (x - mu) @ np.linalg.inv(cov) @ (x - mu).T) / np.sqrt(np.linalg.det(cov))
    return np.exp(-0.5 * (x - mu)**2 / cov) / np.sqrt(2*np.pi*cov)

def do_PCA(filename: str):
    global mean, eigen_values, eigen_vectors, Q

    database = pd.read_csv(filename)
    y = database['Species']
    x = database.drop('Species', axis=1)
    x.drop(labels='Unnamed: 0', axis=1, inplace=True)


    # replace outliers with median
    for i in x.columns:
        q1 = x[i].quantile(0.25)
        q3 = x[i].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        x[i] = x[i].apply(lambda t: t if t > lower_bound and t < upper_bound else x[i].median())

    mean = x.mean()
    x_mean_sub = x - mean
    corr_matrix = x_mean_sub.T @ x_mean_sub

    eigen_values, eigen_vectors = np.linalg.eig(corr_matrix)

    idx = np.argsort(eigen_values)[::-1][0]
    Q = eigen_vectors[:, idx].T

    x_new = x_mean_sub @ Q.T
    # x_new.to_csv(filename[:-4] + '_PCA.csv', index=False)

    return x_new, y

if __name__ == '__main__':
    x_train, y_train = do_PCA('iris_train.csv')
    
    database = pd.read_csv('iris_test.csv')
    y_test = database['Species']
    x_test = database.drop('Species', axis=1)
    x_test.drop(labels='Unnamed: 0', axis=1, inplace=True)
    x_test = (x_test - mean) @ Q.T

    y_pred = np.zeros_like(y_test)

    classes = list(set(y_train))
    means = np.zeros_like(classes, dtype=float)
    covs = np.zeros_like(classes, dtype=float)
    class_probs = np.zeros_like(classes, dtype=float)

    for i, c in enumerate(classes):
        x_train_c = x_train[y_train == c]
        mu_train_c = sum(x_train_c) / len(x_train_c)
        cov_train_c = (x_train_c - mu_train_c) @ (x_train_c - mu_train_c).T / len(x_train_c)
        means[i] = mu_train_c
        covs[i] = cov_train_c
        class_probs[i] = len(x_train_c) / len(x_train)
    
    for idx, test_sample in enumerate(x_test):
        probs = np.zeros_like(classes, dtype=float)
        for i, c in enumerate(classes):
            probs[i] = (get_gaussian_prob(test_sample, means[i], covs[i]) * class_probs[i])   
        probs = probs / sum(probs)
        y_pred[idx] = classes[np.argmax(probs)]
        # print(f"Predicted class: {classes[np.argmax(probs)]}, Actual class: {y_test[idx]}")
    
    # print(y_pred, y_test)
    print('Accuracy: ', np.sum(y_pred == y_test)/len(y_test)*100, '%')

    # confusion matrix
    confusion = pd.DataFrame(0, columns=classes, index=classes)

    for i in range(len(y_pred)):
        confusion.at[y_pred[i], y_test[i]] += 1
    
    print("Confusion matrix: \n", confusion)