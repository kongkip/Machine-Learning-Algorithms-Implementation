import numpy as np
from sklearn.datasets import make_blobs


class SVM:
    def __init__(self, lr=0.0001, lambda_param=0.01, iterations=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.lr * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)


if __name__ == '__main__':
    _X, _y = make_blobs(n_samples=50, n_features=2, centers=2)
    _y = np.where(_y == 0, -1, 1)
    clf = SVM()
    clf.fit(_X, _y)

    print(clf.weights, clf.bias)

    print(clf.predict(_X[0]))
