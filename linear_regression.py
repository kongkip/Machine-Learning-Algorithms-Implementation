import numpy as np
from sklearn.datasets import make_regression


def cost(y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)


class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, y_predicted - y)
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted


if __name__ == '__main__':
    _X, _Y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    print(_X.shape)

    model = LinearRegression(lr=0.01)
    model.fit(_X, _Y)

    _y_hat = model.predict(_X)
    mse = cost(_Y, _y_hat)

    print(mse)
