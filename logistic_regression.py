import numpy as np


class LogisticRegression:

    def __init__(self, lr, n_iters):
        self.lr = lr
        self.n_iters = n_iters

        self.weights = None
        self.bias = None

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # initialize weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) * self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias

        # return 1 where z is greater than 0.5 else 0
        y_predicted = self._sigmoid(linear)
        y_predicted_cls = np.where(y_predicted > 0.5, 1, 0)
        return y_predicted_cls


if __name__ == '__main__':
    x1 = np.random.randn(5, 2) + 5
    x2 = np.random.randn(5, 2) - 5
    _X = np.concatenate([x1, x2], axis=0)
    y = np.concatenate([np.ones(5), np.zeros(5)], axis=0)

    model = LogisticRegression(lr=0.01, n_iters=5)
    model.fit(_X, y)
    
    pred = model.predict(_X)
    print(y)
    print(pred)
