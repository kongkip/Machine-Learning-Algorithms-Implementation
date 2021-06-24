import numpy as np


class LinearRegressor:
    def __init__(self):
        self.m = 0  
        self.b = 0

    @staticmethod
    def sum_squares(x, y):
        return sum((x - y) ** 2)

    def fit(self, x, y):
        self.m = sum((x - np.mean(x)) * (y - np.mean(y))) / sum((x - np.mean(x)) ** 2)
        self.b = np.mean(y) - self.m * np.mean(x)

    def coef(self, y, y_hat):
        return 1 - self.sum_squares(y, y_hat) / self.sum_squares(y, np.mean(y))

    def predict(self, x):
        return self.m * x + self.b


if __name__ == '__main__':
    X = np.linspace(0, 10, 10)

    m, b = 3, -2
    Y = m * X + b + 0.1 * np.random.randn(X.shape[0])

    lr = LinearRegressor()
    lr.fit(X, Y)

    _y_hat = lr.predict(X)

    R2 = lr.coef(Y, _y_hat)

    print(lr.m, lr.b)
    print(R2)
