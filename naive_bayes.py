import numpy as np


class NaiveBayes:
    def __init__(self):
        self._classes = None
        self._mean = None
        self._var = None
        self._priors = None

    def fit(self, X: np.array, y):
        n_samples, n_features = X.shape
        # get unique classes
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # init mean, variances, and priors
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for c in self._classes:
            X_c: np.array = X[c == y]
            self._mean[c, :] = X_c.mean(axis=0)
            self._var[c, :] = X_c.var(axis=0)
            # how often is class c occurring
            self._priors[c] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split

    def accuracy(y_true, y_pred):
        acc = np.sum(y_true == y_pred) / len(y_true)
        return acc

    _X, _y = make_blobs(n_samples=50, n_features=2, centers=2)
    X_train, X_test, y_train, y_test = train_test_split(_X, _y, test_size=0.2, random_state=42)

    model = NaiveBayes()
    model.fit(X_train, y_train)
    _y_pred = model.predict(X_test)

    print(F"Naive bayes accuracy is : {accuracy(y_test, _y_pred) * 100} %")
