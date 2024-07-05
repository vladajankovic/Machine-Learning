import time

import pandas as pd
import numpy as np
from math import ceil


class LinearRegressionGradientDescent:

    def __init__(self, alpha=0):
        self.W = None
        self.alpha = alpha

    def h(self, X: np.ndarray):
        return X.dot(self.W)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, learning_rate, iter: int):
        _X = pd.DataFrame(data=np.ones(shape=(len(X), 1)), columns=['x0'])
        _X = _X.join(X.reset_index(drop=True))

        self.W = np.ones(shape=(len(_X.columns), 1))
        learning_rate = np.array(learning_rate)

        _X = _X.to_numpy()
        _y = y.to_numpy().reshape(-1, 1)

        m = len(_X)

        l2 = 1 - learning_rate * self.alpha / m
        l2[0] = 1

        for _ in range(iter):
            dJ = (1 / m) * _X.T.dot(self.h(_X) - _y)

            self.W = self.W * l2 - learning_rate * dJ

        return self

    def predict(self, X: pd.DataFrame):
        _X = pd.DataFrame(data=np.ones(shape=(len(X), 1)), columns=['x0'])
        _X = _X.join(X.reset_index(drop=True)).to_numpy()

        return self.h(_X).reshape(-1, 1).flatten()

    def error_function(self, X: pd.DataFrame, y: pd.DataFrame):
        m = len(X)
        _X = pd.DataFrame(data=np.ones(shape=(len(X), 1)), columns=['x0'])
        _X = _X.join(X.reset_index(drop=True)).to_numpy()
        _y = y.to_numpy().reshape(-1, 1)

        return (1 / (2 * m)) * (np.sum((self.h(_X) - _y) ** 2) + self.alpha * np.sum(self.W[1:] ** 2))

    def score(self, y_pred: np.ndarray, y_true: np.ndarray):
        # R^2 score
        return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)


class LogisticRegression:

    def __init__(self, alpha=0):
        self.W = None
        self.alpha = alpha

    def h(self, X: np.ndarray):
        return 1 / (1 + np.exp(-X.dot(self.W)))

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, learning_rate, iter: int):
        _X = pd.DataFrame(data=np.ones(shape=(len(X), 1)), columns=['x0'])
        _X = _X.join(X.reset_index(drop=True))

        self.W = np.ones(shape=(len(_X.columns), 1))
        learning_rate = np.array(learning_rate)

        _X = _X.to_numpy()
        _y = y.to_numpy().reshape(-1, 1)

        m = len(_X)

        l2 = 1 - learning_rate * self.alpha / m
        l2[0] = 1

        for _ in range(iter):
            dJ = (1 / m) * _X.T.dot(self.h(_X) - _y)

            self.W = self.W * l2 - learning_rate * dJ

        return self

    def predict(self, X: pd.DataFrame):
        _X = pd.DataFrame(data=np.ones(shape=(len(X), 1)), columns=['x0'])
        _X = _X.join(X.reset_index(drop=True)).to_numpy()

        return np.round(self.h(_X).reshape(-1, 1).flatten())

    def error_function(self, X: pd.DataFrame, y: pd.DataFrame):
        m = len(X)
        _X = pd.DataFrame(data=np.ones(shape=(len(X), 1)), columns=['x0'])
        _X = _X.join(X.reset_index(drop=True)).to_numpy()
        _y = y.to_numpy().reshape(-1, 1)

        return ((-1 / m) * (_y.T.dot(np.log(self.h(_X))) + (1 - _y).T.dot(np.log(1 - self.h(_X))))
                + (self.alpha / (2 * m)) * np.sum(self.W[1:] ** 2))

    def score(self, y_pred: np.ndarray, y_true: np.ndarray):
        # F1 score
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        p = tp / (tp + fp) if tp + fp > 0 else 0
        r = tp / (tp + fn) if tp + fn > 0 else 0
        if p + r == 0:
            return 0

        return 2 * p * r / (p + r)


class LogisticRegressionOneVsOne:

    def __init__(self, alpha=0):
        self.models = []
        self.alpha = alpha

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, learning_rate, iter: int):
        classes = np.unique(y)
        for idx, k1 in enumerate(classes):
            for k2 in classes[idx+1:]:
                _X = X.loc[(y == k1) | (y == k2)]
                _y = y.loc[(y == k1) | (y == k2)]

                positive_class_idx = _y == k1
                negative_class_idx = _y == k2

                _y.loc[positive_class_idx] = 1
                _y.loc[negative_class_idx] = 0

                t = time.time()
                self.models.append((k1, k2, LogisticRegression(alpha=self.alpha).fit(_X, _y, learning_rate, iter)))
                print(f"OneVsOne classifier {k1}_vs_{k2} created, t={time.time() - t:.3f}s")

        return self

    def predict(self, X: pd.DataFrame):
        n = 1 + int(np.sqrt(1 + 8 * len(self.models))) // 2
        predictions = np.zeros(shape=(len(X), n))

        for k1, k2, model in self.models:
            pred = model.predict(X)
            for idx, p in enumerate(pred):
                if p == 1:
                    predictions[idx][k1] += 1
                else:
                    predictions[idx][k2] += 1

        predictions = np.argmax(predictions, axis=1)

        return predictions

    def score(self, y_pred: np.ndarray, y_true: np.ndarray):
        # F1 micro == accuracy
        return np.mean(y_pred == y_true)


class LogisticRegressionMultinomial:

    def __init__(self):
        self.W = None

    def h(self, X: np.ndarray):
        exp_h = np.exp(X.dot(self.W))
        exp_sum = np.sum(exp_h, axis=1, keepdims=1)
        return exp_h / exp_sum

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, learning_rate, iter: int):
        _X = pd.DataFrame(data=np.ones(shape=(len(X), 1)), columns=['x0'])
        _X = _X.join(X.reset_index(drop=True))

        classes = np.unique(y)
        self.W = np.ones(shape=(len(_X.columns), len(classes)))

        _X = _X.to_numpy()
        _y = np.eye(len(classes))[y]

        m = len(_X)

        for k in range(iter):
            dJ = (1 / m) * _X.T.dot(self.h(_X) - _y)

            self.W = self.W - learning_rate * dJ

            if k % 1000 == 0:
                print(f'Multinomial fiting, iteration {k}/{iter}')

        return self

    def predict(self, X: pd.DataFrame):
        _X = pd.DataFrame(data=np.ones(shape=(len(X), 1)), columns=['x0'])
        _X = _X.join(X.reset_index(drop=True)).to_numpy()

        return np.argmax(self.h(_X), axis=1)

    def score(self, y_pred: np.ndarray, y_true: np.ndarray):
        # F1 micro == accuracy
        return np.mean(y_pred == y_true)


def KFoldBayesianTargetEncoding(data: pd.DataFrame, column: str, target: str, k: int, alpha: int):
    categories = sorted(data[column].unique().tolist())
    global_mean = data[target].mean()

    encoded = np.zeros(data.shape[0])
    data = data.sample(frac=1, random_state=7).reset_index(drop=True)

    fold_size = ceil(len(data) / k)
    for t in range(k):
        train_idx = list(range(0, t * fold_size)) + list(range(fold_size + t * fold_size, len(data)))
        test_idx = list(range(0 + t * fold_size, fold_size + t * fold_size if t < 9 else len(data)))

        encode_dict = {c: global_mean for c in categories}

        means = data.loc[train_idx, [column, target]].groupby(by=column).mean()
        counts = data.loc[train_idx, [column, target]].groupby(by=column).count()

        smoothed_means = (counts * means + alpha * global_mean) / (counts + alpha)

        for c in smoothed_means.index:
            encode_dict[c] = smoothed_means[target][c]

        encoded[test_idx] = data.loc[test_idx, column].map(encode_dict)

    encode_dict = data.join(pd.DataFrame(data=encoded, columns=['encoded']))
    encode_dict = encode_dict.loc[:, [column, 'encoded']].groupby(by=column).mean()['encoded'].to_dict()

    data = data.drop(columns=column)
    data[column] = encoded

    return data, encode_dict


def split_dataset(x: pd.DataFrame, y: pd.DataFrame, train_size: float, random_state: int):
    d = pd.DataFrame(x).join(y)
    d = d.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split_idx = ceil(train_size * len(d))
    train_idx = list(range(0, split_idx))
    test_idx = list(range(split_idx, len(d)))
    train_data = d.iloc[train_idx]
    test_data = d.iloc[test_idx]
    x1, x2 = train_data.drop(columns=y.name), test_data.drop(columns=y.name)
    y1, y2 = train_data[y.name], test_data[y.name]
    return x1, x2, y1, y2
