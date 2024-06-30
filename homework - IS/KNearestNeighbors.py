import pandas as pd
import numpy as np
from collections import Counter


class KNearestNeighbors:

    def __init__(self, n_neighbors=5) -> None:
        super().__init__()
        self.X_train = None
        self.y_train = None
        self.neighbors = n_neighbors
        self.score = None

    def get_data(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict_and_score(self, X_test, y_test):
        # Niz u kome se cuvaju predvidjanja
        prediction = []
        for j in range(len(X_test)):
            # Uzima se vrednost atributa jednog testa
            test = X_test.iloc[j].values
            train = self.X_train.values
            d = np.array(list(map(np.sum, np.array(list(map(np.abs, train - test))))))
            d = np.argsort(d)
            d = d[:self.neighbors]
            d = self.y_train[d]
            prediction.append(Counter(d).most_common(1)[0][0])

        # Preciznost implementacije
        score = Counter(np.array(list(map(np.abs, y_test.values - prediction)))).get(0)
        score = score / len(y_test)
        return prediction, score
