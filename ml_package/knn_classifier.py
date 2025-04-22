### ml_package/knn_classifier.py
import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k  # use k nearest neighbors to make predictions

    def fit(self, X_train, y_train):
        # store the training data for use during prediction
        self.X_train = X_train
        self.y_train = np.array(y_train)  # convert to numpy array to enable indexing

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            # compute distance from this test point to every training point
            distances = np.linalg.norm(self.X_train - x, axis=1)

            # find indices of the k closest points
            k_indices = distances.argsort()[:self.k]

            # get the labels of those points
            k_nearest_labels = self.y_train[k_indices]

            # choose the most common label among the neighbors
            most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common_label)

        # return all predictions as a numpy array
        return np.array(predictions)