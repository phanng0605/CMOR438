### tests/test_knn_classifier.py
import numpy as np
from ml_package import KNNClassifier

def test_knn_classifier_predict():
    # create simple training set with two classes
    X_train = np.array([[1], [2], [3], [6], [7], [8]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    # create test set with two points
    X_test = np.array([[2.5], [7.5]])

    # initialize and fit the model
    model = KNNClassifier(k=3)
    model.fit(X_train, y_train)

    # get predictions
    preds = model.predict(X_test)

    # check if predictions are correct (based on majority of nearest neighbors)
    assert np.array_equal(preds, np.array([0, 1]))

# run the test manually
if __name__ == "__main__":
    test_knn_classifier_predict()
    print("all tests passed!")