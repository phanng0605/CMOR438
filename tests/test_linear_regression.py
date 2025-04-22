### tests/test_linear_regression.py
import numpy as np
from ml_package import LinearRegression

def test_linear_regression_fit_predict():
    # create a simple linear relationship y = 2x
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])

    # train the model on the data
    model = LinearRegression()
    model.fit(X, y)

    # predict on the training data
    preds = model.predict(X)

    # check that predictions are close to actual values
    assert np.allclose(preds, y, atol=1e-1)

# run the test manually
if __name__ == "__main__":
    test_linear_regression_fit_predict()
    print("all tests passed!")