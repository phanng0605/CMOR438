### ml_package/linear_regression.py
import numpy as np

class LinearRegression:
    def __init__(self):
        self.coef_ = None  # weights for each feature
        self.intercept_ = None  # bias term

    def fit(self, X, y):
        # add a column of ones to X to account for the bias term
        X = np.c_[np.ones(X.shape[0]), X]

        # compute the optimal theta using the pseudoinverse method
        theta = np.linalg.pinv(X.T @ X) @ X.T @ y

        # split theta into bias and coefficients
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]

    def predict(self, X):
        # return the predicted values using learned weights and bias
        return X @ self.coef_ + self.intercept_