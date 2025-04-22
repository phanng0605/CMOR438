### README.md
# Custom Machine Learning Package

For CMOR 438 modules: This project demonstrates a Python machine learning package built from scratch and applied to the Iris dataset.
---

## Features

### Core Models
- **LinearRegression**: Implements simple linear regression with least-squares estimation and bias handling.
- **KNNClassifier**: Implements k-nearest neighbors classification with distance-based majority voting.

### Utilities (`ml_package/utils.py`)
- **load_iris_data()**: Loads the extended Iris dataset from CSV and encodes categorical columns (`species` as target, `soil_type` as numeric).
- **normalize_data(X)**: Applies MinMax scaling to numerical features.
- **split_data(X, y)**: Splits the data into training and test sets using sklearn’s `train_test_split`.
- **explore_and_visualize(df)**: Displays summary statistics, pairplots, and a correlation heatmap for EDA.
- **visualize_predictions(X_test, y_true, y_pred, model_name)**: Visualizes true vs predicted labels on a scatter plot.

## Project Structure
```
ml_project/
├── data/         # Core ML implementations
    ├── iris_extended.csv       # Data used
├── ml_package/         # Core ML implementations
    ├── __init__.py
    ├── knn_classifier.py    # Implementation of knn classifier
    ├── linear_regression.py    # Implementation of linear regression
    └── utils.py    # Utility functions
├── tests/              # Unit tests
    ├── test_knn_classifier.py      # Unit test of knn classifier     
    └── test_linear_regression.py       # Unit test of linear regression
├── notebook.ipynb        # Jupyter notebooks (EDA, modeling,..)
├── README.md
└── requirements.txt
```

## Install
```bash
pip install -r requirements.txt
```

## Run Tests
```bash
pytest tests/
```

## Example Usage
See [`notebook.ipynb`](notebook.ipynb).