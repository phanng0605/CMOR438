# ml_package/utils.py
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_iris_data():
    # read the extended iris dataset from csv
    df = pd.read_csv("data/iris_extended.csv")

    # encode species as the numeric target label
    if "species" in df.columns:
        le = LabelEncoder()
        df["target"] = le.fit_transform(df["species"])
        df["target_name"] = df["species"]  # keep the original for plotting

    # encode soil_type into numbers so it can be used in models
    if "soil_type" in df.columns:
        df["soil_type_encoded"] = LabelEncoder().fit_transform(df["soil_type"])

    return df

def normalize_data(X):
    # make sure X is a dataframe and filter only numeric columns
    if isinstance(X, pd.DataFrame):
        X_numeric = X.select_dtypes(include=['number'])
    else:
        X_df = pd.DataFrame(X)
        X_numeric = X_df.select_dtypes(include=['number'])

    # raise error if there are no numeric columns to normalize
    if X_numeric.shape[1] == 0:
        raise ValueError("no numeric columns available for normalization.")

    # scale all numeric features to [0, 1] range
    scaler = MinMaxScaler()
    return scaler.fit_transform(X_numeric), scaler

def split_data(X, y, test_size=0.2, random_state=42):
    # split the dataset into train and test sets
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def explore_and_visualize(df):
    # print basic statistics of the dataframe
    print("summary statistics:\n", df.describe())

    # show pairwise relationships with color coded by label if available
    if "target_name" in df.columns:
        sns.pairplot(df.select_dtypes(include=['number']).join(df["target_name"]), hue="target_name", diag_kind="hist")
    elif "target" in df.columns:
        sns.pairplot(df.select_dtypes(include=['number']).join(df["target"]), hue="target", diag_kind="hist")
    else:
        sns.pairplot(df.select_dtypes(include=['number']), diag_kind="hist")

    plt.suptitle("feature relationships", y=1.02)
    plt.show()

    # show correlation heatmap for numeric columns
    plt.figure(figsize=(10, 7))
    numeric_cols = df.select_dtypes(include=['number']).columns
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("feature correlation matrix")
    plt.show()

def visualize_predictions(X_test, y_true, y_pred, model_name):
    # plot true vs predicted labels for visual inspection
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(y_true)), y_true, label="true labels", marker='o')
    plt.scatter(range(len(y_pred)), y_pred, label="predicted labels", marker='x')
    plt.title(f"{model_name} predictions vs true labels")
    plt.xlabel("sample index")
    plt.ylabel("class")
    plt.legend()
    plt.grid(True)
    plt.show()
