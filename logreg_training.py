import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import sys

def feature_scaling(df):
    """
    This function generates a new dataset using z-score normalization without modifying the original one.

    Parameters:
        - dataset: pandas DataFrame containing the dataset.

    Returns: normalized dataset.
    """
    normalized_df = df.copy()

    for col in df.columns:
        col_mean = df[col].mean()
        col_std = df[col].std()
        normalized_df[col] = (df[col] - col_mean) / col_std
    
    return normalized_df

def model(X, w):
    """
    This function return the matrix product of features contained in X and the weights contained w.

    Parameters:
        - X: matrix containing the features. Each row represents an observation and each column represents a feature.
        - w: vector containing the model's weights. Each weight corresponds to a feature in the data.

    Returns: this function returns a vector of predictions.
    """
    return X.dot(w)

def grad(X, y, w):
    """
    XX.

    Parameters:
        - X: matrix containing the features. Each row represents an observation and each column represents a feature.
        - w: vector containing the model's weights. Each weight corresponds to a feature in the data.

    Returns: XX.
    """
    m = len(y)
    return (1/m * X.T.dot(model(X, w) - y))

def gradient_descent(X, y, w, learning_rate, n_iteration):
    for i in range(0, n_iteration):
        w = w - learning_rate * grad(X, y, w)
    return w