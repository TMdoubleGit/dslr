import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import joblib
import sys

def feature_scaling(df):
    """
    This function generates a new dataset using z-score normalization without modifying the original one.

    Parameters:
        - dataset: pandas DataFrame containing the dataset.

    Returns:
        normalized dataset.
    """
    normalized_df = df.copy()

    for col in df.columns:
        if df[col].dtype in ['number']:
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

    Returns:
        array-like: this function returns a vector of predictions.
    """
    return X.dot(w)

def sigmoid(z):
    """
    Compute the sigmoid function.

    Parameters:
        z (array-like): The input to the sigmoid function.

    Returns:
        array-like: The output of the sigmoid function.
    """
    return 1 / (1 + np.exp(-z))

# def logistic_loss(y_pred, y_true):
#     """
#     Compute the logistic loss (cross-entropy loss).

#     Parameters:
#         y_pred (array-like): Predicted probabilities from the logistic regression model.
#         y_true (array-like): True binary labels.

#     Returns:
#         float: The logistic loss.
#     """
#     loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
#     return loss

def grad(X, y, w):
    """
    This function computes the gradient of the logistic loss function with respect to the model weights.

    Parameters:
        X (array-like): Matrix of features. Each row represents an observation, and each column represents a feature.
        y (array-like): Vector of binary target values corresponding to the observations (0 or 1).
        w (array-like): Vector of model weights.

    Returns:
        array-like: Vector representing the gradient of the logistic loss function with respect to the model weights.
    """
    m = len(y)
    z = model(X, w)
    y_pred = sigmoid(z)
    gradient = 1/m * X.T.dot(y_pred - y)
    return gradient

def gradient_descent(X, y, w, learning_rate, n_iteration):
    """
    Performs gradient descent optimization to update the model weights.

    Parameters:
        X (array-like): Matrix of features. Each row represents an observation, and each column represents a feature.
        y (array-like): Vector of target values corresponding to the observations.
        w (array-like): Initial vector of model weights.
        learning_rate (float): The learning rate determines the size of the step taken in each iteration.
        n_iteration (int): Number of iterations to perform gradient descent.

    Returns:
        Vector of updated model weights after performing gradient descent.
    """
    for i in range(0, n_iteration):
        w = w - learning_rate * grad(X, y, w)
    return w

def training(path):
    """
    Trains a logistic regression model using gradient descent optimization on the provided dataset.

    Parameters:
        path (str): The file path to the CSV dataset.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        Exception: If an unexpected error occurs during training.

    Returns:
        None. The trained model weights are saved to a file named 'w_final.pkl'.
    """
    try:
        dataset = pd.read_csv(path)
        if not isinstance(dataset, pd.DataFrame):
            dataset = pd.DataFrame(dataset)
        
        df_subset = dataset.loc[:, ['Index', 'Hogwarts House', 'Astronomy', 'Herbology']]

        normalized_df = feature_scaling(df_subset)
        X = normalized_df[['Astronomy', 'Herbology']].values

        X = np.nan_to_num(X, nan=0)

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(normalized_df['Hogwarts House'])
        y = y_encoded.reshape(-1, 1)

        n_features = X.shape[1]
        w = np.zeros((n_features, 1))
        w_final = gradient_descent(X, y, w, learning_rate=0.001, n_iteration=100000)
        joblib.dump(w_final, "w_final.pkl")
        joblib.dump(label_encoder, "label_encoder.pkl")
    
    except FileNotFoundError:
        print("Error: Specified file path does not exist.")
    except Exception as e:
        print(f"Error: {e}")
    
if __name__ == "__main__":
    av = sys.argv
    training(av[1])
