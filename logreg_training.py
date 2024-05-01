import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import joblib
import math
import sys

def feature_scaling(df):
    normalized_df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            col_mean = df[col].mean()
            col_std = df[col].std()
            normalized_df[col] = (df[col] - col_mean) / col_std
    return normalized_df

def softmax(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def grad(X, y, w):
    m = len(y)
    z = X.dot(w)
    y_pred = softmax(z)
    gradient = 1/m * X.T.dot(y_pred - y)
    return gradient

def gradient_descent(X, y, w, learning_rate, n_iteration):
    for i in range(0, n_iteration):
        w = w - learning_rate * grad(X, y, w)
    return w

def training(path):
    try:
        dataset = pd.read_csv(path)
        if not isinstance(dataset, pd.DataFrame):
            dataset = pd.DataFrame(dataset)
        
        df_subset = dataset.loc[:, ['Index', 'Hogwarts House', 'Astronomy', 'Herbology']]

        normalized_df = feature_scaling(df_subset).ffill()
        
        X = normalized_df[['Astronomy', 'Herbology']].values

        numerical_features = df_subset.select_dtypes(include=['number'])

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(normalized_df['Hogwarts House'])
        n_classes = len(label_encoder.classes_)

        w_models = []

        for i in range(n_classes):
            y = np.where(y_encoded == i, 1, 0).reshape(-1, 1)
            n_features = X.shape[1]
            w = np.zeros((n_features, 1))
            w_final = gradient_descent(X, y, w, learning_rate=0.001, n_iteration=100000)
            w_models.append(w_final)

        joblib.dump(w_models, "w_models.pkl")
        joblib.dump(label_encoder, "label_encoder.pkl")
    
    except FileNotFoundError:
        print("Error: Specified file path does not exist.")
    except Exception as e:
        print(f"Error: {e}")
    
if __name__ == "__main__":
    av = sys.argv
    training(av[1])
