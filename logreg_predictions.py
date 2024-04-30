import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import logreg_training
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

def logreg_predict(path, output_path):
    try:
        w_final = joblib.load("w_final.pkl")
    except FileNotFoundError as e:
        print (e)
        print("Please launch the training program first !")
        exit(1)
    try:
        dataset = pd.read_csv(path)
        if not isinstance(dataset, pd.DataFrame):
            dataset = pd.DataFrame(dataset)
        
        df_subset = dataset.loc[:, ['Index', 'Hogwarts House', 'Astronomy', 'Herbology']]

        normalized_df = feature_scaling(df_subset)
        X = normalized_df[['Astronomy', 'Herbology']].values
        X = np.nan_to_num(X, nan=0)

        label_encoder = joblib.load("label_encoder.pkl")

        y_pred = model(X, w_final)
        # print(X)
        # print(w_final)

        decoded_predictions = label_encoder.inverse_transform(y_pred.flatten())
        predictions_df = pd.DataFrame({
            "Index": dataset["Index"],
            "First Name": dataset["First Name"],
            "Last Name": dataset["Last Name"],
            "Predicted House": decoded_predictions
        })

        predictions_df.to_csv(output_path, index=False)
        print("Predictions saved to: ", output_path)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_csv output_csv")
        exit(1)
logreg_predict(sys.argv[1], sys.argv[2])

