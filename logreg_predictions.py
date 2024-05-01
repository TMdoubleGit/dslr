import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import joblib
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

def model(X, w):
    return X.dot(w)

def predict(X, w_models, label_encoder):
    probabilities = softmax(X.dot(w_models))
    predictions = np.argmax(probabilities, axis=1)
    decoded_predictions = label_encoder.inverse_transform(predictions)
    return decoded_predictions

def logreg_predict(path, output_path):
    try:
        w_models = joblib.load("w_models.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
    except FileNotFoundError as e:
        print (e)
        print("Please launch the training program first !")
        exit(1)
    try:
        dataset = pd.read_csv(path)
        if not isinstance(dataset, pd.DataFrame):
            dataset = pd.DataFrame(dataset)
        
        df_subset = dataset.loc[:, ['Index', 'Hogwarts House', 'Astronomy', 'Herbology']]

        normalized_df = feature_scaling(df_subset).ffill()
        X = normalized_df[['Astronomy', 'Herbology']].values

        predictions = predict(X, w_models, label_encoder)

        predictions_df = pd.DataFrame({
            "Index": dataset["Index"],
            "First Name": dataset["First Name"],
            "Last Name": dataset["Last Name"],
            "Predicted House": predictions
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
