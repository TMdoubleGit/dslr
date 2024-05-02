import pandas as pd
import numpy as np
import joblib
import sys


def feature_scaling(df):
    """
    Normalize the numerical features in the DataFrame using z-score normalization.

    Parameters:
        df (DataFrame): Input DataFrame containing numerical features.

    Returns:
        DataFrame: Normalized DataFrame with z-score scaling applied to numerical features.
    """
    normalized_df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            col_mean = df[col].mean()
            col_std = df[col].std()
            normalized_df[col] = (df[col] - col_mean) / col_std
    return normalized_df


def softmax(z):
    """
    Compute the softmax function.

    Parameters:
        z (array-like): Input array.

    Returns:
        array-like: Output of the softmax function.
    """
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def model(X, w):
    """
    Compute the model's predictions based on the input features and weights.

    Parameters:
        X (array-like): Input features.
        w (array-like): Model weights.

    Returns:
        array-like: Predicted values.
    """
    return X.dot(w)


def predict(X, w_models, label_encoder):
    """
    Make predictions using the trained logistic regression models.

    Parameters:
        X (array-like): Input features.
        w_models (list): List of model weights for each class.
        label_encoder (LabelEncoder): Encoder for target labels.

    Returns:
        array-like: Predicted labels.
    """
    probabilities = softmax(X.dot(w_models))
    predictions = np.argmax(probabilities, axis=1)
    decoded_predictions = label_encoder.inverse_transform(predictions.flatten())
    return decoded_predictions


def logreg_predict(path, output_path):
    """
    Make predictions on a dataset using trained logistic regression models.

    Parameters:
        path (str): Path to the input CSV dataset.
        output_path (str): Path to save the predictions CSV file.

    Returns:
        None. Predictions are saved to the output_path.
    """
    try:
        w_models = joblib.load("w_models.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
    except FileNotFoundError as e:
        print(e)
        print("Please launch the training program first!")
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
        print("Predictions saved to:", output_path)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py dataset_test.csv output.csv")
        exit(1)
    logreg_predict(sys.argv[1], sys.argv[2])
