import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
from sklearn.preprocessing import MinMaxScaler

def normalize_dataset(dataset):
    """
    Normalize each column in the dataset.
    
    Parameters:
    dataset (DataFrame): The input dataset to be normalized.
    
    Returns:
    DataFrame: The normalized dataset.
    """
    scaler = MinMaxScaler()
    normalized_dataset = dataset.copy()
    for column in dataset.columns:
        if dataset[column].dtype in ['int64', 'float64']:
            column_data = dataset[column].values.reshape(-1, 1)
            normalized_column = scaler.fit_transform(column_data)
            normalized_dataset[column] = normalized_column.flatten()
    return normalized_dataset

def histogram(path: str):
    """
    This function displays histograms, one per course,
    showing the score distribution between all four Hogwarts' houses.

    Parameters: path of the dataset you want to visualize.

    Returns: none.
    """

    try:
        dataset = pd.read_csv(path, index_col=0)
        if not isinstance(dataset, pd.DataFrame):
            dataset = pd.DataFrame(dataset)
        
        numerical_features = dataset.select_dtypes(include=['number'])
        dataset_to_display = dataset[numerical_features.columns].fillna(0, inplace = False)

        columns_to_plot = dataset_to_display.columns
        categories = dataset["Hogwarts House"].unique()

        houses_colors ={'Hufflepuff': 'yellow', 'Slytherin':'green', 'Ravenclaw': 'blue', 'Gryffindor': 'red'}
        num_columns = 4
        num_rows = 4
        fig,axs= plt.subplots(nrows=4, ncols=4, figsize=(15, 20))
        plt.subplots_adjust(hspace=25, wspace=10)

        legend_labels = []
        legend_handles = []

        for i, col in enumerate(columns_to_plot):
            row = i // num_columns
            column = i % num_columns
            for j, category in enumerate(categories):
                data = dataset[dataset["Hogwarts House"] == category][col]
                axs[row, column].hist(data, bins=15, alpha=0.5, color=houses_colors.get(category, 'black'), label=category, orientation='vertical')
                if category not in legend_labels:
                    label = category
                    handles, _ = axs[row, column].get_legend_handles_labels()
                    legend_labels.append(label)
                    legend_handles.append((mpatches.Patch(alpha=0.5, color=houses_colors.get(category, 'black'), label=category))
)

            axs[row, column].set_title(col, fontsize=10)
            axs[row, column].tick_params(axis='x', labelsize=7)
            axs[row, column].tick_params(axis='y', labelsize=7)

        plt.legend(legend_handles, legend_labels, loc='lower right', bbox_to_anchor=(1, 0))

        for i in range(len(columns_to_plot), num_rows * num_columns):
            row = i // num_columns
            col = i % num_columns
            axs[row, col].axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    av = sys.argv
    histogram(av[1])