import pandas as pd
import matplotlib.pyplot as plt
import sys

def histogram(path: str):
    """
    This function displays histograms, one per course,
    showing the score distribution between all four Hogwarts' houses.
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
        fig,axs= plt.subplots(nrows=len(columns_to_plot), figsize=(10, 10))

        for i, col in enumerate(columns_to_plot):
            for j, category in enumerate(categories):
                data = dataset[dataset["Hogwarts House"] == category][col]
                axs[i].hist(data, bins=10, alpha=0.5, color=houses_colors.get(category, 'black'), label=category, orientation='vertical')
            axs[i].set_title(col)
            axs[i].set_xlabel('Hogwarts House')
            axs[i].set_ylabel("Scores")
            # axs[i].legend()

        plt.tight_layout()

        plt.show()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    av = sys.argv
    histogram(av[1])