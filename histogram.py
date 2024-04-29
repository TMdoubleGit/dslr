import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches
import sys

def create_histogram_figure(dataset, num_columns=4):
    """
    Create and return a matplotlib figure containing histograms.

    Parameters:
        dataset (pd.DataFrame): The dataset to visualize.
        num_columns (int): Number of columns for subplots.

    Returns:
        fig (matplotlib.figure.Figure): The created figure.
        axs (numpy.ndarray): Array of axes objects.
    """
    numerical_features = dataset.select_dtypes(include=['number'])
    dataset_to_display = dataset[numerical_features.columns].fillna(0, inplace = False)

    columns_to_plot = dataset_to_display.columns
    categories = dataset["Hogwarts House"].unique()

    houses_colors ={'Hufflepuff': 'gold', 'Slytherin':'green', 'Ravenclaw': 'royalblue', 'Gryffindor': 'firebrick'}
    num_rows = -(-len(columns_to_plot) // num_columns)

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(16, 10))

    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.4, wspace=0.3)

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
                legend_handles.append((mpatches.Patch(alpha=0.5, color=houses_colors.get(category, 'black'), label=category)))

        axs[row, column].set_title(col, fontsize=10)
        axs[row, column].tick_params(axis='x', labelsize=7)
        axs[row, column].tick_params(axis='y', labelsize=7)

    plt.legend(legend_handles, legend_labels, loc='lower right', bbox_to_anchor=(1, 0),fontsize=8).set_title('Hogwarts Houses')

    for i in range(len(columns_to_plot), num_rows * num_columns):
        row = i // num_columns
        col = i % num_columns
        axs[row, col].axis('off')

    note = "Note:\n x-axis represents the scores and y-axis the students' repartition given their score"
    plt.text(0.01, 0.01, note, transform=fig.transFigure, fontsize=7, ha="left", va="bottom")

    return fig, axs

def histogram(path: str, save_path: str):
    """
    This function displays histograms, one per course,
    showing the score distribution between all four Hogwarts' houses.

    Parameters:
        path (str): Path of the dataset you want to visualize.
        save_path (str): Path to save the output .jpg file.

    Returns: none.
    """
    try:
        dataset = pd.read_csv(path, index_col=0)
        if not isinstance(dataset, pd.DataFrame):
            dataset = pd.DataFrame(dataset)

        root = tk.Tk()
        root.title("Histograms")
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.geometry(f"{screen_width}x{screen_height}+0+0")

        fig, axs = create_histogram_figure(dataset)

        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        fig.savefig(save_path, format='jpg')

        root.mainloop()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    av = sys.argv
    histogram(av[1], "./plotting output/histogram.jpg")