import pandas as pd
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys


def scatter_plots(dataset, features, save_path):
    """
    Generate scatter plots for all possible combinations of features.

    Parameters:
        dataset (pd.DataFrame): Pandas DataFrame containing the dataset.
        features (list): List of feature combinations to use for generating scatter plots.
        save_path (str): Path to save the output .jpg file.

    Returns:
        None
    """
    houses_colors = {'Hufflepuff': 'gold', 'Slytherin': 'green', 'Ravenclaw': 'royalblue', 'Gryffindor': 'firebrick'}
    num_plots = len(features)
    num_columns = 10
    num_rows = -(-num_plots // num_columns)

    root = tk.Tk()
    root.title("Scatter plots")
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry(f"{screen_width}x{screen_height}+0+0")

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(16, 10))

    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.4, wspace=0.3)

    legend_labels = []
    legend_handles = []

    for i, feature in enumerate(features):
        row = i // num_columns
        col = i % num_columns

        for category, color in houses_colors.items():
            data = dataset[dataset["Hogwarts House"] == category]
            axs[row, col].scatter(data[feature[0]], data[feature[1]], color=color, alpha=0.5, label=category)

            if category not in legend_labels:
                label = category
                handles, _ = axs[row, col].get_legend_handles_labels()
                legend_labels.append(label)
                legend_handles.append((mpatches.Patch(alpha=0.5, color=houses_colors.get(category, 'black'), label=category)))

        axs[row, col].set_title(f"{feature[0]} vs {feature[1]}", fontsize=6)

        axs[row, col].tick_params(axis='both', which='both', length=0, labelleft=False, labelbottom=False, labeltop=False, labelright=False)

        for i in range(len(features), num_rows * num_columns):
            row = i // num_columns
            col = i % num_columns
            axs[row, col].axis('off')

    plt.legend(legend_handles, legend_labels, loc='lower right', bbox_to_anchor=(1, 0), fontsize=7).set_title('Hogwarts Houses')

    note = "Notes:\nfeature[0] vs feature[1] where x-axis is the score of feature[0] and y-axis is the score of feature[1]\nDADA stands for Defense Against the Dark Arts"
    plt.text(0.01, 0.01, note, transform=fig.transFigure, fontsize=7, ha="left", va="bottom")

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    fig.savefig(save_path, format='jpg')

    root.mainloop()


def scatter_plots_from_csv(path, save_path):
    """
    Generate scatter plots based on the provided dataset.

    Parameters:
        path (str): Path of the dataset you want to visualize.
        save_path (str): Path to save the output .jpg file.
    """
    try:
        dataset = pd.read_csv(path, index_col=0)
        if not isinstance(dataset, pd.DataFrame):
            dataset = pd.DataFrame(dataset)

        dataset.rename(columns={"Defense Against the Dark Arts": "DADA"}, inplace=True)
        numerical_features = dataset.select_dtypes(include=['number'])
        dataset_to_display = dataset[numerical_features.columns].ffill()

        columns_to_plot = dataset_to_display.columns

        combinations = [(columns_to_plot[i], columns_to_plot[j]) for i in range(len(columns_to_plot)) for j in range(i+1, len(columns_to_plot))]

        scatter_plots(dataset, combinations, save_path)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    av = sys.argv
    scatter_plots_from_csv(av[1], "./plotting output/scatter_plot.png")
