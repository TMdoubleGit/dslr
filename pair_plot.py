import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys

def pair_plots(dataset, save_path):
    """
    This function generates pair plots for all numerical features in the dataset.

    Parameters:
        - dataset: pandas DataFrame containing the dataset.

    Returns: none.
    """
    root = tk.Tk()
    root.title("Pair plot")
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry(f"{screen_width}x{screen_height}+0+0")

    sns.set(style="ticks")
    g = sns.pairplot(dataset, diag_kind='kde', hue='Hogwarts House', palette='colorblind')

    for ax in g.axes.flatten():
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='both', which='both', length=0, labelleft=False, labelbottom=False, labeltop=False, labelright=False)

    g._legend.remove()

    note = "Notes:\nFeatures order is Arithmancy, Astronomy, Herbology, DADA, Divination, Muggle Studies, Ancient Runes, History of Magic, Transfiguration, Potions, Care of Magical Creatures, Charms and Flying"
    plt.text(0.01, 0.01, note, transform=plt.gcf().transFigure, fontsize=7, ha="left", va="bottom")

    canvas = FigureCanvasTkAgg(plt.gcf(), master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    root.mainloop()

    g.savefig(save_path, format='png')

def pair_plots_from_csv(path, save_path):
    """
    This function generates pair plots based on the provided dataset.

    Parameters: path of the dataset you want to visualize.

    Returns: none.
    """
    try:
        dataset = pd.read_csv(path, index_col=0)
        if not isinstance(dataset, pd.DataFrame):
            dataset = pd.DataFrame(dataset)

        dataset.rename(columns={"Defense Against the Dark Arts": "DADA"}, inplace=True)
        numerical_features = dataset.select_dtypes(include=['number'])
        dataset_to_display = dataset[numerical_features.columns].ffill()

        dataset_to_display['Hogwarts House'] = dataset['Hogwarts House']

        pair_plots(dataset_to_display, save_path)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    av = sys.argv
    pair_plots_from_csv(av[1], "./plotting output/pair_plot.png")
