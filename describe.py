import pandas as pd
import math
import sys

def cleanData(dataframe):
    """
    This function replaces the NA values of a dataframe by 0.
    """
    dataframe.fillna(0, inplace=True)
    return dataframe


def describe(path: str):
    """
    This function loads a csv file and displays few statistics.
    """
    try:
        dataset = pd.read_csv(path, index_col=0)
        if not isinstance(dataset, pd.DataFrame):
            dataset = pd.DataFrame(dataset)

        numerical_features = dataset.select_dtypes(include=['number'])
        dataset_to_analyse = cleanData(dataset[numerical_features.columns])

        stats = []
        for colonne in dataset_to_analyse.columns:
            data = dataset_to_analyse[colonne]
            s_data = sorted(data)
            count = round(len(data), 2)
            mean = round(data.sum() / count, 2)
            std = round(math.sqrt((sum(pow(x - mean, 2) for x in s_data if not pd.isnull(x)) / count)), 2)
            minimum = round(s_data[0], 2)
            q25 = round(s_data[int(0.25 * count)], 2)
            q50 = round(s_data[int(count / 2)] if count % 2 == 1 else s_data[int(count / 2 + 1)], 2)
            q75 = round(s_data[int(0.75 * count)], 2)
            maximum = round(s_data[count - 1], 2)

            stats.append([colonne, count, mean, std, minimum, q25, q50, q75, maximum])
            columns = ['Feature', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
            formated_res = pd.DataFrame(stats, columns=columns)
            formated_res = formated_res.set_index('Feature').transpose()
            
        print(formated_res)
    except Exception as e:
        print(f"Error: {e}")

def main():
    av = sys.argv
    describe(av[1])
    return

if __name__ == "__main__":
    main()