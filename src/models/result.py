# author: Lori Fang
# date: 2020-06-10

"""
This script takes in the validation/testing data and run the model we train on that.
And then it will export a dataframe including precison, recall, F1 score and number of the sample as well as a confusion matrix.
This script assumes the input cleaned dataset is the result from running the clf_model.py.

Usage: result.py --input=<input> --out_dir=<out_dir>

Options:
--input=<input>     Path (including filename) to the testing/validation data.
--out_dir=<out_dir> Path to directory where the dataframe and plot results will be saved.

"""

from docopt import docopt
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

opt = docopt(__doc__)

plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16,
                     'axes.labelweight': 'bold',
                     'figure.figsize': (8,6)})

def main(input, out_dir):
    df = pd.read_csv(input)
    X_test = df.drop(columns = ['buy'])
    y_test = df['buy']
    loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
    y_pred = loaded_model.predict(X_test)
    report = precision_recall_fscore_support(y_test, y_pred)
    model_report = pd.DataFrame(list(report),index=['Precision', 'Recall', 'F1-score', 'Support'], columns=['not_buy', 'buy']).T
    model_report.columns.values[0] = 'Label'
    conf_matrix_array = confusion_matrix(y_test, y_pred)
    matrix_numbers = conf_matrix_array / np.sum(conf_matrix_array, axis=1)[:, None]
    # save result dataframe into .csv file and plot as png
    try:
        model_report.to_csv(out_dir + "/model_report.csv")
        ax = sns.heatmap(matrix_numbers, linewidths=1, linecolor='slategrey', cmap='Blues',
                annot=True, xticklabels=['No Buy', 'Buy'], yticklabels=['No Buy', 'Buy'],
                cbar=False, square=True)
        ax.set_title("XGBClassifier on validation set")
        fig = ax.get_figure()    
        fig.savefig(out_dir +'/confusion_matrix.png')
    except Exception as e:
        print(f"Directory does not exist. Exception: {e}")


if __name__ == "__main__":
    main(opt["--input"], opt["--out_dir"])
