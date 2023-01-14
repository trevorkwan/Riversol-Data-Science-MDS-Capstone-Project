# author: Trevor Kwan
# date: 2020-05-27

'''This script takes in cleaned data and creates barplots on sample-takers that made a purchase
and sample-takers that did not purchase based on features: free_shipping, product_type, and skin_type.

Usage: prelim_explore_classification.py --data_path=<data_path> --file_path=<file_path>

Options:

--data_path=<data_path> name of data file that should be fetched
    eg. "../../data/processed/cleaned_df.csv"
  
--file_path=<file_path>  name of the folder where you want visualisations to be saved 
    eg. "output"
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
from docopt import docopt

opt = docopt(__doc__)


def main(data_path, file_path):

    df = pd.read_csv(f"{data_path}")

    # split data into train test to get train_df
    X = df.drop(columns='buy')
    y = df[['buy']]
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=123)

    train_df = pd.concat([X_train, y_train], axis = 1)

    # first plot
    skin_type_counts = plt.figure(figsize = (10, 6))
    skin_type_counts = sns.countplot(data = train_df, x = 'skin_type', hue = 'buy')
    skin_type_counts.set(xlabel = 'Skin Type', ylabel = 'Count', title = "Sample-Taker to Purchaser Based on Skin Type")
    skin_type_counts.legend(title = "Purchased")
    
    skin_type_counts.figure.savefig(os.path.join(file_path, 'skin_type_counts.png'))
    
    # second plot
    product_type_counts = plt.figure(figsize = (10, 6))
    product_type_counts = sns.countplot(data = train_df, x = 'product_type', hue = 'buy')
    product_type_counts.set(xlabel = 'Product Type', ylabel = 'Count', title = "Sample-Taker to Purchaser Based on Product Type")
    product_type_counts.legend(title = "Purchased")
    
    product_type_counts.figure.savefig(os.path.join(file_path, 'product_type_counts.png'))
    
    # third plot
    free_shipping_counts = plt.figure(figsize = (10, 6))
    free_shipping_counts = sns.countplot(data = train_df, x = 'free_shipping', hue = 'buy')
    free_shipping_counts.set(xlabel = 'Free Shipping', ylabel = 'Count', title = 'Sample-Taker to Purchaser Based on Free Shipping')
    free_shipping_counts.legend(title = "Purchased")

    free_shipping_counts.figure.savefig(os.path.join(file_path, 'free_shipping_counts.png'))
    
# check if images were saved
def test_images_created():
    main("../../data/processed/cleaned_df.csv", "output")
    assert os.path.isfile('output/skin_type_counts.png'), "skin_type_counts plot was not created in the output folder."
    assert os.path.isfile('output/product_type_counts.png'), "product_type_counts plot was not created in the output folder."
    assert os.path.isfile('output/free_shipping_counts.png'), "free_shipping_counts plot was not created in the output folder."
    
test_images_created()

if __name__ == "__main__":
    main(opt["--data_path"], opt["--file_path"])
