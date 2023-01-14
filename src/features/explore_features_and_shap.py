# author: Trevor Kwan
# date: 2020-06-12

'''This script takes in cleaned data and creates plots showing feature importance in predicting
whether or not a sample-taker will buy.

Usage: explore_features_and_shap.py --input=<input> --out_dir=<out_dir>

Options:

--input=<input> name of data file that should be fetched
    eg. "../../data/processed/cleaned_df.csv"
  
--file_path=<out_dir>  name of the folder where you want visualisations to be saved 
    eg. "output"
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from docopt import docopt
import shap
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

opt = docopt(__doc__)


def main(input, out_dir):

    # load data and wrangle
    df = pd.read_csv(input, encoding = 'mac_roman')
    df = df[df['days_from_sample'] > 46]
    top_4 = df[(df['location'].str.contains('ONTARIO, CANADA')) | (df['location'].str.contains('BRITISH COLUMBIA, CANADA')) | (df['location'].str.contains('ALBERTA, CANADA')) | (df['location'].str.contains('QUEBEC, CANADA'))]

    # plot accepts_marketing = True and free_shipping = False by top 4 locations
    df = top_4.groupby(['accepts_marketing', 'free_shipping', 'location']).mean().sort_values(by = 'buy', ascending = False).drop(columns = ['customer_id', 'ordered_month'])[0:4]
    best_conversions = list(df.groupby('location').mean().sort_values(by = 'buy', ascending = False)['buy'])
    best_conversions.append(top_4['buy'].mean())
    locations = ['BRITISH COLUMBIA, CANADA', 'ALBERTA, CANADA', 'ONTARIO, CANADA', 'QUEBEC, CANADA', 'Baseline']
    yes_mark_no_ship_df = pd.DataFrame([locations, best_conversions]).T
    yes_mark_no_ship_df.columns = ['Location', 'Conversion Rate']

    bar_plot_1 = sns.catplot(x="Conversion Rate", y='Location', hue="Location", data=yes_mark_no_ship_df,
                        kind="bar", palette="muted")
    bar_plot_1.fig.subplots_adjust(top=0.9)
    bar_plot_1.fig.suptitle('Sample-Takers Who Accepted Marketing and Had No Free Shipping')
    bar_plot_1.fig.savefig(os.path.join(out_dir, 'bar_plot_1.png'))
    
    # plot accepts_marketing = False and free_shipping = True by top 4 locations
    df = top_4.groupby(['accepts_marketing', 'free_shipping', 'location']).mean().sort_values(by = 'buy', ascending = True).drop(columns = ['customer_id', 'ordered_month'])[0:4]
    best_conversions = list(df.groupby('location').mean().sort_values(by = 'buy', ascending = False)['buy'])
    best_conversions.append(top_4['buy'].mean())
    locations = ['BRITISH COLUMBIA, CANADA', 'ALBERTA, CANADA', 'ONTARIO, CANADA', 'QUEBEC, CANADA', 'Baseline']
    no_mark_yes_ship_df = pd.DataFrame([locations, best_conversions]).T
    no_mark_yes_ship_df.columns = ['Location', 'Conversion Rate']

    bar_plot_2 = sns.catplot(x="Conversion Rate", y='Location', hue="Location", data=no_mark_yes_ship_df,
                        kind="bar", palette="muted")
    bar_plot_2.fig.subplots_adjust(top=0.9)
    bar_plot_2.fig.suptitle('Sample-Takers Who Did Not Accept Marketing and Had Free Shipping')
    bar_plot_2.fig.savefig(os.path.join(out_dir, 'bar_plot_2.png'))
    
    # shap wrangling
    df = pd.read_csv(input)
    X = df.drop(columns = ['customer_id', 'buy', 'days_from_sample', 'ordered_year'])
    y = df['buy']
    seed = 123
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.4, stratify=y_train, random_state=seed)
    train_df = X_train.copy()
    train_df['buy'] = y_train
    valid_df = X_valid.copy()
    valid_df['buy'] = y_valid
    test_df = X_test.copy()
    test_df['buy'] = y_test

    le = LabelEncoder()
    X_train_num = X_train.apply(LabelEncoder().fit_transform)
    model = XGBClassifier(scale_pos_weight=(y_train == False).sum()/(y_train == True).sum())
    model.fit(X_train_num, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_num)

    # shap plots
    shap.force_plot(explainer.expected_value, shap_values[0,:], X_train.iloc[0,:], show = False, matplotlib = True).savefig(os.path.join(out_dir, 'force_plot.png'), bbox_inches='tight')

if __name__ == "__main__":
    main(opt["--input"], opt["--out_dir"])