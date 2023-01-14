# author: Lori Fang
# date: 2020-06-10

"""
This script takes the cleaned dataset as the input, split the dataset into train, valid, and test sets and export them.
And then it will train the model using the training dataset and save the model.
This script assumes the input cleaned dataset is the result from running the data_cleaning.py.

Usage: clf_model.py --input=<input> --out_dir=<out_dir>

Options:
--input=<input>     Path (including filename) to the cleaned data.
--out_dir=<out_dir> Path to directory where the separated dataset and model will be saved.
            
"""

from docopt import docopt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.utils.validation import column_or_1d
from xgboost import XGBClassifier
import pickle

opt = docopt(__doc__)


def main(input, out_dir):
    # load cleaned dataframe
    df = pd.read_csv(input)
    df = df[~((df["days_from_sample"]<46) & (df["buy"]==False))]
    X = df.drop(columns = ['customer_id', 'buy', 'ordered_year', 'days_from_sample'])
    y = df['buy']
    seed = 123
    # get test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    # get train/valid set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.4, stratify=y_train, random_state=seed)
    train_df = X_train.copy()
    train_df['buy'] = y_train
    valid_df = X_valid.copy()
    valid_df['buy'] = y_valid
    test_df = X_test.copy()
    test_df['buy'] = y_test
    categorical_features = ['accepts_marketing', 'ordered_month', 'gender', 'free_shipping',
                       'product_type', 'skin_type', 'location', 'fv_site']
    categorical_transformer = Pipeline(steps =[
                                            ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'unknown')),
                                            ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
    ])

    preprocessor = ColumnTransformer(transformers = [('cat', categorical_transformer, categorical_features)],
                                     remainder = "passthrough")
    
    model = XGBClassifier(scale_pos_weight=(y_train == False).sum()/(y_train == True).sum())
    clf = Pipeline(steps = [('preprocessor', preprocessor),
                                ('classifier', model)])
    clf.fit(X_train, y_train)
    filename = 'finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    # save train valid and test as .csv files
    try:
        train_df.to_csv(out_dir + "/train_df.csv", index=False)
        valid_df.to_csv(out_dir + "/valid_df.csv", index=False)
        test_df.to_csv(out_dir + "/test_df.csv", index=False)
        
    except Exception as e:
        print(f"Directory does not exist. Exception: {e}")


if __name__ == "__main__":
    main(opt["--input"], opt["--out_dir"])

