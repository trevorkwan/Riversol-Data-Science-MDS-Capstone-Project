# author: Lori Fang
# date: 2020-06-23

"""
This script takes in a clean format dataframe and predict whether the sample takers in the profile will become paying customers using the model we trained.
And then it will export a the dataframe with a prediction column called"buy_pred".
This script assumes the input dataset is clean.

Usage: predict.py --input=<input> --out_dir=<out_dir>

Options:
--input=<input>     Path (including filename) to the data.
--out_dir=<out_dir> Path to directory where the data with the prediction column will be saved.

"""

from docopt import docopt
import pandas as pd
import pickle

opt = docopt(__doc__)

def main(input, out_dir):
    df = pd.read_csv(input)
    X = df[["accepts_marketing", "ordered_month", "location", "gender", "free_shipping", "product_type", "skin_type", "fv_site"]]
    loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
    df["buy_pred"] = loaded_model.predict(X)
    try:
        df.to_csv(out_dir + "/prediction.csv", index=False)
    except Exception as e:
        print(f"Directory does not exist. Exception: {e}")

if __name__ == "__main__":
    main(opt["--input"], opt["--out_dir"])