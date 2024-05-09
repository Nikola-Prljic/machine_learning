import numpy as np
import pandas as pd
""" import tensorflow as tf
import keras
from keras import layers """

# The following lines adjust the granularity of reporting.
""" pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
keras.backend.set_floatx('float32') """

def download_split_shuffle() -> pd.DataFrame:
    train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
    test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")
    if train_df is None or test_df is None:
        print("Error\nDownload Failed")
        exit(1)
    train_df = train_df.reindex(np.random.permutation(train_df.index)) # shuffle the training set
    return train_df, test_df

def normalize_values(train_df:pd.DataFrame, test_df:pd.DataFrame) -> pd.DataFrame:
    """
Normalize values
-for 38
-Z-score = (38 - 60) / 10 = -2.2
Calculate the Z-scores of each column in the training set and
write those Z-scores into a new pandas DataFrame named train_df_norm.
"""
    train_df_mean = train_df.mean()
    train_df_std = train_df.std()
    train_df_norm = (train_df - train_df_mean)/ train_df_std
    test_df_norm = (test_df - train_df_mean) / train_df_std
    return train_df_norm, test_df_norm

def create_binary_label(train_df:pd.DataFrame, test_df:pd.DataFrame, threshold) -> pd.DataFrame:
    train_df["median_house_value_is_high"] = (train_df["median_house_value"] > threshold).astype(float)
    test_df["median_house_value_is_high"] = (test_df["median_house_value"] > threshold).astype(float)
    return train_df, test_df
