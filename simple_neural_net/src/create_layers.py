import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
import seaborn as sns

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

def get_data() -> pd.DataFrame:
    train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
    train_df = train_df.reindex(np.random.permutation(train_df.index)) # shuffle the examples
    test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")
    return train_df, test_df

def create_inputs_dict() -> dict:
    inputs = {
        'latitude':
            layers.Input(shape=(1,), dtype=tf.float32, name='latitude'),
        'longitude':
            layers.Input(shape=(1,), dtype=tf.float32, name='longitude'),
        'median_income':
            layers.Input(shape=(1,), dtype=tf.float32, name='median_income'),
        'population':
            layers.Input(shape=(1,), dtype=tf.float32, name='population')
    }
    return inputs

def create_normalization_layer(train_df: pd.DataFrame, inputs: dict, name):
    new_layer = layers.Normalization(
        name='normalization' + name,
        axis=None)
    new_layer.adapt(train_df[name])
    new_layer = new_layer(inputs.get(name))
    return new_layer

def create_discretization_layer(train_df: pd.DataFrame, inputs: dict, name):
    boundaries = np.linspace(-3, 3, 20 + 1)
    normalization_layer = create_normalization_layer(train_df, inputs, name)
    discretization_layer = layers.Discretization(
        bin_boundaries=boundaries,
        name='discretization_'+ name)(normalization_layer)
    return discretization_layer, boundaries

def create_layers(train_df):
    inputs = create_inputs_dict()
    # Create a Normalization layer to normalize the population data.
    median_income = create_normalization_layer(train_df, inputs, 'median_income')
    population = create_normalization_layer(train_df, inputs, 'population')

    latitude, latitude_boundaries = create_discretization_layer(train_df, inputs, "latitude")
    longitude, longitude_boundaries = create_discretization_layer(train_df, inputs, "longitude")

    feature_cross = layers.HashedCrossing(
    # num_bins can be adjusted: Higher values improve accuracy, lower values
    # improve performance.
        num_bins=len(latitude_boundaries) * len(longitude_boundaries),
        output_mode='one_hot',
        name='cross_latitude_longitude')([latitude, longitude])
    
    # Concatenate our inputs into a single tensor.
    preprocessing_layers = layers.Concatenate()(
        [feature_cross, median_income, population])

    return preprocessing_layers, inputs