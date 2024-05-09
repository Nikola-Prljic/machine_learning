import numpy as np
import pandas as pd
import tensorflow as tf

def setup_data(scale_factor=1000.0):
    """  
Load train and test data from url
- scales value down
- shuffle train data
- returns train_df, test_df
    """
    train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
    test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")
    train_df["median_house_value"] /= scale_factor
    test_df["median_house_value"] /= scale_factor
    train_df = train_df.reindex(np.random.permutation(train_df.index))
    return train_df, test_df

# Keras Input tensors of float values.
def create_inputs_dict():
    inputs = {
    'latitude':
        tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                              name='latitude'),
    'longitude':
        tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                              name='longitude')
    }
    return inputs

def create_cross_layers(train_df:pd.DataFrame, resolution_in_degrees=0.4):
    inputs = create_inputs_dict()
    latitude_boundaries = list(np.arange(int(min(train_df['latitude'])),
                                        int(max(train_df['latitude'])),
                                        resolution_in_degrees))
    latitude = tf.keras.layers.Discretization(
    bin_boundaries=latitude_boundaries,
    name='discretization_latitude')(inputs.get('latitude'))
    longitude_boundaries = list(np.arange(int(min(train_df['longitude'])),
                                        int(max(train_df['longitude'])),
                                        resolution_in_degrees))

    # Create a Discretization layer to separate the longitude data into buckets.
    longitude = tf.keras.layers.Discretization(
    bin_boundaries=longitude_boundaries,
    name='discretization_longitude')(inputs.get('longitude'))

    # Cross the latitude and longitude features into a single one-hot vector.
    feature_cross = tf.keras.layers.HashedCrossing(
    num_bins=len(latitude_boundaries) * len(longitude_boundaries),
    output_mode='one_hot',
    name='cross_latitude_longitude')([latitude, longitude])

    dense_output = tf.keras.layers.Dense(units=1, name='dense_layer')(feature_cross)

    # Define an output dictionary we'll send to the model constructor.
    outputs = {
        'dense_output': dense_output
    }
    return inputs, outputs

