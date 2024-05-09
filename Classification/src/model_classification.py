import tensorflow as tf
import keras
from keras import layers
import pandas as pd
import numpy as np

def create_inputs_dict():
    # Features used to train the model on.
    inputs = {
        'median_income': keras.Input(shape=(1,)),
        'total_rooms': keras.Input(shape=(1,))
    }
    return inputs

def create_model(my_inputs, my_learning_rate, METRICS) -> keras.Model:
    # Use a Concatenate layer to concatenate the input layers into a single tensor.
    # as input for the Dense layer. Ex: [input_1[0][0], input_2[0][0]]
    concatenated_inputs = layers.Concatenate()(my_inputs.values())
    dense = layers.Dense(units=1, name="dense_layer", activation=tf.sigmoid)
    dense_output = dense(concatenated_inputs)
    """Create and compile a simple classification model."""
    my_outputs = {
        'dense': dense_output,
    }
    model = keras.Model(inputs=my_inputs, outputs=my_outputs)
    model.compile(optimizer=keras.optimizers.experimental.RMSprop(learning_rate=my_learning_rate),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=METRICS
                 )
    return model

def train_model(model: keras.Model, dataset: pd.DataFrame, epochs, label_name, batch_size=None, shuffle=True):
    """Feed a dataset into the model in order to train it."""

    # The x parameter of tf.keras.Model.fit can be a list of arrays, where
    # each array contains the data for one feature.  Here, we're passing
    # every column in the dataset. Note that the feature_layer will filter
    # away most of those columns, leaving only the desired columns and their
    # representations as features.

    features = {name:np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))

    history = model.fit(x=features, y=label, 
                        batch_size=batch_size,
                        epochs=epochs, 
                        shuffle=shuffle)
    
    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # Isolate the classification metric for each epoch.
    hist = pd.DataFrame(history.history)

    return epochs, hist

def evaluate_model(my_model: keras.Model, label_name, test_df_norm: pd.DataFrame, batch_size):
    features = {name:np.array(value) for name, value in test_df_norm.items()}
    label = np.array(features.pop(label_name))
    my_model.evaluate(x = features, y = label, batch_size=batch_size)
    return features, label