import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from src.plot_data import plot_the_loss_curve

def create_model(my_inputs, my_outputs, my_learning_rate):
  """Create and compile a simple linear regression model."""
  model = keras.Model(inputs=my_inputs, outputs=my_outputs)
  model.compile(optimizer=keras.optimizers.Adam(
      learning_rate=my_learning_rate),
      loss="mean_squared_error",
      metrics=[keras.metrics.MeanSquaredError()])
  return model

def create_normalized_label(train_df, test_df):
    train_median_house_value_normalized = keras.layers.Normalization(axis=None)
    train_median_house_value_normalized.adapt(
    np.array(train_df['median_house_value']))

    test_median_house_value_normalized = keras.layers.Normalization(axis=None)
    test_median_house_value_normalized.adapt(
    np.array(test_df['median_house_value']))
    return train_median_house_value_normalized, test_median_house_value_normalized

def train_model(train_median_house_value_normalized, model, dataset, epochs, batch_size, label_name, validation_split=0.1):
    """Feed a dataset into the model in order to train it."""
    features = {name:np.array(value) for name, value in dataset.items()}
    label = train_median_house_value_normalized(
        np.array(features.pop(label_name)))
    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=True, validation_split=validation_split)
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    mse = hist["mean_squared_error"]

    return epochs, mse, history.history

def get_outputs_linear_regression(preprocessing_layers):
    dense_output = keras.layers.Dense(units=1,
                                name='dense_output')(preprocessing_layers)
    outputs = {
    'dense_output': dense_output
    }
    return outputs

def linear_model(preprocessing_layers, train_df, test_df, inputs, learning_rate, epochs, batch_size):
    learning_rate = 0.01
    epochs = 15
    batch_size = 1000
    label_name = "median_house_value"

    # Split the original training set into a reduced training set and a
    # validation set.
    validation_split = 0.2

    outputs = get_outputs_linear_regression(preprocessing_layers)

    # Establish the model's topography.
    my_model = create_model(inputs, outputs, learning_rate)
    
    train_median_house_value_normalized, test_median_house_value_normalized = create_normalized_label(test_df, train_df)
    
    # Train the model on the normalized training set.
    epochs, mse, history = train_model(train_median_house_value_normalized, my_model, train_df, epochs, batch_size,
                            label_name, validation_split)
    plot_the_loss_curve(epochs, mse, history["val_mean_squared_error"])

    test_features = {name:np.array(value) for name, value in test_df.items()}
    test_label = test_median_house_value_normalized(test_features.pop(label_name)) # isolate the label
    print("\n Evaluate the linear regression model against the test set:")
    my_model.evaluate(x = test_features, y = test_label, batch_size=batch_size, return_dict=True)
