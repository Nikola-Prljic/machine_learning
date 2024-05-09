import tensorflow as tf
import pandas as pd
import numpy as np

def create_model(my_inputs, my_outputs, my_learning_rate):
    model = tf.keras.Model(inputs=my_inputs, outputs=my_outputs)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(
        learning_rate=my_learning_rate),
        loss="mean_squared_error",
        metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def train_model(model, dataset, epochs, batch_size, label_name):
    """Feed a dataset into the model in order to train it."""
    features = {name:np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=True)
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]
    return epochs, rmse
