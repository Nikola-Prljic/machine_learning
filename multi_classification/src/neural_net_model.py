import tensorflow as tf
import pandas as pd
import keras
from keras import layers
from matplotlib import pyplot as plt

def create_model(leaning_rate):
    model = keras.models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(units=300, activation='relu'))
    model.add(layers.Dropout(rate=0.09))
    model.add(layers.Dense(units=110, activation='relu'))
    model.add(layers.Dense(units=10, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=leaning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, train_features, train_label, epochs, batch_size=None, validation_split=0.1):
    history = model.fit(x=train_features, y=train_label, batch_size=batch_size, 
                        epochs=epochs, shuffle=True, validation_split=validation_split)
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    return epochs, hist

#@title Define the plotting function
def plot_curve(epochs, hist, list_of_metrics):
    """Plot a curve of one or more classification metrics vs. epoch."""  
    # list_of_metrics should be one of the names shown in:
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics  

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m)

    plt.legend()
    plt.show()

    print("Loaded the plot_curve function.")