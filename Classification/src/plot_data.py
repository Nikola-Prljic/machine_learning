from matplotlib import pyplot as plt
import numpy as np

#@title Define the plotting function.
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

def plot_predict_against_real(model, features, label):
    features:  np.ndarray = model.predict(features)['dense']
    features:  np.ndarray = features[:,0]
    plt.scatter(np.arange(1, features.size + 1), features)
    plt.scatter(np.arange(1, features.size + 1), label)
    plt.show()