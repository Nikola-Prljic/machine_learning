import src.prepare_data as data
from src.model_classification import create_inputs_dict, create_model, train_model, evaluate_model
import tensorflow as tf
from keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from src.plot_data import plot_curve, plot_predict_against_real

learning_rate=0.001
batch_size = 100
epochs = 100
threshold = 265000
label_name = "median_house_value_is_high"
classification_threshold = 0.6
shuffle = True

METRICS = [BinaryAccuracy(name='accuracy', threshold=classification_threshold),
           Precision(name='precision', thresholds=classification_threshold),
           Recall(name='recall', thresholds=classification_threshold),
           AUC(name='auc', num_thresholds=100)]

if __name__ == "__main__":
    train_df, test_df = data.download_split_shuffle()
    train_df_norm, test_df_norm = data.normalize_values(train_df, test_df)
    train_df_norm, test_df_norm = data.create_binary_label(train_df, test_df, threshold)

    inputs = create_inputs_dict()
    model = create_model(inputs, learning_rate, METRICS)

    epochs, hist = train_model(model, train_df_norm, epochs, label_name, batch_size, shuffle)

    # Plot a graph of the metric(s) vs. epochs.
    # , 'precision', 'recall'
    list_of_metrics_to_plot = ['accuracy', 'precision', 'recall', 'auc']
    plot_curve(epochs, hist, list_of_metrics_to_plot)
    print()
    print('\033[92m### RESULT ###\033[0m'.rjust(75))
    features, label = evaluate_model(model, label_name, test_df_norm, batch_size)
    #plot_predict_against_real(model, features, label)
