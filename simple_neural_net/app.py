from src.create_layers import create_layers, get_data
from src.linear_model import train_model, create_model, create_normalized_label
from src.neural_network import get_outputs_dnn
from src.plot_data import plot_the_loss_curve
import numpy as np

train_df, test_df = get_data()
preprocessing_layers, inputs = create_layers(train_df)

#linear_model(preprocessing_layers, train_df, test_df, inputs, 0.1, 15, 1000)

# The following variables are the hyperparameters.
learning_rate = 0.01
epochs = 30
batch_size = 1000

# Specify the label
label_name = "median_house_value"

# Split the original training set into a reduced training set and a
# validation set.
validation_split = 0.2

dnn_outputs = get_outputs_dnn(preprocessing_layers, 'elu', 10, 6, 1)

train_median_house_value_normalized, test_median_house_value_normalized = create_normalized_label(train_df, test_df)

# Establish the model's topography.
my_model = create_model(
    inputs,
    dnn_outputs,
    learning_rate)

# Train the model on the normalized training set. We're passing the entire
# normalized training set, but the model will only use the features
# defined in our inputs.
epochs, mse, history = train_model(train_median_house_value_normalized, my_model, train_df, epochs,
                                   batch_size, label_name, validation_split)
plot_the_loss_curve(epochs, mse, history["val_mean_squared_error"])

# After building a model against the training set, test that model
# against the test set.
test_features = {name:np.array(value) for name, value in test_df.items()}
test_label = test_median_house_value_normalized(np.array(test_features.pop(label_name))) # isolate the label
print("\n Evaluate the new model against the test set:")
my_model.evaluate(x = test_features, y = test_label, batch_size=batch_size, return_dict=True)
