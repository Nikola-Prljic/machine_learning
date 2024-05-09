import numpy as np
from prepare_data import setup_data, create_cross_layers
from cross_model import create_model, train_model
from plot_the_loss import plot_the_loss_curve

train_df, test_df = setup_data()
inputs, outputs = create_cross_layers(train_df, resolution_in_degrees=0.4)

# The following variables are the hyperparameters.
learning_rate = 0.04
epochs = 40
batch_size = 50
label_name = 'median_house_value'

# Build the model, this time passing in the feature_cross_feature_layer:
my_model = create_model(inputs, outputs, learning_rate)

# Train the model on the training set.
epochs, rmse = train_model(my_model, train_df, epochs, batch_size, label_name)

# Print out the model summary.
my_model.summary(expand_nested=True)

test_features = {name:np.array(value) for name, value in test_df.items()}
test_label = np.array(test_features.pop(label_name))
print("\n: Evaluate the new model against the test set:")
train_rmse = my_model.evaluate(x=test_features, y=test_label, batch_size=batch_size)

print("Difference between train and test loss:")
print(round(list(rmse)[-1] - train_rmse[1], 2))
plot_the_loss_curve(epochs, rmse)
