from keras import layers

#https://keras.io/api/layers/activations/

def get_outputs_dnn(preprocessing_layers, activation='relu', first_layer_neurons=20, second_layer_neurons=12, third=3):
    # Create a Dense layer with 20 nodes.
    dense_output = layers.Dense(units=first_layer_neurons,
                              activation=activation,
                              name='hidden_dense_layer_1')(preprocessing_layers)
    # Create a Dense layer with 12 nodes.
    dense_output = layers.Dense(units=second_layer_neurons,
                              activation=activation,
                              name='hidden_dense_layer_2')(dense_output)
    # Create the Dense output layer.
    dense_output = layers.Dense(units=third,
                              name='dense_output')(dense_output)

    # Define an output dictionary we'll send to the model constructor.
    outputs = {
        'dense_output': dense_output
    }
    return outputs
