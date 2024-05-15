import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from src.neural_net_model import create_model, train_model, plot_curve
from matplotlib import pyplot as plt

(x_train, y_train),(x_test, y_test) = keras.datasets.mnist.load_data()

x_train_normalized = x_train / 255
x_test_normalized = x_test / 255

learning_rate = 0.003
epochs = 50
batch_size = 4000
validation_split = 0.2

my_model = create_model(learning_rate)

epochs, hist = train_model(my_model, x_train_normalized, y_train, 
                           epochs, batch_size, validation_split)

list_of_metrics_to_plot = ['accuracy']
plot_curve(epochs, hist, list_of_metrics_to_plot)

# Evaluate against the test set.
print("\n Evaluate the new model against the test set:")
my_model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)

#plt.imshow(x_test[0])

range = 60

results = my_model.predict(x_test)
results = [np.argmax(x) for x in results[:range]]
index = np.arange(0, range, 1)

plt.scatter(index, y_test[:range], color='red', label='wrong')
plt.scatter(index, results, color='#00ff01')

plt.yticks(np.arange(0, 10, 1), np.arange(1, 11, 1))
plt.xticks(np.arange(0, range + 1, 5))
plt.xlabel('index')
plt.ylabel('number')
plt.legend()
plt.grid()
plt.show()

my_model.save('saved_model/multi_class_model-0.1.keras')

#print(list(y_test))
