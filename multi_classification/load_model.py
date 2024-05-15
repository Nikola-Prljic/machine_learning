import tensorflow as tf
import keras
import numpy as np
from matplotlib import pyplot as plt

i = 0
# index from the Y/x data array

(x_train, y_train),(x_test, y_test) = keras.datasets.mnist.load_data()

new_model: keras.Model = keras.models.load_model('./saved_model/multi_class_model-0.1.keras')

loss = new_model.evaluate(x=(x_test / 255), y=y_test, batch_size=4000)

new_model.summary()


results = new_model.predict(x_test)

predicted_number = np.argmax(results[i])

font = {'color':  '#ff7744',
        'weight': 'heavy',
        'size': 16,
        }

plt.imshow(x_test[i])
plt.title(label=f"NUMBER\nPredicted = {predicted_number} | test = {y_test[i]}", fontdict=font)
plt.show()

print('loss =', loss[1])
print()
