import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = np.array(
    list(map(lambda img: np.array(img).flatten(), x_train))
), np.array(list(map(lambda img: np.array(img).flatten(), x_test)))
x_train, x_test = x_train.astype("float32") / 255, x_test.astype("float32") / 255
y_train, y_test = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(
    y_test
)

print(x_train.shape)
print(y_train.shape)
