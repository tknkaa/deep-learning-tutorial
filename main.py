import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = np.array(
    list(map(lambda img: np.array(img).flatten(), x_train))
), np.array(list(map(lambda img: np.array(img).flatten(), x_test)))
""" x_train, x_test = x_train.astype("float32") / 255, x_test.astype("float32") / 255 """
y_train, y_test = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(
    y_test
)

img = x_train[0].reshape(28, 28)
pil_img = Image.fromarray(np.uint8(img))
pil_img.save("out.png")
