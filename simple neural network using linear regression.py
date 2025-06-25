import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential

x_train=np.array([[1], [5]])
y_train=np.array([[300], [500]])

set_w = np.array([[200]])
set_b = np.array([100])

model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=100)

new_data = np.array([[2], [3], [4]])  
predictions_new = model.predict(new_data)
print(f"New predictions: {predictions_new}")
