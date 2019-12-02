import tensorflow as tf
from tensorflow import keras

model = keras.Sequential()
model.add(keras.layers.Dense(4, activation='tanh'))
model.add(keras.layers.Dense(1, activation='tanh'))

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mse',
              metrics=['accuracy'])

data = [[0,0], [0,1], [1,0], [1,1]]
labels = [[0], [0], [0], [1]]

model.fit(data, labels, epochs=200)

print(model.predict([[0,0]]))
print(model.predict([[0,1]]))
print(model.predict([[1,0]]))
print(model.predict([[1,1]]))
