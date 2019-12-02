import tensorflow as tf
from tensorflow import keras

FULLY_CONNECTED_PATH = 'models/fully_connected.h5'

def compileFullyConnectedModel(model):
    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

def createFullyConnectedModel(height=450, width=600):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(height, width, 3)), # Input layer same size as images
        keras.layers.Dense(128, activation=tf.nn.relu), # Hidden layer 1
        keras.layers.Dense(128, activation=tf.nn.relu), # Hidden layer 2
        keras.layers.Dense(2, activation=tf.nn.softmax) # Output layer some size as classes
    ])

    compileFullyConnectedModel(model)

    return model

def restoreFullyConnectedModel(height, width):
    try:
        model = keras.models.load_model(FULLY_CONNECTED_PATH)
        compileFullyConnectedModel(model)
        return model
    except OSError:
        return createFullyConnectedModel()
