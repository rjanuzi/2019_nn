# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import dataset
import preprocessing
import imageutil
import models

checkpoint_path = "test_1_checkpoints/cp.ckpt"

def trainModel(trainIns, trainOuts, save=True, imgLimit=None):

    if imgLimit:
        trainIns = trainIns[:imgLimit]
        trainOuts = trainOuts[:imgLimit]

    # train_imgs = imageutil.loadJpegImgs(trainIns)
    train_imgs, height, width = imageutil.loadPreparedImgsData(trainIns)
    train_imgs = np.asarray(train_imgs)
    train_outputs = np.asarray(trainOuts)

    model = models.createFullyConnectedModel(height, width)

    # Create checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1, period=5)

    # Train
    model.fit(train_imgs, train_outputs, epochs=10, callbacks = [cp_callback])

    # Saving model
    if save:
        model.save(models.FULLY_CONNECTED_PATH)

    return model

def testModel(testIns, testOuts, model=None, imgLimit=None):

    if imgLimit:
        testIns = testIns[:imgLimit]
        testOuts = testOuts[:imgLimit]

    # test_imgs = imageutil.loadJpegImgs(testIns)
    test_imgs, height, width = imageutil.loadPreparedImgsData(testIns)
    test_imgs = np.asarray(test_imgs)
    test_outputs = np.asarray(testOuts)

    if not model:
        model=models.restoreFullyConnectedModel(height, width)

    loss, acc = model.evaluate(test_imgs, test_outputs)
    print("Model accuracy (Test data): {:5.2f}%".format(100*acc))

trainIns, trainOuts, labels = dataset.loadTrainData()
testIns, testOuts = dataset.loadTestData()

# trainModel(trainIns, trainOuts, imgLimit=2000)
# testModel(testIns, testOuts, model, imgLimit=1000)
