import dataset as ds

from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, callbacks

import logging
from time import time

from _telegram import send_simple_message

import traceback
from datetime import datetime

FORMAT = '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
logging.basicConfig(filename=r'cnn_benign_malignant.log', level=logging.INFO, format=FORMAT)

TELEGRAM_ON = True

MODEL_BKP_NAME = 'benign_malignant_model.h5'
USE_EXISTING_MODEL = False

IMGS_WIDTH = 128
IMGS_LENGTH = 128
BATCH_SIZE = 32
MAX_IMGS_TO_USE = 16384
# MAX_IMGS_TO_USE = 64
EPOCHS = 500

def send_telegram(msg):
    if TELEGRAM_ON:
        send_simple_message(msg)

def show_normalized_img(norm_img_array):
    im = Image.fromarray((norm_img_array*255.0).astype('uint8'))
    im.show()

def save_normalized_img(norm_img_array, filename):
    im = Image.fromarray((norm_img_array*255.0).astype('uint8'))
    im.save(filename)

def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMGS_WIDTH, IMGS_LENGTH, 3))) # RGB Imgs (w x h x 3)
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(2, activation='softmax')) # Two classes

    model.summary()

    # Compile model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'mse'])

    return model

class Model_BKP(callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 25 == 0:
            logging.info('Updating model... epoch %d (%.2f %%)' % (epoch, (epoch/EPOCHS)*100.0))
            send_telegram('Training reached epoch %d (%.2f %%)' % (epoch, (epoch/EPOCHS)*100.0))

        if epoch % 50 == 0:
            logging.info('Saving current model.')
            self.model.save(MODEL_BKP_NAME)

# Starting...
# ===============================================================================================
try:
    log_dir = r'logs\cnn_benign_malignant\%s' % datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=2, write_images=True)

    send_telegram('Loading data.')
    # Prepare the train and test data
    train_X, train_y, test_X, test_y = ds.prepare_classification_data(train_percentage=0.8,
                                                        img_width=IMGS_WIDTH,
                                                        img_length=IMGS_LENGTH,
                                                        max_imgs_to_use=MAX_IMGS_TO_USE)
    send_telegram('Data loaded.')

    # Create/Load the model
    if USE_EXISTING_MODEL:
        try:
            model = models.load_model(MODEL_BKP_NAME)
        except:
            logging.error('Error loading model... creating a fresh one')
            model = create_model()
    else:
        model = create_model()

    send_telegram('Model created/loaded.')

    # Train
    logging.info('\nTrain data: %d\nTest data: %d' % (len(train_X), len(test_X)))
    send_telegram('Train data: %d\nTest data: %d' % (len(train_X), len(test_X)))

    start_time = time()
    history = model.fit(x=train_X,
                        y=train_y,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=0,
                        validation_data=(test_X, test_y),
                        callbacks=[Model_BKP(), tensorboard_callback])
    run_seconds = (time()-start_time)

    logging.info('%s minutes to train.' % int(run_seconds/60))
    send_telegram('Training ended. (%s min)' % int(run_seconds/60))
    send_telegram('Starting evaluation.')

    # Evaluate on test dataset
    test_loss, test_acc, test_mse = model.evaluate(test_X,  test_y, verbose=2)
    ys = model.predict(test_X)

    logging.info('\nTest loss: %.4f\nTest Accuracy: %.4f\nTest MSE: %.4f' % (test_loss, test_acc, test_mse))
    send_telegram('Test loss: %.4f\nTest Accuracy: %.4f\nTest MSE: %.4f' % (test_loss, test_acc, test_mse))
except:
    logging.error(traceback.format_exc())
    send_telegram('Some error occured')
    send_telegram('%s' % traceback.format_exc())
