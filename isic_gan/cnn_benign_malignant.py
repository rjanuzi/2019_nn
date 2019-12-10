import dataset as ds

from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

import logging
from time import time

from _telegram import send_simple_message

FORMAT = '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
logging.basicConfig(filename=r'cnn_benign_malignant.log', level=logging.INFO, format=FORMAT)

MODEL_BKP_NAME = 'benign_malignant_model.h5'
USE_EXISTING_MODEL = True

DEFAULT_WIDTH = 512
DEFAULT_LENGTH = 512
MAX_IMGS_TO_USE = 8192
EPOCHS = 200

def show_normalized_img(norm_img_array):
    im = Image.fromarray((norm_img_array*255.0).astype('uint8'))
    im.show()

def save_normalized_img(norm_img_array, filename):
    im = Image.fromarray((norm_img_array*255.0).astype('uint8'))
    im.save(filename)

def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(DEFAULT_WIDTH, DEFAULT_LENGTH, 3))) # RGB Imgs (w x h x 3)
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='softmax'))

    # model.summary()

    # Compile model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'mse'])

    return model

class Model_BKP(callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 5 == 0:
            logging.info('Updating model...')
            self.model.save('benign_malignant_model.h5')
        if epoch % 25 == 0:
            send_simple_message('Training reached epoch %d (%.2f %%)' % (epoch, (epoch/EPOCHS)*100.0))

# Prepare the train and test data
train_X, train_y, test_X, test_y = ds.prepare_data(train_percentage=0.8,
                                                    img_width=DEFAULT_WIDTH,
                                                    img_length=DEFAULT_LENGTH,
                                                    max_imgs_to_use=MAX_IMGS_TO_USE)

# Create/Load the model
if USE_EXISTING_MODEL:
    try:
        model = models.load_model(MODEL_BKP_NAME)
    except:
        logging.error('Error loading model... creating a fresh one')
        model = create_model()
else:
    model = create_model()

# Train
logging.info('Train data: %d\nTest data: %d' % (len(train_X), len(test_X)))

start_time = time()
history = model.fit(x=train_X,
                    y=train_y,
                    batch_size=32,
                    epochs=EPOCHS,
                    verbose=0,
                    validation_data=(test_X, test_y),
                    callbacks=[Model_BKP()])
run_seconds = (time()-start_time)

logging.info('%s minutes to train.' % int(run_seconds/60))
send_simple_message('Training ended. (%s min)' % int(run_seconds/60))
send_simple_message('Starting evaluation.')

# Evaluate on test dataset
test_loss, test_acc, test_mse = model.evaluate(test_X,  test_y, verbose=2)

logging.info('Test loss: %.4f\nTest Accuracy: %.4f\nTest MSE: %.4f' % (test_loss, test_acc, test_mse))
send_simple_message('Test loss: %.4f\nTest Accuracy: %.4f\nTest MSE: %.4f' % (test_loss, test_acc, test_mse))
