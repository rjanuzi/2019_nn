import tensorflow as tf
from tensorflow.keras import layers, models

import os
from time import time
from PIL import Image
import numpy as np

from dataset import  prepare_gan_data
from _telegram import send_simple_message, send_img

import logging
import traceback

FORMAT = '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
logging.basicConfig(filename=r'skin_cancer_gan.log', level=logging.INFO, format=FORMAT)

TELEGRAM_ON = True

GENERATOR_MODEL_BKP_NAME = 'gan_generator_model.h5'
DISCRIMINATOR_MODEL_BKP_NAME = 'gan_discriminator_model.h5'
USE_EXISTING_MODEL = False

IMGS_SIZE = 256
IMGS_TO_USE = 4096
BATCH_SIZE = 256
NOISE_DIM = 100
EPOCHS = 500
EXAMPLES_TO_GENERATE = 5

def send_telegram(msg):
    if TELEGRAM_ON:
        try:
            send_simple_message(msg)
        except:
            logging.error('Error sending telegram msg: %s' % msg)

def send_telegram_img(img_path):
    if TELEGRAM_ON:
        try:
            send_img(img_path)
        except:
            logging.error('Error sending telegram img: %s' % img_path)

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(32*32*(256), use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((32, 32, 256)))
    assert model.output_shape == (None, 32, 32, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(256, (2, 2), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (10, 10), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 128, 128, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, IMGS_SIZE, IMGS_SIZE, 3)

    return model

def make_discriminator_model():
    model = models.Sequential()
    model.add(layers.Conv2D(128, (2, 2), strides=(1, 1), padding='same',
                                     input_shape=(IMGS_SIZE, IMGS_SIZE, 3)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, (10, 10), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

def show_generated_img(generated_data):
    img = Image.fromarray(np.uint8((generated_image[0]+128)*2))
    img.show()

def save_generated_img(generated_img, file_name):
    img = Image.fromarray(np.uint8((generated_img+128)*2))
    img.save(r'generated_imgs\%s.jpeg' % file_name)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = np.random.randn(BATCH_SIZE, NOISE_DIM)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

last_img_path = ''
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time()

        for image_batch in dataset:
            train_step(image_batch)

        epoch_id = (epoch+1)
        if epoch_id % 5 == 0:
            generator.save(GENERATOR_MODEL_BKP_NAME)
            discriminator.save(DISCRIMINATOR_MODEL_BKP_NAME)
            temp_images = generator(seeds, training=False)

            for seed_id in range(len(seeds)):
                save_generated_img(temp_images[seed_id], '%d_%d_generated' % (seed_id, epoch_id))
                last_img_path = r'generated_imgs\%d_%d_generated.jpeg' % (seed_id, epoch_id)

            send_telegram('Training reached epoch %d (%.2f %%)' % (epoch_id, (epoch_id/EPOCHS)*100.0))
            logging.info('Time for epoch %d is %.2f seconds.' % (epoch_id, time()-start))
            print('Time for epoch %d is %.2f seconds.' % (epoch_id, time()-start))

        if epoch_id % 15 == 0:
            send_telegram_img(last_img_path)

# Starting...
# ===============================================================================================
try:
    send_telegram('Preparing data...')
    train_images = prepare_gan_data(IMGS_SIZE, IMGS_SIZE, IMGS_TO_USE, benign_malignant=True)
    train_images = train_images.reshape(train_images.shape[0], IMGS_SIZE, IMGS_SIZE, 3).astype('float32')
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(IMGS_TO_USE).batch(BATCH_SIZE)
    seeds = np.random.randn(EXAMPLES_TO_GENERATE, NOISE_DIM)

    send_telegram('Creating models')
    if USE_EXISTING_MODEL:
        try:
            generator = models.load_model(GENERATOR_MODEL_BKP_NAME)
            discriminator = models.load_mode(DISCRIMINATOR_MODEL_BKP_NAME)
        except:
            logging.error('Error loading models... creating a fresh one')
            generator = make_generator_model()
            discriminator = make_discriminator_model()
    else:
        generator = make_generator_model()
        discriminator = make_discriminator_model()

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    send_telegram('Starting training')
    train(train_dataset, EPOCHS)

    send_telegram('Skin Cancer GAN training ended.')
except:
    logging.error(traceback.format_exc())
    send_telegram('Some error occured')
    send_telegram('%s' % traceback.format_exc())
