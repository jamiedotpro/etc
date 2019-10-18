#!/usr/bin/env python
# title           :train.py
# description     :to train the model
# author          :Deepak Birla
# date            :2018/10/30
# usage           :python train.py --options
# python_version  :3.5.4

from Network import Generator, Discriminator, complex_Generator
import Utils_model, Utils
from Utils_model import VGG_LOSS
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from tqdm import tqdm
import numpy as np
import os
import argparse

# To fix error Initializing libiomp5.dylib
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.random.seed(10)
# Better to use downscale factor as 4
downscale_factor = 2
# Remember to change image shape if you are having different size of images
image_shape = (128, 128, 3)


# Combined network
def get_gan_network(discriminator, shape, generator, optimizer, vgg_loss):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x, gan_output])
    gan.compile(loss=[vgg_loss, "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)

    return gan


# default values for all parameters are given, if want defferent values you can give via commandline
# for more info use $python train.py -h
def train(epochs, batch_size, input_dir, output_dir, model_save_dir, number_of_images, train_test_ratio, image_extension):
    # Loading images
    x_train_lr, x_train_hr, x_test_lr, x_test_hr = \
        Utils.load_training_data(input_dir, image_extension, image_shape, number_of_images, train_test_ratio)

    print('======= Loading VGG_loss ========')
    # Loading VGG loss
    loss = VGG_LOSS(image_shape)
    loss2 = VGG_LOSS(image_shape)
    print('====== VGG_LOSS =======', loss)

    batch_count = int(x_train_hr.shape[0] / batch_size)
    print('====== Batch_count =======', batch_count)

    shape = (image_shape[0] // downscale_factor, image_shape[1] // downscale_factor, image_shape[2])
    print('====== Shape =======', shape)

    # Generator description
    generator = Generator(shape).generator()
    complex_generator = complex_Generator(shape).generator()
    # Discriminator description
    discriminator = Discriminator(image_shape).discriminator()
    discriminator2 = Discriminator(image_shape).discriminator()

    optimizer = Utils_model.get_optimizer()

    generator.compile(loss=loss.vgg_loss, optimizer=optimizer)
    complex_generator.compile(loss=loss2.vgg_loss, optimizer=optimizer)

    discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)
    discriminator2.compile(loss="binary_crossentropy", optimizer=optimizer)

    gan = get_gan_network(discriminator, shape, generator, optimizer, loss.vgg_loss)
    complex_gan = get_gan_network(discriminator2, shape, complex_generator, optimizer, loss2.vgg_loss)

    loss_file = open(model_save_dir + 'losses.txt', 'w+')

    loss_file.close()

    for e in range(1, epochs + 1):
        print('-' * 15, 'Epoch %d' % e, '-' * 15)
        for _ in tqdm(range(batch_count)):
            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)

            image_batch_hr = x_train_hr[rand_nums]
            image_batch_lr = x_train_lr[rand_nums]
            generated_images_sr = generator.predict(image_batch_lr)
            generated_images_csr = complex_generator.predict(image_batch_lr)
            real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
            fake_data_Y = np.random.random_sample(batch_size) * 0.2

            discriminator.trainable = True
            discriminator2.trainable = True

            d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)
            d_loss_fake = discriminator.train_on_batch(generated_images_sr, fake_data_Y)
            discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            d_loss_creal = discriminator2.train_on_batch(image_batch_hr, real_data_Y)
            d_loss_cfake = discriminator2.train_on_batch(generated_images_csr, fake_data_Y)
            discriminator_c_loss = 0.5 * np.add(d_loss_cfake, d_loss_creal)
            ########
            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
            image_batch_hr = x_train_hr[rand_nums]
            image_batch_lr = x_train_lr[rand_nums]

            gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
            discriminator.trainable = False
            discriminator2.trainable = False
            gan_loss = gan.train_on_batch(image_batch_lr, [image_batch_hr, gan_Y])
            gan_c_loss = complex_gan.train_on_batch(image_batch_lr, [image_batch_hr, gan_Y])

        print("discriminator_loss : %f" % discriminator_loss)
        print("gan_loss :", gan_loss)
        print("gan_c_loss :", gan_c_loss)
        gan_loss = str(gan_loss)

        loss_file = open(model_save_dir + 'losses.txt', 'a')
        loss_file.write('epoch%d : gan_loss = %s ; discriminator_loss = %f\n' % (e, gan_loss, discriminator_loss))
        loss_file.close()

        if e % 1 == 0:
            Utils.plot_generated_images(output_dir, e, generator,complex_generator, x_test_hr, x_test_lr)
        if e % 50 == 0:
            generator.save(model_save_dir + 'gen_model%d.h5' % e)
            discriminator.save(model_save_dir + 'dis_model%d.h5' % e)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('-i', '--input_dir', action='store', dest='input_dir', default='./data_hr/',
#                         help='Path for input images')
#
#     parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', default='./output/',
#                         help='Path for Output images')
#
#     parser.add_argument('-m', '--model_save_dir', action='store', dest='model_save_dir', default='./model/',
#                         help='Path for model')
#
#     parser.add_argument('-b', '--batch_size', action='store', dest='batch_size', default=64,
#                         help='Batch Size', type=int)
#
#     parser.add_argument('-e', '--epochs', action='store', dest='epochs', default=1000,
#                         help='number of iteratios for trainig', type=int)
#
#     parser.add_argument('-n', '--number_of_images', action='store', dest='number_of_images', default=1000,
#                         help='Number of Images', type=int)
#
#     parser.add_argument('-r', '--train_test_ratio', action='store', dest='train_test_ratio', default=0.8,
#                         help='Ratio of train and test Images', type=float)
#
#     values = parser.parse_args()
#
#     train(values.epochs, values.batch_size, values.input_dir, values.output_dir, values.model_save_dir,
#           values.number_of_images, values.train_test_ratio)

# Parameter
param_epochs = 5#50000000
param_batch = 10
param_input_folder = './VN_dataset/'
param_out_folder = './output/'
param_model_out_folder = './model/'
param_number_images = 500
param_train_test_ratio = 0.8
param_image_extension = '.png'

train(param_epochs,
      param_batch,
      param_input_folder,
      param_out_folder,
      param_model_out_folder,
      param_number_images,
      param_train_test_ratio,
      param_image_extension)
