#!/usr/bin/env python
# title           :Network.py
# description     :Architecture file(Generator and Discriminator)
# author          :Deepak Birla
# date            :2018/10/30
# usage           :from Network import Generator, Discriminator
# python_version  :3.5.4

# Modules
from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten
from keras.layers import Input, Add, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import add
import tensorflow as tf


# Residual block
def res_block_gen(model, kernal_size, filters, strides):
    gen = model

    model = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    # Using Parametric ReLU
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
    model = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)

    model = add([gen, model])

    return model

def complex_gen(model, kernal_size, filters, strides):
    #0
    gen = model
    ##
    model = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = Activation("relu")(model)
    #1
    gen1 = model
    ##
    model = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = Activation("relu")(model)
    
    #2
    model = add([gen, model])
    gen2 = model
    ##
    model = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = Activation("relu")(model)
    #3
    model = add([gen1, model])
    gen3 = model
    ##
    model = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = Activation("relu")(model)
    #4
    model = add([gen, model])
    model = add([gen2, model])
    gen4 = model
    ##
    model = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = Activation("relu")(model)
    #5
    model = add([gen1, model])
    model = add([gen3, model])
    gen5 = model
    ##
    model = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = Activation("relu")(model)
    #6
    model = add([gen, model])
    model = add([gen2, model])
    model = add([gen4, model])
    ##
    model = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = Activation("relu")(model)
    #7
    model = add([gen1, model])
    model = add([gen3, model])
    model = add([gen5, model])
    ##
    model = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = Lambda(lambda model: model * 0.2)(model)
    model = add([gen, model])
    return model

def dense_block(model, kernal_size,filters,strides, scale=0.2): #ergan
    x_1 = model

    x = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding='same')(model)
    x = LeakyReLU(alpha=0.2)(x)
    x = x_2 = Add()([x_1, x])

    x = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = x_3 = Add()([x_1, x_2, x])

    x = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Add()([x_1, x_2, x_3, x])

    x = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Add()([x_1, x_2, x_3, x])

    x = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding='same')(x)
    x = Lambda(lambda x: x * scale)(x)
    x = Add()([x_1, x])

    return x



def up_sampling_block(model, kernal_size, filters, strides):
    # In place of Conv2D and UpSampling2D we can also use Conv2DTranspose (Both are used for Deconvolution)
    # Even we can have our own function for deconvolution (i.e one made in Utils.py)
    # model = Conv2DTranspose(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = UpSampling2D(size=2)(model)
    model = LeakyReLU(alpha=0.2)(model)

    return model


def discriminator_block(model, filters, kernel_size, strides):
    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = LeakyReLU(alpha=0.2)(model)

    return model


    # Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
class Generator(object):

    def __init__(self, noise_shape):

        self.noise_shape = noise_shape

    def generator(self):

        gen_input = Input(shape=self.noise_shape)

        model = Conv2D(filters=64, kernel_size=9, strides=1, padding="same")(gen_input)
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(
            model)

        gen_model = model

        # Using 16 Residual Blocks
        for index in range(16):
            # model = res_block_gen(model, 3, 64, 1)
            model = dense_block(model, 3, 64, 1)
            

        model = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(model)
        model = BatchNormalization(momentum=0.5)(model)
        model = add([gen_model, model])

        # Using 2 UpSampling Blocks
        # for index in range(1):
        #     model = up_sampling_block(model, 3, 256, 1)

        model = Conv2D(filters=3, kernel_size=9, strides=1, padding="same")(model)
        model = Activation('tanh')(model)

        generator_model = Model(inputs=gen_input, outputs=model)

        return generator_model

class complex_Generator(object):
    
    def __init__(self, noise_shape):

        self.noise_shape = noise_shape

    def generator(self):

        gen_input = Input(shape=self.noise_shape)

        model = Conv2D(filters=64, kernel_size=9, strides=1, padding="same")(gen_input)
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(
            model)

        gen_model = model

        # Using 16 Residual Blocks
        for index in range(16):
            model = complex_gen(model, 3, 64, 1)

        model = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(model)
        model = BatchNormalization(momentum=0.5)(model)
        model = add([gen_model, model])

        # Using 2 UpSampling Blocks
        # for index in range(1):
        #     model = up_sampling_block(model, 3, 256, 1)
            

        model = Conv2D(filters=3, kernel_size=9, strides=1, padding="same")(model)
        model = Conv2D(filters=3, kernel_size=9, strides=1, padding="same")(model)
        model = Activation('tanh')(model)

        generator_model = Model(inputs=gen_input, outputs=model)

        return generator_model


# Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
class Discriminator(object):

    def __init__(self, image_shape):
        self.image_shape = image_shape

    def discriminator(self):
        dis_input = Input(shape=self.image_shape)

        model = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(dis_input)
        model = LeakyReLU(alpha=0.2)(model)

        model = discriminator_block(model, 64, 3, 2)
        model = discriminator_block(model, 128, 3, 1)
        model = discriminator_block(model, 128, 3, 2)
        model = discriminator_block(model, 256, 3, 1)
        model = discriminator_block(model, 256, 3, 2)
        model = discriminator_block(model, 512, 3, 1)
        model = discriminator_block(model, 512, 3, 2)

        model = Flatten()(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)

        model = Dense(1)(model)
        model = Activation('sigmoid')(model)

        discriminator_model = Model(inputs=dis_input, outputs=model)

        return discriminator_model