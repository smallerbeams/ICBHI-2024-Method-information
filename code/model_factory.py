import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, BatchNormalization,LeakyReLU,ReLU,AveragePooling1D

def AE_DN_smaple(input_layer, ker_num, ks, s, pool_small=True):

    conv1 =Conv1D(ker_num, kernel_size=ks, strides=s, padding='same' )(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = tf.keras.layers.LeakyReLU()(conv1)
    if pool_small:
        pool_1 = MaxPooling1D(pool_size=2, strides=2)(conv1)
    else:
        pool_1 = MaxPooling1D(pool_size=2, strides=1, padding='same' )(conv1)
    return pool_1

def AE_UP_smaple(input_layer, ker_num, ks, s, pool_pluse1=False):

    conv1 =Conv1D(ker_num, kernel_size=3, strides=1, padding='same' )(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = tf.keras.layers.LeakyReLU()(conv1)
    pool_conv1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv1)

    if pool_pluse1:
        gen1 = tf.keras.layers.Conv1DTranspose(ker_num, kernel_size=ks, strides=s, padding='valid')(pool_conv1)
    else:
        gen1 = tf.keras.layers.Conv1DTranspose(ker_num, kernel_size=ks, strides=s, padding='same')(pool_conv1)
    gen1 = BatchNormalization()(gen1)
    gen1 = tf.keras.layers.LeakyReLU()(gen1)
    pool_gen1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(gen1)

    return pool_gen1

def AE_CNN_246_shaopu(input_shape=(25,246),n_classes=3):

    input_layer = tf.keras.layers.Input(shape=input_shape)

####encode

    en_1 = AE_DN_smaple(input_layer,16,3,1)
    en_2 = AE_DN_smaple(en_1,32,3,1)
    en_3 = AE_DN_smaple(en_2,64,3,1,pool_small=False)

####decode
    de_1 = AE_UP_smaple(en_3,32,3,2)
    de_2 = AE_UP_smaple(de_1,16,3,2,pool_pluse1=True)

    xout =Conv1D(246 ,kernel_size=1, strides=1, padding='same' )(de_2)

    model = tf.keras.Model(inputs=input_layer, outputs=xout)
    return model



def AE_CNN_246_shaopu_thirdlayer_plus_Dense64_class(input_shape=(25,246),n_classes=3):

    input_layer = tf.keras.layers.Input(shape=input_shape)

####encode

    en_1 = AE_DN_smaple(input_layer,16,3,1)
    en_2 = AE_DN_smaple(en_1,32,3,1)
    en_3 = AE_DN_smaple(en_2,64,3,1,pool_small=False)

####decode
    x_flatten = Flatten()(en_3)

    x12 = tf.keras.layers.Dense(64, activation='relu')(x_flatten)
    x12 = tf.keras.layers.Dropout(0.5)(x12)


    xout = tf.keras.layers.Dense(n_classes, activation='softmax')(x12)


    model = tf.keras.Model(inputs=input_layer, outputs=xout)
    return model

#######################class########################
def AE_CNN_246_shaopu_seclayer_plus_Dense64_class(input_shape=(25,246),n_classes=3):

    input_layer = tf.keras.layers.Input(shape=input_shape)

####encode

    en_1 = AE_DN_smaple(input_layer,16,3,1)
    en_2 = AE_DN_smaple(en_1,32,3,1)

####decode
    x_flatten = Flatten()(en_2)

    x12 = tf.keras.layers.Dense(64, activation='relu')(x_flatten)
    x12 = tf.keras.layers.Dropout(0.5)(x12)


    xout = tf.keras.layers.Dense(n_classes, activation='softmax')(x12)


    model = tf.keras.Model(inputs=input_layer, outputs=xout)
    return model


def AE_CNN_246_shaopu_fristlayer_plus_Dense64_class(input_shape=(25,246),n_classes=3):

    input_layer = tf.keras.layers.Input(shape=input_shape)

####encode

    en_1 = AE_DN_smaple(input_layer,16,3,1)
####decode
    x_flatten = Flatten()(en_1)

    x12 = tf.keras.layers.Dense(64, activation='relu')(x_flatten)
    x12 = tf.keras.layers.Dropout(0.5)(x12)


    xout = tf.keras.layers.Dense(n_classes, activation='softmax')(x12)


    model = tf.keras.Model(inputs=input_layer, outputs=xout)
    return model

#####################level#########################
def AE_CNN_246_shaopu_fristlayer_plus_Dense128_level(input_shape=(25,246),n_classes=9):

    input_layer = tf.keras.layers.Input(shape=input_shape)

####encode

    en_1 = AE_DN_smaple(input_layer,16,3,1)
####decode
    x_flatten = Flatten()(en_1)

    x12 = tf.keras.layers.Dense(128, activation='relu')(x_flatten)
    x12 = tf.keras.layers.Dropout(0.5)(x12)


    xout = tf.keras.layers.Dense(n_classes, activation='softmax')(x12)


    model = tf.keras.Model(inputs=input_layer, outputs=xout)
    return model
def AE_CNN_246_shaopu_seclayer_plus_Dense128_level(input_shape=(25,246),n_classes=9):

    input_layer = tf.keras.layers.Input(shape=input_shape)

####encode

    en_1 = AE_DN_smaple(input_layer,16,3,1)
    en_2 = AE_DN_smaple(en_1,32,3,1)
####decode
    x_flatten = Flatten()(en_2)

    x12 = tf.keras.layers.Dense(128, activation='relu')(x_flatten)
    x12 = tf.keras.layers.Dropout(0.5)(x12)


    xout = tf.keras.layers.Dense(n_classes, activation='softmax')(x12)


    model = tf.keras.Model(inputs=input_layer, outputs=xout)
    return model
def AE_CNN_246_shaopu_thirdlayer_plus_Dense128_level(input_shape=(25,246),n_classes=9):

    input_layer = tf.keras.layers.Input(shape=input_shape)

####encode

    en_1 = AE_DN_smaple(input_layer,16,3,1)
    en_2 = AE_DN_smaple(en_1,32,3,1)
    en_3 = AE_DN_smaple(en_2,64,3,1,pool_small=False)
####decode
    x_flatten = Flatten()(en_3)

    x12 = tf.keras.layers.Dense(128, activation='relu')(x_flatten)
    x12 = tf.keras.layers.Dropout(0.5)(x12)


    xout = tf.keras.layers.Dense(n_classes, activation='softmax')(x12)


    model = tf.keras.Model(inputs=input_layer, outputs=xout)
    return model
