#!/usr/bin/python2

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import random
from tensorflow import keras
import sys

SHUFFLE_BUFFER_SIZE = 10
BATCH_BUFFER_SIZE = 100

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


def LoadModels(_p, _i):
    train = []

    for f1 in os.listdir(str(_p)):
        f2 = Image.open(str(_p) + "/" + str(f1))
        f2 = f2.resize((128, 128))
        train.append(np.array(f2))
    
    print("Loaded " + str(len(train)) + " images")
    labels = [_i] * len(train)

    train_test =  np.divide(np.asarray(train[:50]), 255.0)
    labels_test = np.asarray(labels[:50])
    train_train =  np.divide(np.asarray(train[50:]), 255.0)
    labels_train = np.asarray(labels[50:])

    return labels_test, labels_train, train_test, train_train


good_test_out, good_train_out, good_test_in, good_train_in = LoadModels("data/g", 1.0)
bad_test_out, bad_train_out, bad_test_in, bad_train_in = LoadModels("data/b", 0.0)

print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
print(np.append(good_train_in, bad_train_in, axis=0).shape)
train = tf.data.Dataset.from_tensor_slices((np.append(good_train_in, bad_train_in), np.append(good_train_out, bad_train_out))).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_BUFFER_SIZE)
test = tf.data.Dataset.from_tensor_slices((np.append(good_test_in, bad_test_in), np.append(good_train_out, bad_train_out))).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_BUFFER_SIZE)

# print(test_in)
# sys.exit(0)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(128, 128)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
    ])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy'])

model.fit(train, epochs=10)

test_loss, test_acc = model.evaluate(test_out,  test_in, verbose=2)

print('\nTest accuracy:', test_acc)



