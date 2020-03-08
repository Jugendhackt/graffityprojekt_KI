#!/usr/bin/python2

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import os
from PIL import Image

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential

SHUFFLE_BUFFER_SIZE = 5
BATCH_BUFFER_SIZE = 100

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


def LoadModels(_p, _i):
    s = 0
    train = []

    for f1 in os.listdir(str(_p)):
        if s >= 1000:
            continue
        s += 1

        f2 = Image.open(str(_p) + "/" + str(f1))
        f2 = f2.resize((128, 128))
        train.append(np.array(f2))
    
    print("Loaded " + str(len(train)) + " images")
    labels = [_i] * len(train)

    numtrain = int(len(labels) * 0.8)

    train_test =   np.expand_dims(np.divide(np.asarray(train[numtrain:]), 255.0), axis=3)
    labels_test =  np.asarray(labels[numtrain:])
    train_train =  np.expand_dims(np.divide(np.asarray(train[:numtrain]), 255.0), axis=3)
    labels_train = np.asarray(labels[:numtrain])

    return labels_test, labels_train, train_test, train_train


good_test_out, good_train_out, good_test_in, good_train_in = LoadModels("data/g", 0.9999)
bad_test_out, bad_train_out, bad_test_in, bad_train_in = LoadModels("data/b", 0.0)

print("test_out" + str(good_test_out.shape) + ", train_out " + str(good_train_out.shape) + ", test_in " + str(good_test_in.shape) + ", train_in " + str(good_train_in.shape))

print(str(len(np.append(good_train_in, bad_train_in, axis=0))) + " / " + str(len(np.append(good_train_out, bad_train_out, axis=0))))
print(str(len(np.append(good_test_in, bad_test_in, axis=0))) + " / " + str(len(np.append(good_test_out, bad_test_out, axis=0))))
train = tf.data.Dataset.from_tensor_slices((np.append(good_train_in, bad_train_in, axis=0), np.append(good_train_out, bad_train_out, axis=0))).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_BUFFER_SIZE)
test = tf.data.Dataset.from_tensor_slices((np.append(good_test_in, bad_test_in, axis=0), np.append(good_test_out, bad_test_out, axis=0))).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_BUFFER_SIZE)


model = Sequential([
    Conv2D(16, 3, data_format='channels_last', padding='same', activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
    ])

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

save_callback = tf.keras.callbacks.ModelCheckpoint(filepath="model/",
                                                   save_weights_only=True,
                                                   verbose=1)


model.fit(train, epochs=5, callbacks=[save_callback])

test_loss, test_acc = model.evaluate(test, verbose=0)

print('\nTest accuracy:', test_acc)



