#!/usr/bin/python2

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import os
from PIL import Image

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential

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

    train_test =   np.expand_dims(np.divide(np.asarray(train[:50]), 255.0), axis=3)
    labels_test =  np.asarray(labels[:50])
    train_train =  np.expand_dims(np.divide(np.asarray(train[50:]), 255.0), axis=3)
    labels_train = np.asarray(labels[50:])

    return labels_test, labels_train, train_test, train_train


good_test_out, good_train_out, good_test_in, good_train_in = LoadModels("data/g", 0.9999)
bad_test_out, bad_train_out, bad_test_in, bad_train_in = LoadModels("data/b", 0.0)

print(np.append(good_test_out, bad_test_out, axis=0))
print(str(len(np.append(good_train_in, bad_train_in, axis=0))) + " / " + str(len(np.append(good_train_out, bad_train_out, axis=0))))
train = tf.data.Dataset.from_tensor_slices((np.append(good_train_in, bad_train_in, axis=0), np.append(good_train_out, bad_train_out, axis=0))).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_BUFFER_SIZE)
test = tf.data.Dataset.from_tensor_slices((np.append(good_test_in, bad_test_in, axis=0), np.append(good_test_out, bad_test_out, axis=0))).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_BUFFER_SIZE)

# print(test_in)
# sys.exit(0)

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


model.compile(optimizer=tf.keras.optimizers.RMSprop(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy'])

save_callback = tf.keras.callbacks.ModelCheckpoint(filepath="model/",
                                                   save_weights_only=True,
                                                   verbose=1)


for i in range(1):
    model.fit(train, epochs=2, callbacks=[save_callback])

test_loss, test_acc = model.evaluate(test, verbose=0)

print('\nTest accuracy:', test_acc)



