#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.losses import categorical_crossentropy
from keras import backend as K
from keras.optimizers import Adam
from keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt

# dimensions of our images.
width, height = 182, 182

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 3000
nb_validation_samples = 1200
num_features = 64
epochs = 200
batch_size = 16
output_n = 10

if K.image_data_format() == 'channels_first':
    input_shape = (3, width, height)
else:
    input_shape = (width, height, 3)

#desinging the CNN
model = Sequential()

model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=input_shape, data_format='channels_last', kernel_regularizer=l2(0.01)))
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(output_n, activation='softmax'))
model.summary()
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='categorical',
    seed=123
)

print(train_generator)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='categorical',
    seed=123
)

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    shuffle=True
)

# fig = plt.figure()
# plt.plot(np.arange(0, epochs), H.history['loss'], label='training loss')
# plt.plot(np.arange(0, epochs), H.history['val_loss'], label='validation loss')
# plt.plot(np.arange(0, epochs), H.history['accuracy'], label='accuracy')
# plt.plot(np.arange(0, epochs), H.history['val_accuracy'], label='validation accuracy')
# plt.title('Accuracy and Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss|Accuracy')
# plt.show()
model.save('model.h5')
