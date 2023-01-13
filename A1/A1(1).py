import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
from scipy import misc
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc
from random import shuffle
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras.layers import BatchNormalization
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import imageio as iio

#Pre-procesing of dataset
#csv file was originally in one column, column was split into seperate columns with their respective column titles
#redundant columns where then dropped
img_files = os.listdir('dataset_AMLS_22-23\celeba\img')
print(len(img_files))
df = pd.read_csv("C:/Users/taiwo/OneDrive\Desktop/Machine Learning Project/dataset_AMLS_22-23/celeba/labels.csv")
df.columns = ['a']
df[['index', 'img_name','gender','smiling']] = df.a.str.split("\t", expand = True)
df= df.drop(['a','index'],axis=1)
#Image data Generator intialised also normalises the data from the images
datagen = ImageDataGenerator(rescale=1./255,validation_split=0.20)

# create train data generator
train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory='dataset_AMLS_22-23\celeba\img',
    x_col='img_name',
    y_col='gender',
    target_size=(224, 224),
    batch_size=32,
    subset='training',
    class_mode='binary')

#create validation data generator
validation_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory='dataset_AMLS_22-23\celeba\img',
    x_col='img_name',
    y_col='gender',
    target_size=(224, 224),
    batch_size=32,
    subset='validation',
    class_mode='binary')

# preprocessing of test data set
img_files = os.listdir('dataset_AMLS_22-23_test\celeba_test\img')
print(len(img_files))
df = pd.read_csv("dataset_AMLS_22-23_test\celeba_test\labels.csv")
df.columns = ['a']
df[['index', 'img_name','gender','smiling']] = df.a.str.split("\t", expand = True)
df= df.drop(['a','index'],axis=1)
datagen = ImageDataGenerator(rescale=1./255)

#create test data generator
test_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory='dataset_AMLS_22-23_test\celeba_test\img',
    x_col='img_name',
    y_col='gender',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

# create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# set up callbacks
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
checkpointer = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True)

# train the model
A1 = model.fit_generator(train_generator,
                    steps_per_epoch=train_generator.n // train_generator.batch_size,
                    epochs=8 ,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.n // validation_generator.batch_size,
                    callbacks=[earlystopper, checkpointer]) 
acc = A1.history['accuracy']
val_acc = A1.history['val_accuracy']
loss = A1.history['loss']
val_loss = A1.history['val_loss']

epochs = range(len(acc))
#Displays learning curve
plt.plot(epochs, acc, 'g', label='Training accuracy')
plt.plot(epochs, val_acc, 'y', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()

#Evaluates model with test data
score = model.evaluate(test_generator)
print('Test accuracy:', score[1])