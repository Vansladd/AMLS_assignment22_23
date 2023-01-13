import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

def fix_gpu():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)


fix_gpu()
#Pre-procesing of dataset
#csv file was originally in one column, column was split into seperate columns with their respective column titles
#redundant columns where then dropped
img_files = os.listdir('dataset_AMLS_22-23\cartoon_set\img')
print(len(img_files))
df = pd.read_csv("dataset_AMLS_22-23\cartoon_set\labels.csv")
df.columns = ['a']
df[['index', 'eye_colour','face_shape','filename']] = df.a.str.split("\t", expand = True)
df= df.drop(['a','index'],axis=1)
#Image data Generator intialised also normalises the data from the images
datagen = ImageDataGenerator(rescale=1./255,validation_split=0.20)

# create train data generator
train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory='datasets\cartoon_set\img',
    x_col='filename',
    y_col='face_shape',
    target_size=(224, 224),
    batch_size=32,
    subset="training",
    class_mode='categorical')
#create validation data generator
validation_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory='datasets\cartoon_set\img',
    x_col='filename',
    y_col='face_shape',
    target_size=(224, 224),
    batch_size=32,
    subset="validation",
    class_mode='categorical')

# preprocessing of test data set
img_files = os.listdir('datasets\cartoon_set_test\img')
df = pd.read_csv("datasets\cartoon_set_test\labels.csv")
df.columns = ['a']
df[['index', 'eye_colour','face_shape','filename']] = df.a.str.split("\t", expand = True)
df= df.drop(['a','index'],axis=1)
datagen = ImageDataGenerator(rescale=1./255)

#create test data generator
test_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory='datasets\cartoon_set_test\img',
    x_col='filename',
    y_col='face_shape',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

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
model.add(Dense(5))
model.add(Activation('softmax'))

# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

# set up callbacks
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
checkpointer = ModelCheckpoint('modelA2  .h5', monitor='val_loss', verbose=1, save_best_only=True)

# train the model
B1 = model.fit_generator(train_generator,
                    steps_per_epoch=train_generator.n // train_generator.batch_size,
                    epochs=8 ,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.n // validation_generator.batch_size,
                    callbacks=[earlystopper, checkpointer])
acc = B1.history['accuracy']
val_acc = B1.history['val_accuracy']
loss = B1.history['loss']
val_loss = B1.history['val_loss']

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
