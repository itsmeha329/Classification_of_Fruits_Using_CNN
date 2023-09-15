#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, PIL
from glob import glob
import tensorflow as tf
from io import StringIO 
from PIL import Image
import pydot
import imageio as iio
#import cv2


#import seaborn as sns
from sklearn import model_selection


import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import  img_to_array
from tensorflow.keras.preprocessing.image import array_to_img


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# In[2]:


""" Sequential Model Architecture """
Sequential = tf.keras.models.Sequential

""" Data Preprocessing Functions """
Resizing = tf.keras.layers.experimental.preprocessing.Resizing
Rescaling = tf.keras.layers.experimental.preprocessing.Rescaling

""" Data Augmentation Functions """
RandomFlip = tf.keras.layers.experimental.preprocessing.RandomFlip
RandomRotation = tf.keras.layers.experimental.preprocessing.RandomRotation
RandomZoom = tf.keras.layers.experimental.preprocessing.RandomZoom

""" Artificial Neural Network Layer Inventory """
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout

""" Convolutional Neural Network Layer Inventory """
Conv2D = tf.keras.layers.Conv2D
MaxPool2D = tf.keras.layers.MaxPool2D
Flatten = tf.keras.layers.Flatten

""" Residual Network Layer Inventory """
ResNet50 = tf.keras.applications.resnet50.ResNet50

""" Function to Load Images from Target Folder """
get_image_data = tf.keras.utils.image_dataset_from_directory


# In[3]:


path ='Fruits_360' 


# In[4]:


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

img_width, img_height = 100, 100


# In[5]:


if K.image_data_format() == 'channels_first':
	input_shape = (3, img_width, img_height)
else:
	input_shape = (img_width, img_height, 3)


# In[6]:


train_data_dir ='Training'
validation_data_dir ='Test'
nb_train_samples =400
nb_validation_samples = 100
epochs = 10
batch_size = 25


# In[7]:


model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width,img_height,3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(131,activation = 'softmax'))


# In[8]:


model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			optimizer='Adam',
			metrics=['accuracy'])


# In[9]:


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # warning disabling

training_dataset = get_image_data(
    directory=train_data_dir,
    seed=42,
    image_size=(img_width, img_height),
    batch_size=batch_size
)

validation_dataset = get_image_data(
    directory=validation_data_dir,
    seed=42,
    image_size=(img_width, img_height),
    batch_size=batch_size
)
class_names=training_dataset.class_names
print(class_names)


# In[10]:


def configure_performant_datasets(dataset, shuffling=None):
    """ Custom function to prefetch and cache stored elements 
    of retrieved image data to boost latency and performance 
    at the cost of higher memory usage. """    
    AUTOTUNE = tf.data.AUTOTUNE
    # Cache and prefetch elements of input data for boosted performance
    if not shuffling:
        return dataset.cache().prefetch(buffer_size=AUTOTUNE)
    else:
        return dataset.cache().shuffle(shuffling).prefetch(buffer_size=AUTOTUNE)


# In[11]:


training_dataset = configure_performant_datasets(training_dataset, 
                                                 shuffling=1000)
validation_dataset = configure_performant_datasets(validation_dataset)


# In[12]:


resizing_layer = layers.experimental.preprocessing.Resizing(img_width, img_height)
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255, 
                                                                  input_shape=(img_width, img_height, 
                                                                               3))


# In[13]:


model.summary()


# In[ ]:


epochs = 100
history = model.fit(training_dataset, validation_data = validation_dataset, epochs=epochs, validation_steps = 800 // batch_size)


# In[14]:


batch_size = 25
plt.figure(figsize=(10, 10))
for images, labels in training_dataset.take(1):
    for i in range(batch_size):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")


# In[ ]:


batch_size = 25
plt.figure(figsize=(10, 10))
for images, labels in training_dataset.take(1):
    for i in range(batch_size):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")


# In[15]:


def plot_training_results(history):
    """
    Visualize results of the model training using `matplotlib`.

    The visualization will include charts for accuracy and loss, 
    on the training and as well as validation data sets.

    INPUTS:
        history(tf.keras.callbacks.History): 
            Contains data on how the model metrics changed 
            over the course of training.
    
    OUTPUTS: 
        None.
    """
    accuracy = history.history['accuracy']
    accuracy
    validation_accuracy = history.history['val_accuracy']
    validation_accuracy

    loss = history.history['loss']
    loss
    validation_loss = history.history['val_loss']
    validation_loss

    epochs_range = range(epochs)
    epochs_range

    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accuracy, label='Training Accuracy')
    plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, validation_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


# In[ ]:


plot_training_results(history)

