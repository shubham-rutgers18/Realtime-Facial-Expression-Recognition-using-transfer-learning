#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 22:39:48 2018

@author: shubhamsinha
"""
from keras import applications 
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
#from PIL import Image

train_data_dir = 'Training'
nb_train_samples = 29059
validation_data_dir = 'PublicTest'
nb_validation_samples = 3589
test_data_dir = 'PrivateTest'
nb_test_samples = 3589
batch_size = 16


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    

    generator = datagen.flow_from_directory(
        train_data_dir,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)


    train_features = np.zeros(shape=(int(nb_train_samples/batch_size)*batch_size, 8, 8, 512))
    train_labels = np.zeros(shape=(int(nb_train_samples/batch_size)*batch_size,7))
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = model.predict(inputs_batch)
        train_features[i * batch_size : (i + 1) * batch_size] = features_batch
        train_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if (i+1) * batch_size >= nb_train_samples:
            break
    np.save(open('bottleneck_features_train.npy', 'wb'),
            train_features)
    np.save(open('labels_train.npy', 'wb'),
            train_labels)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)


    validation_features = np.zeros(shape=(int(nb_validation_samples/batch_size)*batch_size, 8, 8, 512))
    validation_labels = np.zeros(shape=(int(nb_validation_samples/batch_size)*batch_size,7))
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = model.predict(inputs_batch)
        validation_features[i * batch_size : (i + 1) * batch_size] = features_batch
        validation_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if (i+1) * batch_size >= nb_validation_samples:
            break
    np.save(open('bottleneck_features_validation.npy', 'wb'),
            validation_features)
    np.save(open('labels_validation.npy', 'wb'),
            validation_labels)

    
def main(): 
    save_bottlebeck_features()
    
    
if __name__ == '__main__':
    main()
