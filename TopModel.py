#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 15:14:32 2018

@author: shubhamsinha
"""

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint 
#Keras model can be sequential or graphical
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint ,ReduceLROnPlateau
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam
from keras.models import model_from_json
from sklearn.externals import joblib
scaler_filename = "scaler.sav"

sc = StandardScaler()

def train_top_model(epochs):
    
    callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1),
            ModelCheckpoint(filepath='top_models_weights.h5', 
                               verbose=1, save_best_only=True)

    ]
    batch_size=16
#    top_model_weights_path = 'top_models_weights.h5'
    train_data = np.load(open('bottleneck_features_train.npy','rb'))
    train_labels = np.load(open('labels_train.npy','rb'))
# 
    validation_data = np.load(open('bottleneck_features_validation.npy','rb'))
    validation_labels = np.load(open('labels_validation.npy','rb'))
    train_data = np.reshape(train_data,(train_data.shape[0],8*8*512))
    validation_data = np.reshape(validation_data,(validation_data.shape[0],8*8*512))
    train_data = sc.fit_transform(train_data)
    validation_data = sc.transform(validation_data)
    joblib.dump(sc, scaler_filename) 


    model = Sequential()

    model.add(Dense(1024, activation='relu', input_dim=8*8*512))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.summary()
    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    

    history=model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),
              callbacks=callbacks)

    
    # summarize history for accuracy
    plt.show()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc']) 
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.figure()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # serialize model to JSON
    model_json = model.to_json()
    with open("topmodel.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("topmodel.h5")
    print("Saved model to disk")



def main(): 
    epochs=200
    train_top_model(epochs)
    
    
if __name__ == '__main__':
    main()