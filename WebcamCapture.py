#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 15:14:32 2018

@author: shubhamsinha
"""
import cv2
import sys
from keras import applications 
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.externals import joblib
from keras.preprocessing.image import img_to_array
from keras.models import model_from_json

basemodel = applications.VGG16(include_top=False, weights='imagenet')
#classifier=joblib.load('./finalized_model.sav')
scaler_filename = "scaler.sav"
scaler = joblib.load(scaler_filename)

# load json and create model
json_file = open('topmodel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("topmodel.h5")
print("Loaded model from disk")
  

emotions =["anger","disgust","fear","happy","sad","surprise","neutral"]

cascPath = './haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        x, y, width, height = (x,y,w,h)
        x_off, y_off = (20,40)
        x1 , x2 , y1 , y2 = (x - x_off, x + w + x_off, y - y_off, y + h + y_off)
        if len(faces) == 1: 
                gray_face = gray[y1:y2, x1:x2]
#                print(x1 , x2 , y1 , y2)
                try:
                    gray_face = cv2.resize(gray_face, (256,256))
                    gray_face = gray_face.astype("float") / 255.0
                    gray_face = img_to_array(gray_face)
                    gray_face = cv2.cvtColor(gray_face,cv2.COLOR_GRAY2RGB)
                    gray_face = np.expand_dims(gray_face, axis=0)
#                    gray_face = np.expand_dims(gray_face, 0)
#                    gray_face = np.expand_dims(gray_face, -1)
                except:
                    continue
                features = basemodel.predict(gray_face)
                features = np.reshape(features,(1,8*8*512))
                features=scaler.transform(features)
                predicted=loaded_model.predict(features)
                print(predicted)
                label=np.argmax(predicted, axis=1)
#                predicted=classifier.predict(features)
#                text=emotions[predicted[0]]
                text=emotions[label[0]]
                print(text)
                cv2.putText(frame, text, (x , y),cv2.FONT_HERSHEY_SIMPLEX, 2, 2, cv2.LINE_AA)
                cv2.imshow('Video', frame)
#                return faceslice#slice face from image
           
        else:
                print("no/multiple faces detected, passing over frame")
   
    # Display the resulting frame
#    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()