import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from tensorflow import keras 
from keras.preprocessing.image import load_img, img_to_array
import streamlit as st
import warnings 
warnings.filterwarnings('ignore')
from PIL import Image
import matplotlib.pyplot as plt
import cv2



def load_keras_model(model_path):
    return tf.keras.models.load_model(model_path)

model = load_keras_model('signlang_model_3.keras')


classes = ['A','B','C','D','E','F','G',
 'H',
 'I',
 'J',
 'K',
 'L',
 'M',
 'N',
 'O',
 'P',
 'Q',
 'R',
 'S',
 'T',
 'U',
 'V',
 'W',
 'X',
 'Y',
 'Z',
 'del',
 'nothing',
 'space']


cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    show_frame = cv2.resize(frame, (400, 400))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = cv2.resize(rgb_frame, (224, 224))
    

        
    prediction = model.predict(rgb_frame[np.newaxis, ...])
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    cv2.putText(show_frame, f"Sign: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(show_frame, f"Confidence: {confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('ASL Translator', show_frame)
    if cv2.waitKey(1) == ord('q'):
        break

        
    


