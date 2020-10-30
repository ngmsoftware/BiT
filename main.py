#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 13:43:00 2020

@author: ninguem
"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import PIL
import json
import os
import time
import hub
import cv2



def preprocess(I):
    
    minSide = np.min(I.size)
    
    width, height = I.size   # Get dimensions

    left = (width - minSide)/2
    top = (height - minSide)/2
    right = (width + minSide)/2
    bottom = (height + minSide)/2

    # Crop the center of the image
    I = I.crop((left, top, right, bottom)).resize((224,224))
    
    I = np.array(I)
    I = np.expand_dims(I, axis=0)
    I = tf.keras.applications.resnet_v2.preprocess_input(I)
    
    return I


def recognize(I, allClasses, model):
    
    a = time.time()
    with tf.device('cpu:0'):
        logits = model(I)  # Logits with shape [batch_size, 21843].
        p = tf.nn.sigmoid(logits).numpy()[0]
    print('%.4f s'%(time.time()-a))

    idxs = list(np.argpartition(p.reshape(len(p)), -N_labels)[-N_labels:])

    p = p[idxs]
    classes = [allClasses[i] for i in idxs]
    

    _p, _classes = zip(*sorted(zip(p, classes)))


    return _p, _classes



def stringfy(_p, _classes):
    strFeatures = ''

    for thisProb, thisClass in zip(_p, _classes):
        strFeatures = strFeatures + '%s : %.4f\n'%(thisClass[:-1], thisProb)
    
    return strFeatures


N_labels = 6

a = time.time()
with tf.device('cpu:0'):
    model = tf.saved_model.load('bit_m-r152x4_imagenet21k_classification_1')
    #model = tf.keras.models.load_model('bit_m-r152x4_imagenet21k_classification_1')
print('%.4f s'%(time.time()-a))


fid = open("classes.txt","rt")
allClasses = fid.readlines()

directory =  os.path.join('..','images')


# for filename in os.listdir(directory):
#     if filename.endswith(".jpg") or filename.endswith(".png"):


#         fileName = os.path.join(directory , filename)

#         _I = PIL.Image.open(fileName)

#         I = preprocess(_I)

#         _p, _classes = recognize(I, allClasses, model)

#         strFeatures = stringfy(_p, _classes)


#         plt.figure()

#         plt.subplot(1,2,1)
#         plt.imshow((1+I.reshape(224,224,3))/2)
        
#         plt.subplot(1,2,2)
#         plt.text(0.1,0.3, strFeatures )
#         plt.axis('off')
        
#         print( fileName )
#         print(strFeatures )
        
    
    

cap = cv2.VideoCapture(0)
while 1:
    ret, frame = cap.read()
    frame = frame[:,:,[2,1,0]]
    
    _I = PIL.Image.fromarray(frame)
    
    I = preprocess(_I)

       

    _p, _classes = recognize(I, allClasses, model)

    strFeatures = stringfy(_p, _classes)

 
    plt.subplot(1,2,1)
    plt.cla()
    plt.imshow((1.0+I).reshape((224,224,3))/2.0)
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.cla()
    plt.text(0.1,0.3, strFeatures )
    plt.axis('off')

    plt.pause(0.001)


