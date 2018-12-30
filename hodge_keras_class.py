#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import MaxPooling1D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.regularizers import l1
from keras import optimizers
from keras import regularizers
from keras.layers import GaussianNoise
from keras.layers import BatchNormalization

#Create network architechture
network = Sequential()
# Convolution layers
network.add(Conv2D(57,kernel_size=(3,3),input_shape=(12,15,1)))
network.add(Activation('relu'))
network.add(Conv2D(56,kernel_size=(3,3)))
network.add(Activation('relu'))
network.add(Conv2D(55,kernel_size=(3,3)))
network.add(Activation('relu'))
network.add(Conv2D(43,kernel_size=(3,3)))
network.add(Activation('relu'))
network.add(Flatten())
# Fully connected layers
network.add(Dense(169,input_dim=180))
network.add(Activation('relu'))
network.add(Dropout(0.5))
network.add(Dense(491))
network.add(Activation('relu'))
network.add(Dropout(0.5))
network.add(Dense(20))
network.add(Activation('sigmoid'))

# val_acc_stop = EarlyStopping(monitor='val_acc',min_delta=0,patience=5,verbose=1,mode='auto')
val_loss_stop = EarlyStopping(monitor='val_loss',min_delta=0,patience=20,verbose=1,mode='auto')
    
##Main##
#Import data
def train_test_split(test_size,data):
#test_size = fraction of total data to be test sample
    length_train = int(np.size(data,axis=0)-np.floor(test_size*np.size(data,axis=0)))
    train_data=data[0:length_train,:]
    test_data=data[length_train:,:]
    return np.array((train_data,test_data))

def gen_y(h11):
    y = np.zeros(((np.size(h11),20)))
    for count in range(0,np.size(h11,axis=0)):
        y[count,int(h11[count])] = 1
    return y

data=pd.read_csv('C:/Users/Johar/Desktop/ML_CY/ML_exp/Hodge/data',sep=",",skiprows=[0],header=None)
data=np.asfarray(data,float)

np.random.seed(1)
np.random.shuffle(data)

train_data,test_data = train_test_split(0.2,data)

x_train,x_test = train_data[:,4:],test_data[:,4:]
y_train,y_test = gen_y(train_data[:,2]),gen_y(test_data[:,2])

x_train=np.reshape(x_train,(np.size(x_train,axis=0),12,15,1))
x_test=np.reshape(x_test,(np.size(x_test,axis=0),12,15,1))

network.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history = network.fit(x_train,y_train,batch_size=32,epochs=1000,verbose=1,validation_data=(x_test,y_test),callbacks=[val_loss_stop])
network.save('./network.h5')
# network.load_weights('./network.h5')

def class_to_hodge(y):
    h11 = np.zeros(np.size(y,axis=0))
    for count in range(0,np.size(y,axis=0)):
        h11[count] = np.argmax(y[count,:])
    return h11

train_predict = class_to_hodge(network.predict(x_train,verbose = 1))
test_predict = class_to_hodge(network.predict(x_test,verbose = 1))
y_train,y_test = class_to_hodge(y_train),class_to_hodge(y_test)

from h11_freq_plot_functions import plot_h11_freq
plot_h11_freq(test_predict,y_test,0.2)

from performance_metrics import acc_metrics, ROC, F, reg_acc

train_acc,test_acc = reg_acc(train_predict,y_train,0.5),reg_acc(test_predict,y_test,0.5)

print(train_acc,test_acc)
