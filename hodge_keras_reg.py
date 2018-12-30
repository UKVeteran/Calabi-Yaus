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
network.add(Dense(876,input_dim=180))
network.add(Activation('relu'))
network.add(Dropout(0.2))
network.add(Dense(461))
network.add(Activation('relu'))
network.add(Dropout(0.2))
network.add(Dense(437))
network.add(Activation('relu'))
network.add(Dropout(0.2))
network.add(Dense(929))
network.add(Activation('relu'))
network.add(Dropout(0.2))
network.add(Dense(404))
network.add(Activation('relu'))
network.add(Dropout(0.2))
network.add(Dense(1))

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

data=pd.read_csv('C:/Users/Johar/Desktop/ML_CY/ML_exp/Hodge/data',sep=",",skiprows=[0],header=None)
data=np.asfarray(data,float)

np.random.seed(1)
np.random.shuffle(data)

train_data,test_data = train_test_split(0.2,data)

x_train,x_test = train_data[:,4:],test_data[:,4:]
n = np.max(data[:,2])
y_train,y_test = train_data[:,2]/n,test_data[:,2]/n

# x_train=np.reshape(x_train,(np.size(x_train,axis=0),12,15,1))
# x_test=np.reshape(x_test,(np.size(x_test,axis=0),12,15,1))

from keras import backend as K
def r2_met(y_true,y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

network.compile(loss='mean_absolute_error',optimizer='adam',metrics=[r2_met])
history = network.fit(x_train,y_train,batch_size=32,epochs=1000,verbose=1,validation_data=(x_test,y_test),callbacks=[val_loss_stop])
network.save('./network.h5')
# network.load_weights('./network.h5')

#rescale by n
train_predict = n*network.predict(x_train,verbose = 1)[:,0]
test_predict = n*network.predict(x_test,verbose = 1)[:,0]
y_train = n*y_train
y_test = n*y_test

from h11_freq_plot_functions import plot_h11_freq
plot_h11_freq(test_predict,y_test,0.2)

from performance_metrics import acc_metrics,rms_error,reg_acc,r2
print(y_train)
print(train_predict)
print(y_test)
print(test_predict)

print(rms_error(train_predict,y_train),rms_error(test_predict,y_test))
print(r2(train_predict,y_train),r2(test_predict,y_test))
print(reg_acc(train_predict,y_train,0.5),reg_acc(test_predict,y_test,0.5))
