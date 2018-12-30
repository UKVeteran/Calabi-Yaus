#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

import keras
from keras.models import Sequential,load_model
from keras.layers import Dense,Activation,Dropout,Conv1D,Conv2D,MaxPooling1D,MaxPooling2D,Flatten
from keras.callbacks import EarlyStopping
from keras.regularizers import l1,l2
    
##Main##
#Import data
def train_test_split(test_size,data):
#test_size = fraction of total data to be test sample
    length_train = int(np.size(data,axis=0)-np.floor(test_size*np.size(data,axis=0)))
    train_data=data[0:length_train,:]
    test_data=data[length_train:,:]
    return np.array((train_data,test_data))

def gen_y(NumPs,h11):
#return 1 if NumPs = h11 (favourable), 0 if not
    temp = np.zeros(np.size(h11))
    for count in range(0,np.size(h11)):
        if NumPs[count] == h11[count]:
            temp[count] = 1
        else:
            temp[count] = 0
    return temp

data=pd.read_csv('C:/Users/Johar/Desktop/ML_CY/ML_exp/Favourable/data',sep=",",skiprows=[0],header=None)
data=np.asfarray(data,float)
Numps = data[:,0]
h11 = data[:,2]
temp =gen_y(Numps,h11)

inputs=data[:,3:]
np.random.seed(1)
np.random.shuffle(data)

train_data,test_data = train_test_split(0.2,data)

x_train,x_test = train_data[:,4:],test_data[:,4:]
y_train,y_test = gen_y(train_data[:,0],train_data[:,2]),gen_y(test_data[:,0],test_data[:,2])

# x_train = np.reshape(x_train,(np.size(x_train,axis=0),12,15,1))
# x_test = np.reshape(x_test,(np.size(x_test,axis=0),12,15,1))

#Create network architechture
network = Sequential()
network.add(Dense(985,input_dim=180))
network.add(Activation('relu'))
network.add(Dropout(0.46))
network.add(Dense(1))
network.add(Activation('sigmoid'))

val_acc_stop = EarlyStopping(monitor='val_acc',min_delta=0,patience=5,verbose=1,mode='auto')
val_loss_stop = EarlyStopping(monitor='val_loss',min_delta=0,patience=5,verbose=1,mode='auto')
network.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
history = network.fit(x_train,y_train,batch_size=32,epochs=1000,verbose=1,validation_data=(x_test,y_test),callbacks=[val_acc_stop,val_loss_stop])

train_predict = network.predict(x_train,verbose = 1)[:,0]
test_predict = network.predict(x_test,verbose = 1)[:,0]

from performance_metrics import acc_metrics,ROC,F,class_acc,MCC

metric_train, metric_test = acc_metrics(train_predict,y_train,'nn'),acc_metrics(test_predict,y_test,'nn')
ROC_train, ROC_test = ROC(train_predict,y_train,'nn'),ROC(test_predict,y_test,'nn')
F_train, F_test = F(train_predict,y_train,'nn'),F(test_predict,y_test,'nn')
acc_train, acc_test = class_acc(train_predict,y_train,'nn'),class_acc(test_predict,y_test,'nn')

print('tp:'+str(metric_train[0])+' tn:'+str(metric_train[1])+' fp:'+str(metric_train[2])+' fn:'+str(metric_train[3]))
print('tp:'+str(metric_test[0])+' tn:'+str(metric_test[1])+' fp:'+str(metric_test[2])+' fn:'+str(metric_test[3]))
print('train ROC: '+str(ROC_train))
print('test ROC: '+str(ROC_test))
print(F_train,F_test)
print(acc_train,acc_test)
print(MCC(test_predict,y_test,'nn'))
