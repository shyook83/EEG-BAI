#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 21:36:36 2020

@ Modified by Hoseok Choi
"""
"""
Created on Sun Aug 16 21:40:26 2020

@author: Soonhyun Yook
"""


import inspect, os, sys
import numpy as np

from keras.models import Sequential
import tensorflow
from keras.layers import TimeDistributed
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Activation, Dropout, Reshape
from keras.layers.wrappers import TimeDistributed
from keras.models import model_from_json
from keras.layers import LSTM, Dense, Activation, ThresholdedReLU, MaxPooling2D, Embedding, Dropout,Conv2D, Flatten
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd # for using pandas daraframe
import numpy as np # for som math operations
from sklearn.preprocessing import StandardScaler # for standardizing the Data
import math
import h5py as h5
import glob
#import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.utils import plot_model
from datetime import datetime


#################### VARIABLES ####################
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
model_id = 5    # -1: resnet2D,   0: resnet3D,   1:vgg3D,   2: vgg2D,   3: vgg163D,   4: densenet2d,   5: densenet3d

densenettype = -2    # 0: densenet3dcam
#densenettype = 3    # For DenseNet-169

datafilename = 'M24_CNN_ERSP2_4cls_3c_fold0_3dtvtdat.npz'
codepath = '/ifs/loni/faculty/hkim/soonhyun/EEG_BA/Brain_age/model/densnet/'


fold=2 #0-9
j = 0         # nth_fold 0-4
class_num = 1
#class_num = 1  # regression
lru, bs, opt = 0.5, 2, 'Adam'
fit_iter, fit_ep = 1, 600
###################################################

currentcode = inspect.getfile(inspect.currentframe())
print('\n [Info] Running code: ', currentcode)


if model_id == 4:
    if densenettype == -2:
        fnprefix = 'dense12710_2dCAM'
    elif densenettype == -1:
        fnprefix = 'denseshort2dCAM'
    elif densenettype == 0:
        fnprefix = 'densenet2dCAM'
    elif densenettype == 1:
        fnprefix = 'densenet2d121'
    elif densenettype == 1:
        fnprefix = 'densenet2d161'
    elif densenettype == 1:
        fnprefix = 'densenet2d169'
elif model_id == 5:
    if densenettype == -2:
        fnprefix = 'dense12710_3dCAM'
    elif densenettype == -1:
        fnprefix = 'denseshort3dCAM'
    elif densenettype == 0:
        fnprefix = 'densenet3dCAM'
    elif densenettype == 1:
        fnprefix = 'densenet3d121'
    elif densenettype == 1:
        fnprefix = 'densenet3d161'
    elif densenettype == 1:
        fnprefix = 'densenet3d169'


os.getcwd()
procpath2 = str(datetime.now().year)+'_i'+str(datetime.now().month)+'_'+str(datetime.now().day)+'_'+fnprefix+'_'+datafilename[0:16]+"C"+str(class_num)+str(j+1)
if not os.path.exists(procpath2):
    os.makedirs(procpath2)
os.chdir(procpath2)
procpath2 = os.getcwd()
print('\tProcessing path: ', procpath2)

os.chdir(codepath)

#%% Build Model
if model_id == 4:     #densenet2d
    modelname = 'densenet2d4CAMmodel'
    from densenet2d4CAM import build_densenet_forCAM
    if datafilename[12] == '_':
        model = build_densenet_forCAM((720, 12, 4, 1), class_num, densenettype)
    elif datafilename[12] == '1':
        model = build_densenet_forCAM((100, 28, 51, 1), class_num, densenettype)
    elif datafilename[12] == '2':
        model = build_densenet_forCAM((720, 12, 4, 1), class_num, densenettype)
elif model_id == 5:     #densenet3d
    modelname = 'densenet3d4CAMmodel'
    from densenet3d4CAM import build_densenet_forCAM
    if datafilename[12] == '_':
        model = build_densenet_forCAM((2000, 16, 7, 1), class_num, densenettype)
    elif datafilename[12] == '1':
        model = build_densenet_forCAM((100, 28, 51, 1), class_num, densenettype)
    elif datafilename[12] == '2':
        model = build_densenet_forCAM((2000, 16, 7, 1), class_num, densenettype)
    
finalconv_name = 'CAM_conv' # this is the last conv layer of the network


os.chdir(procpath2)

print('\n [Info] Model set: ', modelname)
model.summary()
with open(modelname+'.txt', 'w') as f2:
    model.summary(print_fn=lambda x: f2.write(x+'\n'))

#%%

print('\n [Info] Loading Data')
_in_dir = os.path.join('/ifs/loni/faculty/hkim/soonhyun/EEG_BA/Brain_age/data/scalogram/whole-10-3D_2070_healthy_v5', "*.h5")
files1 = []
files2 = []
count=0
#c3,c4,f3,f4,01,02
for file1 in glob.glob(_in_dir):
    if count%10==fold:
        files1.append(file1)
    else:
        files2.append(file1)
    count=count+1
data=[]
output=[]
data1=[]
output1=[]
#%%
for i in files1:
    f = h5.File(i, 'r')
    x=np.array(f['meta']['age']).reshape(1,1).tolist()
    scal=np.array(f['scalogram'])
    
    
    data.append(scal)
    output.append(x)
for i in files2:
    f = h5.File(i, 'r')
    x=np.array(f['meta']['age']).reshape(1,1).tolist()
    scal=np.array(f['scalogram'])
    data1.append(scal)
    output1.append(x)      

output=np.concatenate(output)
data=np.concatenate(data,axis=0)
x=np.vsplit(data, len(files1))
output1=np.concatenate(output1)
data1=np.concatenate(data1,axis=0)
x1=np.vsplit(data1, len(files2))

for n in range(int(len(x))):
    i=x[n]
    i=np.transpose(i,(3,0,1,2))
    x[n]=i

for n in range(int(len(x1))):
    i=x1[n]
    i=np.transpose(i,(3,0,1,2))
    x1[n]=i
#%%
y=x[:int(len(x))]
test_data=np.concatenate(y)
y1=x1[:int(len(x1))]
train_data=np.concatenate(y1)
#%%
train_data=train_data.reshape(int(train_data.shape[0]/2000),2000,train_data.shape[1],train_data.shape[2],train_data.shape[3])
test_data=test_data.reshape(int(test_data.shape[0]/2000),2000,test_data.shape[1],test_data.shape[2],test_data.shape[3])


#%%
X_test=test_data
X_train=train_data
y_train=output1
y_test=output

train_X,val_X,train_y,val_y=train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
#%% Training
learning_rate = 0.0001

if 'Adam' in opt:
    optm = Adam(lr = learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
elif 'SGD' in opt:
    optm = SGD(lr = learning_rate)
elif 'RMSprop' in opt:
    optm = RMSprop(lr = learning_rate, rho = 0.9, epsilon = 1e-08, decay = 0.0)
cvscores = []

print("\n [Info] Training Start!")

for i in range(fit_iter) :
    print("\t Validating setnum:", str(j+1), "-Training iter:",str(i+1))
    filepath_weights_best = './weights.best_c'+str(class_num)+'_lr_' \
        +str(lru)+'_bs_'+str(bs)+'_opt_'+opt+'_iter'+str(i+1)+'.h5'
    filepath_weights_best_past = './weights.best_c'+str(class_num)+'_lr_' \
        +str(lru)+'_bs_'+str(bs)+'_opt_'+opt+'_iter'+str(i)+'.h5'

    if i>0:
        model.load_weights(filepath_weights_best_past)
        learning_rate *= lru            
        print('\tcurrent leraning_rate : ', learning_rate)

    if 'Adam' in opt:
        optm = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    elif 'SGD' in opt:
        optm = SGD(lr=learning_rate)
    elif 'RMSprop' in opt:
        optm = RMSprop(lr = learning_rate, rho = 0.9, epsilon = 1e-08, decay = 0.0)
        
    if class_num == 1 :
        model.compile(loss = 'mae', optimizer = optm)
    else :
        model.compile(loss = 'categorical_crossentropy', optimizer = optm, metrics=['accuracy'])

    #histogram = LossHistory()
    checkpoint = ModelCheckpoint(filepath_weights_best, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=6, verbose=1, mode='auto')
    csv_logger = CSVLogger('training.log', separator=',', append=True)
    callbacks_list = [csv_logger, checkpoint]



    trained_model=model.fit(train_X, train_y, batch_size=bs, epochs=fit_ep, verbose=1, \
                            shuffle=True, callbacks=callbacks_list, validation_data=(val_X, val_y))


    # list all data in history
    print(trained_model.history.keys())

    
    model.load_weights(filepath_weights_best)
    y_pred = model.predict(X_test, batch_size=bs)
    y_pred = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1]))
    Y_pred = np.argmax(y_pred, axis=1)

 

scores = model.evaluate(X_test, y_test, verbose=1, batch_size=bs)


