# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 13:13:16 2018

@author: Hp
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import random
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from scipy.misc import imread, imsave, imresize, imshow


from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json


#%%

nb_classes = 10
 
def load_dataset():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
 
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
 
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
 
    return X_train, Y_train, X_test, Y_test

X_train, y_train, X_test, y_test = load_dataset()

#%%

def create_network(channels, image_rows, image_cols, lr, decay, momentum):
    model = Sequential()
 
    model.add(Conv2D(32, (3, 3), padding='valid', input_shape = (image_rows, image_cols, channels)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    model.summary()
    
    return model

model = create_network(3, 32, 32, 0.01, 1e-6, 0.9)
#%%

earlystop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
checkpoint = ModelCheckpoint(filepath = 'best_model_0718.hdf5', verbose = 1, save_best_only = True)
    
model.fit(X_train, y_train, nb_epoch=30, batch_size=128, 
            validation_split=0.1, verbose=1, callbacks=[checkpoint, earlystop])  
 


   
json_model = model.to_json()
with open("model_final.json", "w") as json_file:
    json_file.write(json_model)
    
model.save_weights('model_weights_final.h5')  

#%%

#loading model from .h5    
json_file = open('model_final.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.load_weights("model_weights_final.h5")  

#%%
test_loss = model.evaluate(X_test, y_test, batch_size = 64, verbose=1)
print('Test Loss:', test_loss)

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Y_test = np.dot(y_test, classes).astype(int)

y_predict1 = model.predict_classes(X_test, verbose = 1)
test_accuracy1 = accuracy_score(Y_test, y_predict1)
print('Test Accuracy:', test_accuracy1)


#%%   
def predict_test(test_file, model):
    class_info = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
    image_data = imread(test_file)
    image_resized = imresize(image_data, (32, 32))
    image_arr = np.asarray (image_resized.transpose(0,1,2), dtype = 'float32')
    dataset = np.ndarray((1, 32, 32, 3), dtype = np.float32)
    dataset [0, :, :, :] = image_arr
    dataset /= 255
    pred_pr = model.predict_proba(dataset, verbose = 0)
    pred_cl = model.predict_classes(dataset, verbose = 0)[0]
   
    print('Predicted Class:', class_info[pred_cl])
    print('Predicted Probabilities:')
    prob_list =[]
    for i in range(0,10):
        print(class_info[i],':', pred_pr[0][i])
        prob_list.insert(i, pred_pr[0][i]) 
    return class_info, prob_list

def plot_bar_x(class_info, prob_list):
    # this is for plotting purpose
    index = np.arange(len(class_info))
    plt.bar(index, prob_list)
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Probablities', fontsize=12)
    plt.xticks(index, class_info, fontsize=11, rotation=30)
    plt.title('Probablities for each classes')
    plt.show()
        
    
class_info, prob_list = predict_test('C:/Users/Hp/.spyder-py3/Deep_Learning_cifar-10/test_images/airplane.jpg', model) 
#print(class_info)
#print(prob_list)   
plot_bar_x(class_info, prob_list)

test_image = imread('test_images/automobile.jpg')
plt.imshow(test_image)







