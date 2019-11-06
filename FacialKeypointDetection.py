### Facial KeyPoint Detection ###

### Import libraries ###

import numpy as np, matplotlib.pyplot as plt, pandas as pd
from time import sleep
import webbrowser, os

### Model libraries ###

from keras.layers import Conv2D,Dropout,Dense,Flatten
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.layers import (Activation, Convolution2D, MaxPooling2D, BatchNormalization, 
                          Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D)

### Related notebook of KaranJakhar ###

webbrowser.open_new_tab("https://www.kaggle.com/karanjakhar/facial-keypoint-detection")

### Change working directory ###

os.chdir("C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\Skrypty\\Facial-KD")

### Import data ###

train_data = pd.read_csv("training.csv")  
test_data = pd.read_csv("test.csv")
lookid_data = pd.read_csv("IdLookupTable.csv")

### Count null values ###

train_data.head().T ##!## Transpozycja
train_data.isnull().any().value_counts() ##!## Check if there are nulls in the dataset
train_data.fillna(method = 'ffill',inplace = True) ##!## fill values with last one available

### Convert numbers into images and show them ###
imag = []


for i in range(0,7049):
    img = train_data['Image'][i].split(' ') ##!!## split into vector-like object
    img = ['0' if x == '' else x for x in img] ##!!## set 0 for empty values
    imag.append(img) # append to a list

image_list = np.array(imag,dtype = 'float') ##!!## convert to np.array
X_train = image_list.reshape(-1,96,96,1) ##!!## reshape to image-like table

### Example image ###

plt.imshow(X_train[0].reshape(96,96),cmap='gray')
plt.show()

### Create training set and y ###

training = train_data.drop('Image',axis = 1)

y_train = []
for i in range(0,7049):
    y = training.iloc[i,:]
    y_train.append(y)

y_train = np.array(y_train,dtype = 'float')

### Setup Neural Network ###

model = Sequential([Flatten(input_shape=(96,96)),
                         Dense(128, activation="relu"),
                         Dropout(0.1),
                         Dense(64, activation="relu"),
                         Dense(30)
                         ])
    
### Create all layers ###
    
model = Sequential()

model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(96,96,1)))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
# model.add(BatchNormalization())
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(30))
model.summary()

model.compile(optimizer='adam', 
              loss='mean_squared_error',
              metrics=['mae'])

model.fit(X_train,y_train,epochs = 1,batch_size = 256,validation_split = 0.2) ### Was 50

### Build X_test set ###

timag = []

for i in range(0,1783):
    timg = test_data['Image'][i].split(' ')
    timg = ['0' if x == '' else x for x in timg]
    timag.append(timg)
    
timage_list = np.array(timag,dtype = 'float')
X_test = timage_list.reshape(-1,96,96,1) 

### Show test set image ###

plt.imshow(X_test[0].reshape(96,96),cmap = 'gray')
plt.show()

### Create predictions ###
pred = model.predict(X_test)

### Create lists for every single column ###

lookid_list = list(lookid_data['FeatureName'])
imageID = list(lookid_data['ImageId']-1)
pre_list = list(pred)

rowid = lookid_data['RowId']
rowid=list(rowid)

feature = []
for f in list(lookid_data['FeatureName']):
    feature.append(lookid_list.index(f))
    
preded = []
for x,y in zip(imageID,feature): ##!## using zip in for loop
    preded.append(pre_list[x][y])
    
rowid = pd.Series(rowid,name = 'RowId')
loc = pd.Series(preded,name = 'Location')
submission = pd.concat([rowid,loc],axis = 1)

### Create submission ###

submission.to_csv('face_key_detection_submission.csv',index = False)