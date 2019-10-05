
#### Import of the libraries ####

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import webbrowser, os
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

#### Basic settings ####

np.random.seed(2)
sns.set(style='white', context='notebook', palette='deep')
os.chdir("C:\\Users\\Marek\\Desktop\\Python\\Kaggle\\Datasets\\DigitRecognizer")

#### Related kernel ####

webbrowser.open_new_tab("https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6")

##### Data preparation and NULL check ####

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1) 

del train 

g = sns.countplot(Y_train)
Y_train.value_counts()
X_train.isnull().any().describe()

#### Normalization of the data for faster CNN convergence ####

X_train = X_train / 255.0
test = test / 255.0

#### Reshaping the 'images' ####

X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

#### Encoding y_labels to vectors ####

Y_train = to_categorical(Y_train, num_classes = 10)

#### Splitting data to the training/test set ####

random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

#### Example image ####

g = plt.imshow(X_train[70][:,:,0])

#### ANN Definition ####

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

#### Define optimizer, metrics and loss function ####

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

#### Define Learning Rate ####

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

epochs = 1 
batch_size = 86

#### Data Augmentation