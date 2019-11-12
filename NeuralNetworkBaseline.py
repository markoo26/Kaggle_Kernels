

### Import libraries ###

import webbrowser, os, gc 
import pandas as pd, numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics, preprocessing
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from keras.layers import Dense, Input
from collections import Counter
from keras.layers import BatchNormalization ##!!## Sprawdzic co to?
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras import callbacks
from keras import backend as K
from keras.layers import Dropout

import warnings
warnings.filterwarnings("ignore")

### Related kernel ###

webbrowser.open_new_tab("https://www.kaggle.com/speedwagon/neural-network-baseline")

### Functions ###

def submit(predictions):
    submit = pd.read_csv('sample_submission.csv')
    submit["target"] = predictions
    submit.to_csv("submission.csv", index=False)

def fallback_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except:
        return 0.5

def auc(y_true, y_pred):
    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)

### Settings ###
    
os.chdir("C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\Skrypty\\Datasets\\instant-gratification")
NFOLDS = 5
RANDOM_STATE = 42


### Import datasets ###
df_tr = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

### Enlist numeric columns ###
numeric = [c for c in df_tr.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

### Transformation 

len_train = df_tr.shape[0]
df_test['target'] = -1
data = pd.concat([df_tr, df_test])
data['magic_count'] = data.groupby(['wheezy-copper-turtle-magic'])['id'].transform('count')
##!!## concat data with get_dummies
data = pd.concat([data, pd.get_dummies(data['wheezy-copper-turtle-magic'])], axis=1, sort=False) 

df_tr = data[:len_train]
df_test = data[len_train:]

### Stratified KFlod validation

folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=RANDOM_STATE)

### Clear memory 
gc.collect()


### Preparing data for the NN

y = df_tr.target
ids = df_tr.id.values
train = df_tr.drop(['id', 'target'], axis=1)
test_ids = df_test.id.values
test = df_test[train.columns]

oof_preds = np.zeros((len(train)))
test_preds = np.zeros((len(test)))

scl = preprocessing.StandardScaler()
scl.fit(pd.concat([train, test]))
train = scl.transform(train)
test = scl.transform(test)

### Training the NN ### 

for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
    print("Current Fold: {}".format(fold_))
    trn_x, trn_y = train[trn_, :], y.iloc[trn_]
    val_x, val_y = train[val_, :], y.iloc[val_]

    inp = Input(shape=(trn_x.shape[1],))
    x = Dense(2000, activation="relu")(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(1000, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(500, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(100, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation="sigmoid")(x)
    clf = Model(inputs=inp, outputs=out)
    clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc])

    es = callbacks.EarlyStopping(monitor='val_auc', min_delta=0.001, patience=10,
                                 verbose=1, mode='max', baseline=None, restore_best_weights=True) ##!!## Sprawdzic co to?

    rlr = callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5,
                                      patience=3, min_lr=1e-6, mode='max', verbose=1)

    clf.fit(trn_x, trn_y, validation_data=(val_x, val_y), callbacks=[es, rlr], epochs=1, batch_size=1024)##!!## Przywrocic stare epochs
    
    val_preds = clf.predict(val_x)
    test_fold_preds = clf.predict(test)
    
    print("AUC = {}".format(metrics.roc_auc_score(val_y, val_preds)))
    oof_preds[val_] = val_preds.ravel()
    test_preds += test_fold_preds.ravel() / NFOLDS
    
    K.clear_session()
    gc.collect()
    
submit(test_preds)