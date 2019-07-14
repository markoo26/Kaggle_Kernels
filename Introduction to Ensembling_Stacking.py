###### INTRODUCTION TO ENSEMBLING/STACKING IN PYTHON ######


#####!# - c.d.
####!# - blok
###!# - nowe
##!# - komentarz
#!# - do poprawy / zrobienia

### Do sprawdzenia:
 
# Dlugosc imienia (apply)
# Funkcja get_title
# HeatMap
# Null Random List
# np.isnan
#funkcja qcut/cut, fillna, for dataset in full_data, .shape(),
# re.search
# dataset loc / iloc
# dataset.apply
# dataset.map
# macierz korelacji Pearsona z elementami wiekszymi > 0,5 
# plotly - ladne wykresy
# Podpinanie plotly
#!# Algorytm Extra Trees
#!# AdaBoost


###### Kernel z Kaggle #####
import webbrowser
webbrowser.open_new_tab('https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python')

####!# Import bibliotek ####!#

####!# Biblioteki podstawowe ####!# 

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import plotly.offline as py, plotly.graph_objs as go, plotly.tools as tls
from plotly.offline import plot
import os, re

os.chdir('C:\\Users\\Marek\\Desktop\\Python\\Kaggle\\Titanic')

####!# Biblioteki do modelowanie ####!#

import sklearn
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.svm import SVC
from sklearn.model_selection import KFold
import xgboost as xgb

####!# Other settings ####!#

%matplotlib inline
py.init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')

####!# Import of the data ####!#

train = pd.read_csv('titanic.csv')
test = pd.read_csv('test_titanic.csv')
PassengerID = test['PassengerId']
full_data = [train,test]

####!# Dodajemy dodatkowe zmienne - dlugosc imienia ####!# 

train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)

####!# Oraz zmienna 0-1 - czy ktos mial kabine na Titanicu ####!#

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

####!# Tworzenie zmiennej FamilySizae jako suma potomkow i przodkow ####!#

for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1    

####!# Dalsza obrobka danych ####!#

for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset['Sex'] = dataset['Sex'].fillna(0)
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
    train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

# Create a New feature CategoricalAge

for dataset in full_data:

    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    train['CategoricalAge'] = pd.cut(train['Age'], 5)

####!# Funkcja get_title do wyciagania tytulu danej osoby ####!#

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""

####!# Stworzenie nowej zmiennej Title za pomoca poprzedniej funkcji ####!#

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

####!# Pogrupowanie rzadkich tytulow do kategorii Rare ####!#

for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

####!# Mapowanie plci ####!#

    dataset['Sex'] = dataset['Sex'].map( {"female": 0, "male": 1} ).astype(int)

####!# Mapowanie tytulow ####!#

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)    

####!# Mapowanie portu startowego ####!#

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

####!# Mapowanie oplaty ####!#

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

####!# Mapowanie wieku ####!# 

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;

####!# Wybor zmiennych do modelu ####!#

    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
    train = train.drop(drop_elements, axis = 1)
    train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
    test  = test.drop(drop_elements, axis = 1)

####!# Heatmap z korelacjami Pearsona ####!#

    colormap = plt.cm.RdBu
    plt.figure(figsize=(14,12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(train.corr()[abs(train.corr())>0.4],linewidths=0.1,vmax=1.0, #!# usunalem astype(float)
                square=False, cmap=colormap, linecolor='white', annot=True)

####!# Wykresy pairplot ####!#
    
    g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked',
       u'FamilySize', u'Title']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
    g.set(xticklabels=[])    

####!# Parametry do dalszej analizy ####!#

    ntrain = train.shape[0]
    ntest = test.shape[0]
    SEED = 0 # for reproducibility
    NFOLDS = 5 # set folds for out-of-fold prediction
    kf = KFold(ntrain, random_state=SEED) ## pominiete: n_folds= NFOLDS,    

####!# Rozszerzenie klasyfikatorow z paczki sklearn ####!#

class SklearnHelper(object):

    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self,x,y):
        return self.clf.fit(x,y)

    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)    

####!# Rozszerzenie funkcjonalnosci klasyfikatora XGBoost ####!#

def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS-2, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)): 
        #print(str(i) + ';' + str(train_index) + ';' + str(test_index)) 
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        
        clf.train(x_tr, y_tr)
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

####!# Parametry klasyfikatora Random Forest ####!#

rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

####!# Parametry algorytmu Extra Trees ####!#

et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

####!# Parametry algorytmu AdaBoost ####!#

ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

####!# Parametry algorytmu Gradient Boosting ####!#

gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

####!# Parametry algorytmu Support Vector Classifier ####!#

svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }

####!# Modele startowe / przypisanie obiektów ####!#

rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

####!# Przygotowanie danych wejsciowych ####!#
y_train = train['Survived'].ravel()
train = train.drop(['Survived', 'CategoricalAge'], axis=1)
train['Sex'] = train['Sex'].map( {"female": 0, "male": 1} ).astype(int)
train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train.loc[ train['Fare'] <= 7.91, 'Fare'] = 0
train.loc[(train['Fare'] > 7.91) & (train['Fare'] <= 14.454), 'Fare'] = 1
train.loc[(train['Fare'] > 14.454) & (train['Fare'] <= 31), 'Fare'] = 2
train.loc[ train['Fare'] > 31, 'Fare'] = 3
train['Fare'] = train['Fare'].astype(int)
train.loc[ train['Age'] <= 16, 'Age'] = 0
train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1
train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2
train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3
train.loc[ train['Age'] > 64, 'Age'] = 4 
train['Title'] = train['Title'].map(title_mapping)
train['Title'] = train['Title'].fillna(0)    
x_train = train.values 
x_test = test.values

####!# Tworzenie prognoz ####!#

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

print("Training is complete")

rf_feature = rf.feature_importances(x_train,y_train)
et_feature = et.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
gb_feature = gb.feature_importances(x_train,y_train)

#
##
##train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
##test['Embarked'] = test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
#
rf_features = [0.10474135,  0.21837029,  0.04432652,  0.02249159,  0.05432591,  0.02854371
  ,0.07570305,  0.01088129 , 0.24247496,  0.13685733 , 0.06128402]
et_features = [ 0.12165657,  0.37098307  ,0.03129623 , 0.01591611 , 0.05525811 , 0.028157
  ,0.04589793 , 0.02030357 , 0.17289562 , 0.04853517,  0.08910063]
ada_features = [0.028 ,   0.008  ,      0.012   ,     0.05866667,   0.032 ,       0.008
  ,0.04666667 ,  0.     ,      0.05733333,   0.73866667,   0.01066667]
gb_features = [ 0.06796144 , 0.03889349 , 0.07237845 , 0.02628645 , 0.11194395,  0.04778854
  ,0.05965792 , 0.02774745,  0.07462718,  0.4593142 ,  0.01340093]

cols = train.columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
     'Random Forest feature importances': rf_features,
     'Extra Trees  feature importances': et_features,
      'AdaBoost feature importances': ada_features,
    'Gradient Boost feature importances': gb_features
    })

####!# Random Forest features chart ####!#

trace = go.Scatter(
    y = feature_dataframe['Random Forest feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['Random Forest feature importances'].values,
        colorscale='Hot',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Random Forest Feature Importance',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig1 = go.Figure(data=data, layout=layout)
plot(fig1)


####!# Extra Trees features chart ####!#

trace = go.Scatter(
    y = feature_dataframe['Extra Trees  feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['Extra Trees  feature importances'].values,
        colorscale='Earth',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Extra Trees Feature Importance',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig2 = go.Figure(data=data, layout=layout)
plot(fig2)

####!# AdaBoost features chart ####!#

trace = go.Scatter(
    y = feature_dataframe['AdaBoost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['AdaBoost feature importances'].values,
        colorscale='Jet',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'AdaBoost Feature Importance',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig3 = go.Figure(data=data, layout=layout)
plot(fig3)

####!# Gradient boosting features chart ####!#

trace = go.Scatter(
    y = feature_dataframe['Gradient Boost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['Gradient Boost feature importances'].values,
        colorscale='Blackbody',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Gradient Boosting Feature Importance',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig4 = go.Figure(data=data, layout=layout)
plot(fig4)

####!# Obliczanie srednich znaczen poszczegolnych zmiennych ####!#

feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) 
feature_dataframe = feature_dataframe.sort_values(by = 'mean', axis = 0, ascending = False)
feature_dataframe.head(3)

####!# Wykres srednich waznosci dla wszystkich modeli ####!#

y = feature_dataframe['mean'].values
x = feature_dataframe['features'].values
data = [go.Bar(
            x= x,
             y= y,
            width = 0.5,
            marker=dict(
               color = feature_dataframe['mean'].values,
            colorscale='Portland',
            showscale=True,
            reversescale = False
            ),
            opacity=0.6
        )]

layout= go.Layout(
    autosize= True,
    title= 'Barplots of Mean Feature Importance',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig5 = go.Figure(data=data, layout=layout)
plot(fig5)

####!# Wyciaganie bazowych predykcji w oparciu o startowe modele ####!#

base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel()
    })
base_predictions_train.head()
