# https://www.kaggle.com/tentotheminus9/what-causes-heart-disease-explaining-the-model
# cd "C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\Scripts"

### Krok 1 - Biblioteki

#os.environ[PATH] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


import os
import numpy as np
import pandas as pd

import plotly.plotly as py
import plotly.tools as plotly_tools
import plotly.graph_objs as go

from IPython.display import HTML

import matplotlib.pyplot as plt
import seaborn as sns #for plotting
from sklearn.ensemble import RandomForestClassifier #for the model
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz #plot tree
from sklearn.metrics import roc_curve, auc #for model evaluation
from sklearn.metrics import classification_report #for model evaluation
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.model_selection import train_test_split #for data splitting
import eli5 #for purmutation importance
from eli5.sklearn import PermutationImportance
import shap #for SHAP values 
from pdpbox import pdp, info_plots #for partial plots
import jupyter
from metakernel.display import display

os.chdir('C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\Heart Disease')
dataset = pd.read_csv('heart.csv')

### HARK - Hypothesize After Results are Known

dataset.head(10)
dataset.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']

### Zakodowanie przekodowanych danych

dataset['sex'][dataset['sex'] == 0] = 'female'
dataset['sex'][dataset['sex'] == 1] = 'male'

dataset['chest_pain_type'][dataset['chest_pain_type'] == 1] = 'typical angina'
dataset['chest_pain_type'][dataset['chest_pain_type'] == 2] = 'atypical angina'
dataset['chest_pain_type'][dataset['chest_pain_type'] == 3] = 'non-anginal pain'
dataset['chest_pain_type'][dataset['chest_pain_type'] == 4] = 'asymptomatic'

dataset['fasting_blood_sugar'][dataset['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
dataset['fasting_blood_sugar'][dataset['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'

dataset['rest_ecg'][dataset['rest_ecg'] == 0] = 'normal'
dataset['rest_ecg'][dataset['rest_ecg'] == 1] = 'ST-T wave abnormality'
dataset['rest_ecg'][dataset['rest_ecg'] == 2] = 'left ventricular hypertrophy'

dataset['exercise_induced_angina'][dataset['exercise_induced_angina'] == 0] = 'no'
dataset['exercise_induced_angina'][dataset['exercise_induced_angina'] == 1] = 'yes'

dataset['st_slope'][dataset['st_slope'] == 1] = 'upsloping'
dataset['st_slope'][dataset['st_slope'] == 2] = 'flat'
dataset['st_slope'][dataset['st_slope'] == 3] = 'downsloping'

dataset['thalassemia'][dataset['thalassemia'] == 1] = 'normal'
dataset['thalassemia'][dataset['thalassemia'] == 2] = 'fixed defect'
dataset['thalassemia'][dataset['thalassemia'] == 3] = 'reversable defect'

dataset.dtypes

### Konwersja typow danych

dataset['sex'] = dataset['sex'].astype('object')
dataset['chest_pain_type'] = dataset['chest_pain_type'].astype('object')
dataset['fasting_blood_sugar'] = dataset['fasting_blood_sugar'].astype('object')
dataset['rest_ecg'] = dataset['rest_ecg'].astype('object')
dataset['exercise_induced_angina'] = dataset['exercise_induced_angina'].astype('object')
dataset['st_slope'] = dataset['st_slope'].astype('object')
dataset['thalassemia'] = dataset['thalassemia'].astype('object')

### Tworzenie dummy variables

dataset = pd.get_dummies(dataset, drop_first=True) ### skoczylo z 14 do 20 kolumn

### Podzial na probe testowa i uczaca
list(dataset)

X_train, X_test, y_train, y_test = train_test_split(dataset.drop('target',1), dataset['target'], test_size = .2, random_state=0) #split the data

### Tworzenie modelu RandomForest

model = RandomForestClassifier(n_estimators = 500, max_depth=5)
model.fit(X_train, y_train)

### Wykres Decision Tree

estimator = model.estimators_[1]
feature_names = [i for i in X_train.columns]

y_train_str = y_train.astype('str')
y_train_str[y_train_str == '0'] = 'no disease'
y_train_str[y_train_str == '1'] = 'disease'
y_train_str = y_train_str.values

export_graphviz(estimator, out_file='tree.dot', 
                feature_names = feature_names,
                class_names = y_train_str,
                rounded = True, proportion = True, 
                label='root',
                precision = 2, filled = True)

#!# Sekcja kodu do poprawy

#from subprocess import call
#import subprocess as sp
#
#plik_dot = os.path.join(os.getcwd(), 'tree.dot')
#plik_png = os.path.join(os.getcwd(), 'tree.png')
#
#sp.Popen(['dot', '-Tpng', plik_dot, '-o', plik_png, '-Gdpi=600'])
#call('dot -Tpng tree.dot -o tree.png -Gdpi=600')

#from IPython.display import Image
#Image(filename = 'tree.png')

#!# CIAG DALSZY: export_graphviz

### Krok3 - prognozy i ich jakosc

y_predict = model.predict(X_test)
y_pred_quant = model.predict_proba(X_test)[:, 1]
y_pred_bin = model.predict(X_test)

# Confusion matrix

cm = confusion_matrix(list(y_test), list(y_pred_bin))
cm #!# Nie uzywac nazw zmiennych takich jak nazwy funkcji 

# Sensitivity i specificity

total=sum(sum(cm))

sensitivity = cm[0,0]/(cm[0,0]+cm[1,0])
print('Sensitivity : ', sensitivity )

specificity = cm[1,1]/(cm[1,1]+cm[0,1])
print('Specificity : ', specificity)

# Wykres ROC

fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

## AUC - Area Under Curve
auc(fpr, tpr)

#$ Nie dziala wyswietlanie HTML object wg Spydera ma byc w 2020 taka funkcja
#
#perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)
#eli5.show_weights(perm, feature_names = X_test.columns.tolist())

## Wykres Partial Dependence Plot
## Dla zmiennej num_major_vessels - im wieksze cisnienie, tym wiecej krwi i mniejsza szansa na choroby serca

base_features = dataset.columns.values.tolist()
base_features.remove('target')

feat_name = 'num_major_vessels'
pdp_dist = pdp.pdp_isolate(model=model, dataset=X_test, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()

### Wykres dla zmiennej wiek

feat_name = 'age'
pdp_dist = pdp.pdp_isolate(model=model, dataset=X_test, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()

### St_depression

feat_name = 'st_depression'
pdp_dist = pdp.pdp_isolate(model=model, dataset=X_test, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()

### Interakcja dwoch zmiennych

inter1  =  pdp.pdp_interact(model=model, dataset=X_test, model_features=base_features, features=['st_slope_upsloping', 'st_depression'])

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['st_slope_upsloping', 'st_depression'], plot_type='contour')
plt.show()

inter1  =  pdp.pdp_interact(model=model, dataset=X_test, model_features=base_features, features=['st_slope_flat', 'st_depression'])

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['st_slope_flat', 'st_depression'], plot_type='contour')
plt.show()

#Wykres Shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test, plot_type="bar")

# Wykres Shap2

shap.summary_plot(shap_values[1], X_test)

# Sprawdzenie jak rozne zmienne wplywaja na indywidualnych pacjentow
# Rowniez problem z HTML object

#def heart_disease_risk_factors(model, patient):
#
#    explainer = shap.TreeExplainer(model)
#    shap_values = explainer.shap_values(patient)
#    shap.initjs()
#    return shap.force_plot(explainer.expected_value[1], shap_values[1], patient)
#
#
#data_for_prediction = X_test.iloc[1,:].astype(float)
#heart_disease_risk_factors(model, data_for_prediction)

# Dependence Plot

ax2 = fig.add_subplot(224)
shap.dependence_plot('num_major_vessels', shap_values[1], X_test, interaction_index="st_depression")

# Force Plot

shap_values = explainer.shap_values(X_train.iloc[:50])
shap.force_plot(explainer.expected_value[1], shap_values[1], X_test.iloc[:50])
