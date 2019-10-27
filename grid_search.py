from skmultilearn.problem_transform import BinaryRelevance
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from numpy import genfromtxt
from joblib import dump
import numpy as np
import configparser
import lib

# carrega os arquivos
X_samples = genfromtxt('{}/stage1_x.csv'.format(lib.config['FeaturesPath']), delimiter=',')
y_samples = genfromtxt('{}/stage1_y.csv'.format(lib.config['FeaturesPath']), delimiter=',')

# split datasets 80/20
X_train, X_test, y_train, y_test = train_test_split(X_samples, y_samples, test_size=0.2, random_state=1)

# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

parameters = [
    {
        'classifier': [MultinomialNB()],
        'classifier__alpha': [0.7, 1.0],
    },
    {
        'classifier': [RandomForestClassifier()]
    },
    {
        'classifier': [MLPClassifier()]
    },
    {
        'classifier': [KNeighborsClassifier()]
    },
]

clf = GridSearchCV(BinaryRelevance(), parameters, scoring='accuracy')
clf.fit(X_samples, y_samples)
print (clf.best_params_, clf.best_score_)

model_file = '{}/grid_search.joblib'.format(lib.config['ModelsPath'])
dump(clf, model_file)


print ("Done")