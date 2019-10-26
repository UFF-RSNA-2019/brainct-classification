from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy import genfromtxt
import configparser
from joblib import dump, load

configparser = configparser.ConfigParser()
configparser.read('config.ini')
config = configparser['Geral']

# carrega os arquivos
X_samples = genfromtxt('{}/stage1_x.csv'.format(config['FeaturesPath']), delimiter=',')
y_samples = genfromtxt('{}/stage1_y.csv'.format(config['FeaturesPath']), delimiter=',')

# split datasets
X_train, X_test, y_train, y_test = train_test_split(X_samples, y_samples, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

# configura o classificador
classifier = BinaryRelevance(classifier=SVC(), require_dense=[False,True])

# treina
clf = classifier.fit(X_train, y_train)

model_file = '{}/svm.joblib'.format(config['ModelsPath'])
dump(clf, model_file)

# predict
# y_pred = classifier.predict(X_test)
# y_pred = y_pred.toarray()
# acuracia
# print("Accuracy = {}".format(accuracy_score(y_test, y_pred)))

print("Done")