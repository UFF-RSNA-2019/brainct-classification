from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC

# carrega os arquivos
from numpy import genfromtxt
x_samples = genfromtxt('my_file.csv', delimiter=',')

# configura o classificador
classifier = BinaryRelevance(classifier=SVC(), require_dense=[False,True])

# treina
classifier.fit(X_train, y_train)

# predict
y_pred = classifier.predict(X_test)