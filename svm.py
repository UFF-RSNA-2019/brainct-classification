# Import BinaryRelevance from skmultilearn
from skmultilearn.problem_transform import BinaryRelevance

# Import SVC classifier from sklearn
from sklearn.svm import SVC

# Setup the classifier
classifier = BinaryRelevance(classifier=SVC(), require_dense=[False,True])

# Train
classifier.fit(X_train, y_train)

# Predict
y_pred = classifier.predict(X_test)