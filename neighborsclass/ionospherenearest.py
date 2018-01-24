import numpy as np
import csv
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score

x = np.zeros((351, 34), dtype='float')
y = np.zeros((351,), dtype='bool')


with open("ionosphere.data", 'r') as input_file:
    reader = csv.reader(input_file)
    
    for i, row in enumerate(reader):
        data = [float(datum) for datum in row[:-1]]
        x[i] = data
        y[i] = row[-1] == 'g'

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=14)
estimator = KNeighborsClassifier()
estimator.fit(X_train, y_train)

y_predicted = estimator.predict(X_test)
accuracy = np.mean(y_test == y_predicted) * 100
print "The accuracy is {0:.1f}%".format(accuracy)

scores = cross_val_score(estimator, x, y, scoring='accuracy')
average_accuracy = np.mean(scores) * 100
print "The average accuracy is {0:.1f}%".format(average_accuracy)
