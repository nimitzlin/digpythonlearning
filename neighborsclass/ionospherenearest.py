import numpy as np
import csv
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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


avg_scores = []
all_scores = []
parameter_values = list(range(1, 21)) # Include 20
for n_neighbors in parameter_values:
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(estimator, x, y, scoring='accuracy')
    avg_scores.append(np.mean(scores))
    all_scores.append(scores)


plt.plot(parameter_values, avg_scores, '-o')
plt.savefig('images/plot1.png', format='png')


#make dirty data
X_broken = np.array(x)
X_broken[:,::2] /= 10
broken_scores = cross_val_score(estimator, X_broken, y, scoring='accuracy')
print("The 'broken' average accuracy for is{0:.1f}%".format(np.mean(broken_scores) * 100))

X_transformed = MinMaxScaler().fit_transform(X_broken)

estimator = KNeighborsClassifier()
transformed_scores = cross_val_score(estimator, X_transformed, y,scoring='accuracy')
print("The average accuracy for is {0:.1f}%".format(np.mean(transformed_scores) * 100))
