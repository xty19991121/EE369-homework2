import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
import numpy as np

# download and read mnist
mnist = fetch_mldata('mnist-original', data_home='./')

# 'mnist.data' is 70k x 784 array, each row represents the pixels from a 28x28=784 image
# 'mnist.target' is 70k x 1 array, each row represents the target class of the corresponding image
images = mnist.data
targets = mnist.target

# make the value of pixels from [0, 255] to [0, 1] for further process
X = mnist.data / 255.
Y = mnist.target


from sklearn.model_selection import train_test_split
# split data to train and test (for faster calculation, just use 1/10 data)
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)

'''
#use logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
lr=LogisticRegression()
lr.fit(X_train, Y_train)
train_accuracy=lr.score(X_train,Y_train)
test_accuracy=lr.score(X_test,Y_test)
'''

'''
#use naive bayes
from sklearn.naive_bayes import BernoulliNB
bnb=BernoulliNB()
bnb.fit(X_train, Y_train)
train_accuracy=bnb.score(X_train, Y_train)
test_accuracy=bnb.score(X_test, Y_test)
'''


'''
#use support vector machine
from sklearn.svm import LinearSVC
lsvc=LinearSVC()
lsvc.fit(X_train, Y_train)
train_accuracy=lsvc.score(X_train, Y_train)
test_accuracy=lsvc.score(X_test, Y_test)
'''





# TODO:use SVM with another group of parameters
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

svc=LinearSVC()
parameters = [
    {
        'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 10, 100],
        'dual': [False, True],
    }
]
clf = GridSearchCV(svc, parameters)
clf.fit(X_train, Y_train)
print(clf.best_params_)
best_model = clf.best_estimator_
train_accuracy=best_model.score(X_train, Y_train)
test_accuracy=best_model.score(X_test, Y_test)

print('Training accuracy: %0.2f%%' % (train_accuracy * 100))
print('Testing accuracy: %0.2f%%' % (test_accuracy * 100))
