#!/usr/bin/env python3.8

# 2 Load the Data

# 2.1 Importing Libraries
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from pandas import read_csv

# 2.2 Load Dataset
file_path = "/Users/GOD/Documents/Datasets/iris_data.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(file_path, names=names)


# 3 Summarise the Data

a = "----------------------------------------------------------------------"
# 3.1 Dimensions of a Dataset
# Dataset Shape
# Size (rows and columns) of the dataset
print(dataset.shape)
pyplot.show()
print(a)

# 3.2 Peak at the Data
# Dataset Head
# First 20 rows (specimens) of the dataset
print(dataset.head(20))
pyplot.show()
print(a)

# 3.3 Statistical Summary
# Descriptions
# Statistical summary of each attribute
print(dataset.describe())
pyplot.show()
print(a)

# 3.4 Class Distribution
# Class Distribution
# Number of instances of each class
print(dataset.groupby('class').size())
pyplot.show()
print(a)

# 4 Data Visualisation

# 4.1 Univariate Plots - Understand each specific attribute better

# 4.1.1 Box and Whisker Plots - Gives us a better idea of the distribution of the input variables
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

# 4.1.2 Histograms - Also gives us an idea of the distribution
# dataset.hist()

# 4.2 Multivariate Plots - Study interactions between variables

# 4.2.1 Scatter Plot Matrix
# Useful to spot structured relationships between variables
# scatter_matrix(dataset)
# In the scatter matrix, it is clear to see a high correlation and predictable relationship between some of the
# combinations of variables

# 5 Evaluate Some Algorithms
# Will create some models and estimate their accuracy on unseen data, steps as follows:
# Step 1) Separate out a validation dataset.
# Step 2) Set up the test harness to use 10-fold cross validation.
# Step 3) Build several models to predict species from flower measurements.
# Step 4) Select the best model.

# 5.1 Create a Validation Dataset
# We know the model we created earlier is good.
# Later on, we will evaluate the models that we create on unseen data using statistical methods.
# Evaluating the accuracy of the best model by testing it on actual unseen data.
# By holding back some of the data, we will be able to see how accurate the best model really is.
# We will split the dataset into two parts, 80% to test the models and the other 20% held back as validation data.

# Split-out validation dataset
# This assigns training data into X_train, Y_train for testing the models and
# X_validation, Y_validation to validate the models.
# We use a python slice to select columns in the NumPy array.
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# 5.2 Test Harness
# Using stratified 10-fold cross validation to estimate the model accuracy.
# Then split dataset into 10 parts, train on 9 and test on 1.
# Then repeat with all different combinations of test-train splits. e.g. 8-2, 7-3, 6-4 etc.
# Stratified means each split of the dataset will aim to have the same distribution of example by class
# as exist in the whole training dataset.
# We set the random seed via the random_state argument to a fixed number to ensure that the algorithms are evaluated on
# the same splits of the training set data.
# The specific random_seed value does not matter.
# We are using 'accuracy' to evaluate the models individually. This is a ratio of the total number of
# correctly predicted instances divided by the total number of instances, multiplied by 100 to give a percentage.
# We will use the 'scoring' variable when we run, build and evaluate each model next.

# 5.3 Build Models
# We don't know what algorithms to use, or which ones would be good for this problem.
# From the earlier results, we could see some of the classes are partially linearly separable, which is good.
# We will test the following six algorithms:
# Algo 1) Logistic Regression (LR)
# Algo 2) Linear Discriminant Analysis (LDA)
# Algo 3) K-Nearest Neighbour (KNN)
# Algo 4) Classification and Regression Trees (CART)
# Algo 5) Gaussian Naive Bayes (NB)
# Algo 6) Support Vector Machines (SVM)
# A mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms.

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# Evaluating each model
# Initialising results, names list
results = []
names = []
# Running stratified 10-fold cross validation
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Results:
# LR: 0.941667 (0.065085)
# LDA: 0.975000 (0.038188)
# KNN: 0.958333 (0.041667)
# CART: 0.941667 (0.053359)
# NB: 0.950000 (0.055277)
# SVM: 0.983333 (0.033333)

# From the results we can that Support Vector Machines (SVM) is the most accurate model with ~98.3%.
# We can create a univariate plot from the accuracy results to compare the spread and mean of the models.
# There is a population of accuracy measures as each model was tested 10 times using the stratified 10-fold
# cross validation.
# We will use a box and whisker plot to compare the accuracy of the models.

# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# 6 Make Predictions
# We will use SVM as our algorithm to make predictions as this was the best performing model
# Now we want to get an idea of the accuracy of our model on the validation dataset.
# It will give us a final, independent check on the accuracy of the best model.
# The validation set adds value in case you made a slip during training, like overfitting the dataset or a data leak.
# In both of these cases you will end up with an overly optimistic result.

# 6.1 Make Predictions
# We can fit the model on the entire training dataset and make predictions on the validation dataset.
# Make predictions on validation dataset
print(a)
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# 6.2 Evaluate Predictions
# Comparing the predictions to expected results in the validation dataset and calculating classification accuracy,
# a confusion matrix and a classification report.
# Evaluate predictions

# Classification Accuracy
print(accuracy_score(Y_validation, predictions))
print(a)
# Confusion Matrix
print(confusion_matrix(Y_validation, predictions))
print(a)
# Classification Report
print(classification_report(Y_validation, predictions))

# Prediction Results

# Classification Accuracy
# 96.66%

# Confusion Matrix

# [[11  0  0]
#  [ 0 12  1]
#  [ 0  0  6]]

# Shows where mistake in classification made
# Iris-virginica mistaken for Iris-versicolor


# Classification Report

#                  precision    recall  f1-score   support
#
#     Iris-setosa       1.00      1.00      1.00        11
# Iris-versicolor       1.00      0.92      0.96        13
#  Iris-virginica       0.86      1.00      0.92         6
#
#        accuracy                           0.97        30
#       macro avg       0.95      0.97      0.96        30
#    weighted avg       0.97      0.97      0.97        30

# Provides breakdown of each class by precision, recall, f1-score and support showing excellent results.
