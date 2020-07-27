#!/usr/bin/env python3.8

# Load libraries
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

# Load dataset
file_path = "/Users/GOD/Documents/Datasets/iris_data.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(file_path, names=names)
a = "----------------------------------------------------------------------"
# Dataset Shape
# Size (rows and columns) of the dataset
print(dataset.shape)
print(a)
# Dataset Head
# First 20 rows (specimens) of the dataset
print(dataset.head(20))
print(a)
# Descriptions
# Statistical summary of each attribute
print(dataset.describe())
print(a)
# Class Distribution
# Number of instances of each class
print(dataset.groupby('class').size())
print(a)

# Data Visualisation
# Univariate Plots - Understand each attribute better

# Box and Whisker plots
# Gives us a better idea of the distribution of the input variables
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

# Histograms
# Also gives us an idea of the distribution
dataset.hist()
pyplot.show()