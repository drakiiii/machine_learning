--- READ ME FOR IRIS MODEL ---

Dataset to download can be found here: https://www.kaggle.com/uciml/iris

Guide to create ML model can be found here: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

Introductory ML model which classifies an Iris plant species based on 4 features, 'Sepal Length', 'Sepal Width', 'Petal Length' and 'Petal Width'.

The guide runs through some EDA (Explanatory Data Analysis) using basic statistical methods, followed by some data visualisation techniques using univariate and multivariate plots.

The Iris dataset is then split into training and test datasets using 'train_test_split' and by slicing the original data into features and target variables, X_train, X_test, y_train, y_test.

Using 'cross_val_score' we then evaluate the accuracy of 6 machine learning algorithms on the newly-split data. The algorithms are:
- Logistic Regression (LR)
- Linear Discriminant Analysis (LDA)
- K-Nearest Neighbors (KNN)
- Classification and Regression Trees (CART)
- Gaussian Naive Bayes (GNB)
- Support Vector Machines (SVM)
