# Iris-classification
ML model to classify different Iris Flower images
Iris Species Classification using SVM
This project demonstrates the use of Support Vector Machines (SVM) to classify the Iris dataset into three different species: Iris Setosa, Iris Versicolor, and Iris Virginica. The dataset is visualized using Seaborn's pairplot, and SVM is implemented using the Scikit-learn library. Additionally, GridSearchCV is employed to optimize the hyperparameters of the SVM model.

Table of Contents
Introduction
Dataset
Installation
Code Walkthrough
Data Visualization
Train-Test Split
Model Training
Model Evaluation
Hyperparameter Tuning
Results
Conclusion
Acknowledgments
Introduction
The Iris dataset is one of the most well-known datasets in machine learning. It includes 150 samples of iris flowers, each described by four features: sepal length, sepal width, petal length, and petal width. The goal is to classify these samples into three species:

Iris Setosa
Iris Versicolor
Iris Virginica
This project applies an SVM classifier to predict the species of the flowers based on the provided features.

Dataset
The dataset used in this project is the famous Iris dataset, which contains 150 rows and 5 columns:

Sepal Length
Sepal Width
Petal Length
Petal Width
Species
Installation
To run this code, you need to have the following libraries installed:

Python 3.x
Jupyter Notebook
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Install the required packages using pip:

bash
pip install pandas numpy matplotlib seaborn scikit-learn
Code Walkthrough
Data Visualization
Using Seaborn's pairplot, we visualize the relationship between features and the species. The following images of the Iris species are also displayed:

Iris Setosa


Iris Versicolor


Iris Virginica


Train-Test Split
We split the dataset into training (70%) and testing (30%) sets using train_test_split from Scikit-learn.

python
X = iris.drop('species', axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
Model Training
The SVM model is trained using the training data:

python
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train, y_train)
Model Evaluation
Predictions are made on the test set, and we evaluate the model using the confusion matrix and classification report.

python
predictions = svc_model.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
Hyperparameter Tuning
We apply GridSearchCV to optimize the C and gamma hyperparameters for the SVM model.

python
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)
After tuning, we evaluate the model again:

python
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))
Results
The final model achieves 98% accuracy on the test set. Below is the classification report after hyperparameter tuning:

markdown
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        16
Iris-versicolor       0.94      1.00      0.97        17
 Iris-virginica       1.00      0.92      0.96        12

       accuracy                           0.98        45
      macro avg       0.98      0.97      0.98        45
   weighted avg       0.98      0.98      0.98        45
Conclusion
This project demonstrates the effectiveness of SVM for classifying the Iris dataset with high accuracy. Using GridSearchCV to optimize hyperparameters further improved the performance.
