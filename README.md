# Supervised Machine Learning

## Project Overview

Credit risk is an unbalanced classification problem, as the number of good loans easily outnumber the number of risky loans.  For this module, one needs to employ different techniques to train and evaluate models with unbalanced classes. Also, needs to use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling. The final task is to evaluate the performance of these models and make a recommendation on whether they should be used to predict credit risk.

### Objectives

The goals of this challenge:

1. Implement machine learning models.
2. Use resampling to attempt to address class imbalance.
3. Evaluate the performance of machine learning models.

## Resources

Data Source: LoanStats_2019Q1.csv
Software: Python 3.7.6, Anaconda 4.8.4, Jupyter Notebook 6.0.3

## Challenge Overview

 1. Oversample the data using the RandomOverSampler and SMOTE algorithms.
 2. Undersample the data using the cluster centroids algorithm.
 3. Use a combination approach with the SMOTEENN algorithm.
 
For each of the above, you’ll:
  - Train a logistic regression classifier (from Scikit-learn) using the resampled data.
  - Calculate the balanced accuracy score using balanced_accuracy_score from sklearn.metrics.
  - Generate a confusion_matrix.
  - Print the classification report (classification_report_imbalanced from imblearn.metrics).

### Extension

Use 100 estimators for both classifiers, and complete the following steps for each model:

 1. Train the model and generate predictions.
 2. Calculate the balanced accuracy score.
 3. Generate a confusion matrix.
 4. Print the classification report (classification_report_imbalanced from imblearn.metrics).
 5. For the BalancedRandomForestClassifier, print the feature importance, sorted in descending order (from most to least important feature), along with the feature score.
 
### Summary 
