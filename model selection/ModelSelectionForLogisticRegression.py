#!/usr/bin/env python
# coding: utf-8

# # Lab 5: Model Selection for Logistic Regression

# In[1]:


import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve


# In this lab assignment, you will:
# 
# 1. Load the Airbnb "listings" data set.
# 2. Train and test a logistic regression (LR) model using the scikit-learn default hyperparameter values.
# 2. Perform a grid search to identify the LR hyperparameter value that results in the best cross-validation score.
# 3. Fit the optimal model to the training data and make predictions on the test data.
# 4. Create a confusion matrix for both models.
# 5. Plot a precision-recall curve for both models.
# 6. Plot the ROC and compute the AUC for both models.
# 7. Perform feature selection.
# 
# **<font color='red'>Note: Some of the code cells in this notebook may take a while to run.</font>**

# ## Part 1: Load the Data Set

# We will work with a preprocessed version of the Airbnb NYC "listings" data set. 
# 
# <b>Task</b>: In the code cell below, use the same method you have been using to load the data using `pd.read_csv()` and save it to DataFrame `df`.
# 
# You will be working with the file named "airbnb_readytofit.csv.gz" that is located in a folder named "data".

# In[2]:


filename = os.path.join(os.getcwd(), "data", "airbnb_readytofit.csv.gz")
df = pd.read_csv(filename, header=0)


# ## Part 2: Create Training and Test Data Sets

# ### Create Labeled Examples 
# 
# <b>Task</b>: Create labeled examples from DataFrame `df`. 
# In the code cell below, carry out the following steps:
# 
# * Get the `host_is_superhost` column from DataFrame `df` and assign it to the variable `y`. This will be our label.
# * Get all other columns from DataFrame `df` and assign them to the variable `X`. These will be our features. 

# First, we will store the label column as a separate object, called `y`, and consequently remove that column from the `X` feature set:

# In[3]:


y = df['host_is_superhost']
X = df.drop(columns=['host_is_superhost'])


# ### Split Labeled Examples Into Training and Test Sets
# 
# <b>Task</b>: In the code cell below, create training and test sets out of the labeled examples. 
# 
# 1. Use scikit-learn's `train_test_split()` function to create the data sets.
# 
# 2. Specify:
#     * A test set that is 10 percent of the size of the data set.
#     * A seed value of '1234'. 
#     
# 

# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)


# In[5]:


X_train.head()


# ## Part 3: Fit and Evaluate a Logistic Regression Model With Default Hyperparameter Values

# <b>Task</b>: In the code cell below:
# 
# 1. Using the scikit-learn `LogisticRegression` class, create a logistic regression model object with the following arguments: `max_iter=1000`. You will use the scikit-learn default value for hyperparameter $C$, which is 1.0. Assign the model object to the variable `model_default`.
# 
# 2. Fit the model to the training data.

# In[6]:


# 1. Create the Scikit-learn LogisticRegression model object with max_iter=1000
model_default = LogisticRegression(max_iter=1000)

# 2. Fit the model to the training data
model_default.fit(X_train, y_train)


# <b>Task:</b> Test your model on the test set (`X_test`). 
# 
# 1. Use the ``predict_proba()`` method  to use the fitted model to predict class probabilities for the test set. Note that the `predict_proba()` method returns two columns, one column per class label. The first column contains the probability that an unlabeled example belongs to class `False` (`host_is_superhost` is "False") and the second column contains the probability that an unlabeled example belongs to class `True` (`host_is_superhost` is "True"). Save the values of the *second* column to a list called ``proba_predictions_default``.
# 
# 2. Use the ```predict()``` method to use the fitted model `model_default` to predict the class labels for the test set. Store the outcome in the variable ```class_label_predictions_default```. Note that the `predict()` method returns the class label (True or False) per unlabeled example.

# In[7]:


# 1. Make predictions on the test data using the predict_proba() method
proba_predictions_default = model_default.predict_proba(X_test)[:, 1].tolist()

# 2. Make predictions on the test data using the predict() method
class_label_predictions_default = model_default.predict(X_test)


# <b>Task</b>: Evaluate the accuracy of the model using a confusion matrix. In the cell below, create a confusion matrix out of `y_test` and `class_label_predictions_default`.
# 
# First, create the confusion matrix, then create a Pandas DataFrame out of the confusion matrix for display purposes.
# Recall that we are predicting whether the host is a 'superhost' or not. Label the confusion matrix accordingly.

# In[8]:


# Create the confusion matrix
confusion_mat_default = confusion_matrix(y_test, class_label_predictions_default)

# Create a Pandas DataFrame from the confusion matrix
confusion_df_default = pd.DataFrame(confusion_mat_default, index=['Actual False', 'Actual True'],
                                    columns=['Predicted False', 'Predicted True'])
confusion_df_default


# ## Part 4: Perform Logistic Regression Model Selection Using `GridSearchSV`
# 
# Our goal is to find the optimal choice of hyperparameter $C$. 

# ### Set Up a Parameter Grid 
# 
# The code cell below creates a dictionary called `param_grid` with:
# * a key called 'C' 
# * a value which is a list consisting of 10 values for the hyperparameter $C$
# 
# It uses a scikit-learn function `11_min_c()` to assist in the creation of possible values for $C$. For more information, consult the online [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.l1_min_c.html).

# In[9]:


from sklearn.svm import l1_min_c

cs = l1_min_c(X_train, y_train, loss="log") * np.logspace(0, 7, 16)
param_grid = dict(C = list(cs))
param_grid


# ### Perform Grid Search Cross-Validation

# <b>Task:</b> Use `GridSearchCV` to search over the different values of hyperparameter $C$ to find the one that results in the best cross-validation (CV) score.
# 
# Complete the code in the cell below.

# In[10]:


print('Running Grid Search...')

# 1. Create a LogisticRegression model object with the argument max_iter=1000. 
#    Save the model object to the variable 'model'
model = LogisticRegression(max_iter=1000)

# 2. Run a grid search with 5-fold cross-validation and assign the output to the 
# object 'grid'.
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}  # Provide a range of C values to explore
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

# 3. Fit the model on the training data and assign the fitted model to the 
#    variable 'grid_search'
# YOUR CODE HERE
grid_search = grid.fit(X_train, y_train)

print('Done')


# <b>Task</b>: Retrieve the value of the hyperparameter $C$ for which the best score was attained. Save the result to the variable `best_c`.

# In[11]:


best_c = grid_search.best_params_['C']
print("The best value of C is:", best_c)


# ## Part 5: Fit and Evaluate the Optimal Logistic Regression Model 

# <b>Task</b>: Initialize a `LogisticRegression` model object with the best value of hyperparameter `C` model and fit the model to the training data. The model object should be named `model_best`. Note: Supply `max_iter=1000` as an argument when creating the model object.

# In[12]:


# 1. Create the model object with the best value of hyperparameter C
model_best = LogisticRegression(max_iter=1000, C=best_c)

# 2. Fit the model to the training data
model_best.fit(X_train, y_train)


# <b>Task:</b> Test your model on the test set (`X_test`).
# 
# 1. Use the ``predict_proba()`` method  to use the fitted model `model_best` to predict class probabilities for the test set. Save the values of the *second* column to a list called ``proba_predictions_best``.
# 
# 2. Use the ```predict()``` method to use the fitted model `model_best` to predict the class labels for the test set. Store the outcome in the variable ```class_label_predictions_best```. 

# In[13]:


# 1. Make predictions on the test data using the predict_proba() method
proba_predictions_best = model_best.predict_proba(X_test)[:, 1].tolist()

# 2. Make predictions on the test data using the predict() method
class_label_predictions_best = model_best.predict(X_test)


# <b>Task</b>: Evaluate the accuracy of the model using a confusion matrix. In the cell below, create a confusion matrix out of `y_test` and `class_label_predictions_best`.

# In[14]:


confusion_mat_best = confusion_matrix(y_test, class_label_predictions_best)
confusion_df_best = pd.DataFrame(confusion_mat_best, index=['Actual False', 'Actual True'],
                                 columns=['Predicted False', 'Predicted True'])
confusion_df_best


# ## Part 6:  Plot Precision-Recall Curves for Both Models

# <b>Task:</b> In the code cell below, use `precision_recall_curve()` to compute precision-recall pairs for both models.
# 
# For `model_default`:
# * call `precision_recall_curve()` with `y_test` and `proba_predictions_default`
# * save the output to the variables `precision_default`, `recall_default` and `thresholds_default`, respectively
# 
# For `model_best`:
# * call `precision_recall_curve()` with `y_test` and `proba_predictions_best`
# * save the output to the variables `precision_best`, `recall_best` and `thresholds_best`, respectively
# 

# In[17]:


precision_default, recall_default, thresholds_default = precision_recall_curve(y_test, proba_predictions_default)
precision_best, recall_best, thresholds_best = precision_recall_curve(y_test, proba_predictions_best)


# In the code cell below, create two `seaborn` lineplots to visualize the precision-recall curve for both models. "Recall" will be on the $x$-axis and "Precision" will be on the $y$-axis. 
# 
# The plot for "default" should be green. The plot for the "best" should be red.
# 

# In[18]:


plt.figure(figsize=(8, 6))

# model_default (green)
sns.lineplot(x=recall_default, y=precision_default, color='green', label='Default')

# model_best (red)
sns.lineplot(x=recall_best, y=precision_best, color='red', label='Best')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Logistic Regression Models')
plt.legend()

# show plot
plt.show()


# ## Part 7: Plot ROC Curves and Compute the AUC for Both Models

# You will next use scikit-learn's `roc_curve()` function to plot the receiver operating characteristic (ROC) curve and the `auc()` function to compute the area under the curve (AUC) for both models.
# 
# * An ROC curve plots the performance of a binary classifier for varying classification thresholds. It plots the fraction of true positives out of the positives vs. the fraction of false positives out of the negatives. For more information on how to use the `roc_curve()` function, consult the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html).
# 
# * The AUC measures the trade-off between the true positive rate and false positive rate. It provides a broad view of the performance of a classifier since it evaluates the performance for all the possible threshold values; it essentially provides a value that summarizes the the ROC curve. For more information on how to use the `auc()` function, consult the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html).
# 
# Let's first import the functions.

# In[19]:


from sklearn.metrics import roc_curve
from sklearn.metrics import auc


# <b>Task:</b> Using the `roc_curve()` function, record the true positive and false positive rates for both models. 
# 
# 1. Call `roc_curve()` with arguments `y_test` and `proba_predictions_default`. The `roc_curve` function produces three outputs. Save the three items to the following variables, respectively: `fpr_default` (standing for 'false positive rate'),  `tpr_default` (standing for 'true positive rate'), and `thresholds_default`.
# 
# 2. Call `roc_curve()` with arguments `y_test` and `proba_predictions_best`. The `roc_curve` function produces three outputs. Save the three items to the following variables, respectively: `fpr_best` (standing for 'false positive rate'),  `tpr_best` (standing for 'true positive rate'), and `thresholds_best`.

# In[20]:


fpr_default, tpr_default, thresholds_default = roc_curve(y_test, proba_predictions_default)
fpr_best, tpr_best, thresholds_best = roc_curve(y_test, proba_predictions_best)


# <b>Task</b>: Create <b>two</b> `seaborn` lineplots to visualize the ROC curve for both models. 
# 
# The plot for the default hyperparameter should be green. The plot for the best hyperparameter should be red.
# 
# * In each plot, the `fpr` values should be on the $x$-axis.
# * In each plot, the`tpr` values should be on the $y$-axis. 
# * In each plot, label the $x$-axis "False positive rate".
# * In each plot, label the $y$-axis "True positive rate".
# * Give each plot the title "Receiver operating characteristic (ROC) curve".
# * Create a legend on each plot indicating that the plot represents either the default hyperparameter value or the best hyperparameter value.
# 
# <b>Note:</b> It may take a few minutes to produce each plot.

# #### Plot ROC Curve for Default Hyperparameter:

# In[21]:


plt.figure(figsize=(8, 6))
sns.lineplot(x=fpr_default, y=tpr_default, color='green', label='Default')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Default Hyperparameter')
plt.legend()

plt.show()


# #### Plot ROC Curve for Best Hyperparameter:

# In[22]:


plt.figure(figsize=(8, 6))
sns.lineplot(x=fpr_best, y=tpr_best, color='red', label='Best')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Best Hyperparameter')
plt.legend()

plt.show()


# <b>Task</b>: Use the `auc()` function to compute the area under the receiver operating characteristic (ROC) curve for both models.
# 
# For each model, call the function with the `fpr` argument first and the `tpr` argument second. 
# 
# Save the result of the `auc()` function for `model_default` to the variable `auc_default`.
# Save the result of the `auc()` function for `model_best` to the variable `auc_best`. 
# Compare the results.

# In[23]:


auc_default = auc(fpr_default, tpr_default)

auc_best = auc(fpr_best, tpr_best)

print(auc_default)
print(auc_best)


# ## Deep Dive: Feature Selection Using SelectKBest

# In the code cell below, you will see how to use scikit-learn's `SelectKBest` class to obtain the best features in a given data set using a specified scoring function. For more information on how to use `SelectKBest`, consult the online [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html).
# 
# We will extract the best 5 features from the Airbnb "listings" data set to create new training data, then fit our model with the optimal hyperparameter $C$ to the data and compute the AUC. Walk through the code to see how it works and complete the steps where prompted. Analyze the results.

# In[29]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Note that k=5 is specifying that we want the top 5 features
selector = SelectKBest(f_classif, k=5)
selector.fit(X, y)
filter = selector.get_support()
top_5_features = X.columns[filter]

print("Best 5 features:")
print(top_5_features)

# Create new training and test data for features
new_X_train = X_train[top_5_features]
new_X_test = X_test[top_5_features]


# Initialize a LogisticRegression model object with the best value of hyperparameter C 
# The model object should be named 'model'
# Note: Supply max_iter=1000 as an argument when creating the model object
# YOUR CODE HERE
model = LogisticRegression(max_iter=1000, C=best_c)

# Fit the model to the new training data
# YOUR CODE HERE
model.fit(new_X_train, y_train)

# Use the predict_proba() method to use your model to make predictions on the new test data 
# Save the values of the second column to a list called 'proba_predictions'
# YOUR CODE HERE
proba_predictions = model.predict_proba(new_X_test)[:, 1]
    
# Compute the auc-roc
fpr, tpr, thresholds = roc_curve(y_test, proba_predictions)
auc_result = auc(fpr, tpr)
print(auc_result)

# k = 20
selector = SelectKBest(f_classif, k=20)
selector.fit(X, y)
filter = selector.get_support()
top_20_features = X.columns[filter]

new_X_train = X_train[top_20_features]
new_X_test = X_test[top_20_features]

model = LogisticRegression(max_iter=1000, C=best_c)
model.fit(new_X_train, y_train)

proba_predictions = model.predict_proba(new_X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, proba_predictions)
auc_result = auc(fpr, tpr)
print("AUC for 'k = 20':", auc_result)


# <b>Task</b>: Consider the results. Change the specified number of features and re-run your code. Does this change the AUC value? What number of features results in the best AUC value? Record your findings in the cell below.

# <Double click this Markdown cell to make it editable, and record your findings here.>
# 
# By increasing k to 20, we can observe how the AUC value changes with an increase in k. The number of features that results in the best AUC value will indicate the optimal feature selection. The results I got from when I ran the model indicate that increasing the k value, or number of features, improves the AUC value and is beneficial for the model's performance. 
# 

# In[ ]:




