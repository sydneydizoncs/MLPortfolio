#!/usr/bin/env python
# coding: utf-8

# # Lab 3: Training Decision Tree & KNN Classifiers

# In[59]:


import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.mode.chained_assignment = None 


from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In this Lab session, you will implement the following steps:
# 
# 1. Load the Airbnb "listings" data set
# 2. Convert categorical features to one-hot encoded values
# 3. Split the data into training and test sets
# 4. Fit a Decision Tree classifier and evaluate the accuracy
#  - Plot the accuracy of the DT model as a function of hyperparameter max depth
# 5. Fit a KNN classifier and evaluate the accuracy
#  - Plot the accuracy of the KNN model as a function of hyperparameter $k$

# ## Part 1. Load the Dataset

# We will work with a preprocessed version of the Airbnb NYC "listings" data set.

# <b>Task</b>: load the data set into a Pandas DataFrame variable named `df`:

# In[60]:


# Do not remove or edit the line below:
filename = os.path.join(os.getcwd(), "data", "airbnb.csv.gz")

df = pd.read_csv(filename, header=0)


# In[61]:


df.shape


# In[62]:


df.head(10)


# In[63]:


df.columns


# ## Part 2. One-Hot Encode Categorical Values
# 

# Transform the string-valued categorical features into numerical boolean values using one-hot encoding.

# ### a. Find the Columns Containing String Values

# First, let us identify all features that need to be one-hot encoded:

# In[64]:


df.dtypes


# **Task**: add all of the column names of variables of type 'object' to a list named `to_encode`

# In[65]:


to_encode = list(df.select_dtypes(include = ['object']).columns)
print(to_encode)


# Let's take a closer look at the candidates for one-hot encoding

# In[66]:


df[to_encode].nunique()


# Notice that one column stands out as containing two many values for us to attempt to transform. For this exercise, the best choice is to simply remove this column. Of course, this means losing potentially useful information. In a real-life situation, you would want to retain all of the information in a column, or you could selectively keep information in.
# 
# In the code cell below, drop this column from Dataframe `df` and from the `to_encode` list.

# In[67]:


df.drop(columns = ['amenities'], inplace = True)
to_encode.remove('amenities')


# ### b. One-Hot Encode all Unique Values

# All of the other columns in `to_encode` have reasonably small numbers of unique values, so we are going to simply one-hot encode every unique value of those columns.

# <b>Task</b>: complete the code below to create one-hot encoded columns
# Tip: Use the sklearn `OneHotEncoder` class

# In[68]:


from sklearn.preprocessing import OneHotEncoder

# Create the encoder:
encoder = OneHotEncoder(handle_unknown="error", sparse=False)

# Apply the encoder:
df_enc = pd.DataFrame(encoder.fit_transform(df[to_encode]))


# Reinstate the original column names:
df_enc.columns = encoder.get_feature_names(to_encode)


# In[69]:


df_enc.head()


# <b>Task</b>: You can now remove the original columns that we have just transformed from DataFrame `df`.
# 

# In[70]:


df.drop(to_encode, axis=1, inplace=True)


# In[71]:


df.head()


# <b>Task</b>: You can now join the transformed categorical features contained in `df_enc` with DataFrame `df`

# In[72]:


df = df.join(df_enc)


# Glance at the resulting column names:

# In[73]:


df.columns


# Check for missing values.

# In[74]:


print(df.isna().sum())


# ## Part 3. Create Training and Test Data Sets

# ### a. Create Labeled Examples 

# <b>Task</b>: Choose columns from our data set to create labeled examples. 
# 
# In the `airbnb` dataset, we will choose column `host_is_superhost` to be the label. The remaining columns will be the features.
# 
# Obtain the features from DataFrame `df` and assign to `X`.
# Obtain the label from DataFrame `df` and assign to `Y`
# 

# In[75]:


y = df["host_is_superhost"]
X = df.drop(["host_is_superhost"],axis=1)


# In[76]:


print("Number of examples: " + str(X.shape[0]))
print("\nNumber of Features:" + str(X.shape[1]))
print(str(list(X.columns)))


# ### b. Split Examples into Training and Test Sets

# <b>Task</b>: In the code cell below create training and test sets out of the labeled examples using Scikit-learn's `train_test_split()` function. 
# 
# Specify:
#     * A test set that is one third (.33) of the size of the data set.
#     * A seed value of '123'. 

# In[77]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)


# Check that the dimensions of the training and test datasets are what you expected

# In[78]:


print(X_train.shape)
print(X_test.shape)


# ## Part 4. Implement a Decision Tree Classifier

# The code cell below contains a shell of a function named `train_test_DT()`. This function should train a Decision Tree classifier on the training data, test the resulting model on the test data, and compute and return the accuracy score of the resulting predicted class labels on the test data. Remember to use ```DecisionTreeClassifier()``` to create a model object.
# 
# <b>Task:</b> Complete the function to make it work.

# In[79]:


def train_test_DT(X_train, X_test, y_train, y_test, leaf, depth, crit='entropy'):
    '''
    Fit a Decision Tree classifier to the training data X_train, y_train.
    Return the accuracy of resulting predictions on the test set.
    Parameters:
        leaf := The minimum number of samples required to be at a leaf node 
        depth := The maximum depth of the tree
        crit := The function to be used to measure the quality of a split. Default: gini.
    '''
    
      # 1. Create the  Scikit-learn DecisionTreeClassifier model object below and assign to variable 'model'
    model = DecisionTreeClassifier(criterion = crit, max_depth = depth, min_samples_leaf = leaf)
  
    # 2. Fit the model to the training data below
    model.fit(X_train, y_train)
    
    # 3. Make predictions on the test data and assign the result to the variable 'class_label_predictions' below
    class_label_predictions = model.predict(X_test)    
  
    # 4. Compute the accuracy and save the result to the variable 'acc_score' below
    acc_score = accuracy_score(y_test, class_label_predictions)  
    
    return acc_score
   
    
    return acc_score


# #### Visualization

# The cell below contains a function that you will use to compare the accuracy results of training multiple models with different hyperparameter values.
# 
# Function `visualize_accuracy()` accepts two arguments:
# 1. a list of hyperparamter values
# 2. a list of accuracy scores
# 
# Both lists must be of the same size.

# In[80]:


# Do not remove or edit the code below

def visualize_accuracy(hyperparam_range, acc):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    p = sns.lineplot(x=hyperparam_range, y=acc, marker='o', label = 'Full training set')
        
    plt.title('Test set accuracy of the model predictions, for ' + ','.join([str(h) for h in hyperparam_range]))
    ax.set_xlabel('Hyperparameter value')
    ax.set_ylabel('Accuracy')
    plt.show()


# #### Train on Different Values of Hyperparameter Max Depth

# <b>Task:</b> 
# 
# Complete function `train_multiple_trees()` in the code cell below. The function should train multiple decision trees and return a list of accuracy scores.
# 
# The function will:
# 
# 1. accept list `max_depth_range` and `leaf` as parameters; list `max_depth_range` will contain multiple values for hyperparameter max depth.
# 
# 2. loop over list `max_depth_range` and at each iteration:
# 
#     a. index into list `max_depth_range` to obtain a value for max depth<br>
#     b. call `train_test_DT` with the training and test set, the value of max depth, and the value of `leaf`<br>
#     c. print the resulting accuracy score<br>
#     d. append the accuracy score to list `accuracy_list`<br>
# 

# In[81]:


def train_multiple_trees(max_depth_range, leaf):
    
    accuracy_list = []

    for md in max_depth_range:
        score = train_test_DT(X_train, X_test, y_train, y_test, leaf, md)
        accuracy_list.append(float(score))
    
    return accuracy_list


# The code cell below tests function `train_multiple_trees()` and calls function `visualize_accuracy()` to visualize the results.

# In[82]:


max_depth_range = [8, 32]
leaf = 1

acc = train_multiple_trees(max_depth_range, leaf)

visualize_accuracy(max_depth_range, acc)


# <b>Analysis</b>: Is this graph conclusive for determining a good value of max depth?

# <Double click this Markdown cell to make it editable, and record your findings here.>
# 
# <b>This graph is not conclusive because there are not enough parameters to test the max_depth_range.</b>

# <b>Task:</b> Let's train on more values for max depth.
# 
# In the code cell below:
# 
# 1. call `train_multiple_trees()` with arguments `max_depth_range` and `leaf`
# 2. call `visualize_accuracy()` with arguments `max_depth_range` and `acc`
# 

# In[83]:


max_depth_range = [2**i for i in range(6)]
leaf = 1
acc = train_multiple_trees(max_depth_range, leaf)

visualize_accuracy(max_depth_range, acc) 


# <b>Analysis</b>: Analyze this graph. Keep in mind that this is the performance on the test set, and pay attention to the scale of the y-axis. Answer the following questions in the cell below.<br>
# How would you go about choosing the best model based on this plot? Is it conclusive? <br>
# What other hyperparameters of interest would you want to vary to make sure you are finding the best model fit?

# <Double click this Markdown cell to make it editable, and record your answers here.>
# 
# <b>Based on the graphs shown, the best model would use a hyperparameter of 8, abeit it being non-conclusive because it uses too few values; it assumes that the accuracy decreases linearly, which is false. Another hyperparameter of interest that you can vary could be the leaf value.</b>

# ## Part 5. Implement a KNN Classifier
# 

# Note: In this section you will train KNN classifiers using the same training and test data.

# The code cell below contains a shell of a function named `train_test_knn()`. This function should train a KNN classifier on the training data, test the resulting model on the test data, and compute and return the accuracy score of the resulting predicted class labels on the test data. 
# 
# Remember to use ```KNeighborsClassifier()``` to create a model object and call the method with one parameter: `n_neighbors = k`. 
# 
# <b>Task:</b> Complete the function to make it work.

# In[84]:


def train_test_knn(X_train, X_test, y_train, y_test, k):
    '''
    Fit a k Nearest Neighbors classifier to the training data X_train, y_train.
    Return the accuracy of resulting predictions on the test data.
    '''
    
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(X_train, y_train)
    class_label_predictions = model.predict(X_test)
    acc_score = accuracy_score(y_test, class_label_predictions)

    
    return acc_score


# #### Train on Different Values of Hyperparameter K
# 
# <b>Task:</b> 
# 
# Just as you did above, complete function `train_multiple_knns()` in the code cell below. The function should train multiple KNN models and return a list of accuracy scores.
# 
# The function will:
# 
# 1. accept list `k_range` as a parameter; this list will contain multiple values for hyperparameter $k$
# 
# 2. loop over list `k_range` and at each iteration:
# 
#     a. index into list `k_range` to obtain a value for $k$<br>
#     b. call `train_test_knn` with the training and test set, and the value of $k$<br>
#     c. print the resulting accuracy score<br>
#     d. append the accuracy score to list `accuracy_list` <br>
# 

# In[56]:


def train_multiple_knns(k_range):
    
    accuracy_list = []

    for k in k_range:
        score = train_test_knn(X_train, X_test, y_train, y_test, k)
        print('k=' + str(k) + ', accuracy score: ' + str(score))
        accuracy_list.append(float(score))
    
    return accuracy_list


# The code cell below uses your `train_multiple_knn()` function to train 3 KNN models, specifying three values for $k$: $3, 30$, and $300$. It calls function `visualize_accuracy()` to visualize the results. Note: this make take a second.

# In[57]:


k_range = [3, 30, 300]
acc = train_multiple_knns(k_range)

visualize_accuracy(k_range, acc)


# <b>Task:</b> Let's train on more values for $k$
# 
# In the code cell below:
# 
# 1. call `train_multiple_knns()` with argument `k_range`
# 2. call `visualize_accuracy()` with arguments `k_range` and the resulting accuracy list obtained from `train_multiple_knns()`
# 

# In[58]:


k_range = np.arange(1, 40, step = 3) 

acc = train_multiple_knns(k_range)
visualize_accuracy(k_range, acc)


# <b>Analysis</b>: Compare the performance of the KNN model relative to the Decision Tree model, with various hyperparameter values and record your findings in the cell below.

# <Double click this Markdown cell to make it editable, and record your findings here.>
# 
# <b>The KNN model shows varying performance with different values of the hyperparameter 'k'. For smaller values of 'k', the KNN model tends to overfit the training data. As 'k' increases, the model's complexity decreases, leading to a decrease in overfitting; this results in better generalization and improved accuracy on the test set. However, with very large values of 'k', the KNN model may oversimplify the decision boundaries, leading to a decrease in accuracy.
# 
# On the other hand, with an appropriate choice of hyperparameters, such as limiting the depth of the tree or using pruning techniques, the Decision Tree model can achieve good accuracy and generalization. Decision Trees are capable of capturing complex relationships and interactions within the data, making them effective for both simple and complex datasets.
# 
# In general, the Decision Tree model can handle both numerical and categorical features without requiring one-hot encoding, which can simplify the preprocessing steps. KNN models can be more sensitive to the scaling and normalization of features, as they rely on distance-based calculations.The choice between the KNN model and the Decision Tree model is dependent on many factors, such as on the specific dataset or the complexity of the problem.
# </b>
