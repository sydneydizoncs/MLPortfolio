#!/usr/bin/env python
# coding: utf-8

# # Assignment 3: Building a Decision Tree After Feature Transformations

# In[4]:


import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In this assignment, you will implement the following steps to build a Decision Tree classification model:
# 
# 1. Load the "cell2celltrain" data set
# 2. Convert categorical features to one-hot encoded values
# 3. Split the data into training and test sets
# 4. Fit a Decision Tree classifier and evaluate the accuracy of its predictions
# 5. Plot the training set accuracy

# ## Part 1. Load the Data Set

# We will work with the "cell2celltrain" data set.

# In[5]:


# Do not remove or edit the line below:
filename = os.path.join(os.getcwd(), "data", "cell2celltrain.csv")


# **Task**: Load the data and save it to DataFrame `df`.

# In[6]:


df = pd.read_csv(filename)


# **Task**: Display the shape of `df` -- that is, the number of records (rows) and variables (columns)

# In[7]:


df.shape


#  For the purpose of this assignment, we will remove the `Married` column due to missing values

# In[8]:


df.drop(columns = ['Married'], inplace=True)


# ## Part 2. One-Hot Encode Categorical Values
# 

# To implement a decision tree model, we must first transform the string-valued categorical features into numerical boolean values using one-hot encoding.

# ### a. Find the Columns Containing String Values

# In[9]:


df.dtypes


# **Task**: Add all of the column names whos values are of type 'object' to a list named `to_encode`.

# In[11]:


to_encode = list(df.select_dtypes(include = ['object']).columns)
print(to_encode)


# Let's take a closer look at the candidates for one-hot encoding:

# In[12]:


df[to_encode].nunique()


# For all of the columns except for `ServiceArea`, it should be straightforward to replace a given column with a set of several new binary columns for each unique value. However, let's first deal with the special case of `ServiceArea`.

# ### b. One Hot-Encoding 'ServiceArea': The Top 10 Values

# Take a look at the number of unique values of the `ServiceArea` column. There are two many unique values in the `ServiceArea` column to attempt to create a new binary indicator column per value! 
# One thing we could do is to see if some of the values in `ServiceArea` are occurring frequently. We will then one-hot encode just those frequent values.

# <b>Task</b>: Get the top 10 most frequent values in 'ServiceArea' and store them in list `top_10_SA`.

# In[13]:


top_10_SA = df['ServiceArea'].value_counts().nlargest(10).index.tolist()


# <b>Task</b>: Write a `for` loop that loops through every value in `top_10_SA` and creates one-hot encoded columns, titled <br>'ServiceArea + '\_' + $<$service area value$>$'. For example, there will be a column named  'ServiceArea\_NYCBRO917'. Use the NumPy `np.where()`function  to accomplish this.

# In[14]:


for x in top_10_SA:
    columnName = 'ServiceArea_' + x
    df[columnName] = np.where(df['ServiceArea'] == x, 1, 0)


# <b>Task</b>: 
# 1. Drop the original, multi-valued `ServiceArea` column from the DataFrame `df`. 
# 2. Remove 'ServiceArea' from the `to_encode` list.

# In[15]:


df = df.drop('ServiceArea', axis=1)
to_encode.remove('ServiceArea')


# In[16]:


df.head()


# ### c. One Hot-Encoding all Remaining Columns: All Unique Values per Column

# All other columns in `to_encode` have reasonably small numbers of unique values, so we are going to simply one-hot encode every unique value of those columns.
# 
# <b>Task</b>: In the code cell below, iterate over column names and create new columns for all unique values.
# 1. Use a loop to loop over the column names in `to_encode` 
# 2. In the loop:
#     1. Use the Pandas `pd.get_dummies()` function and save the result to variable `temp_df`
#     2. Use `df.join` to join `temp_df` with DataFrame `df`
# 

# In[17]:


for columnName in to_encode:
    temp_df = pd.get_dummies(df[columnName], prefix=columnName)
    df = df.join(temp_df)


# In[18]:


df.head()


# <b>Task</b>: Remove all the original columns from DataFrame `df`

# In[19]:


df = df.drop(to_encode, axis=1)


# In[20]:


df.columns


# Check that the data does not contain any missing values. The absense of missing values is necessary for training a Decision Tree model.

# In[21]:


missingVals = df.isna().any()
if missingVals.any():
    print("Columns with missings vals")
else:
    print("DataFrame has no missing vals")


# ## Part 3: Create Labeled Examples from the Data Set 

# <b>Task</b>: Create labeled examples from DataFrame `df`. 
# In the code cell below carry out the following steps:
# 
# * Get the `Churn` column from DataFrame `df` and assign it to the variable `y`. This will be our label.
# * Get all other columns from DataFrame `df` and assign them to the variable `X`. These will be our features. 

# In[22]:


y = df['Churn']
X = df.drop('Churn', axis=1)


# ## Part 4:  Create Training and Test Data Sets

# <b>Task</b>: In the code cell below create training and test sets out of the labeled examples. 
# 
# 1. Use Scikit-learn's `train_test_split()` function to create the data sets.
# 
# 2. Specify:
#     * A test set that is 30 percent (.30) of the size of the data set.
#     * A seed value of '123'. 
#     
# 

# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)


# Check that the dimensions of the training and test datasets are what you expected:

# In[24]:


print(X_train.shape)
print(X_test.shape)


# ## Part 5. Fit a Decision Tree Classifer and Evaluate the Model

# The code cell below contains a shell of a function named `train_test_DT()`. This function should train a Decision Tree classifier on the training data, test the resulting model on the test data, and compute and return the accuracy score of the resulting predicted class labels on the test data.
# 
# <b>Task:</b> Complete the function to make it work.

# In[25]:


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


# ### Train on Different Hyperparameter Values

# <b>Task:</b> Train two Decision Tree classifiers using your function. 
# 
# - one with a low value of depth
# - one high value of depth
# 
# Specify the minimum number of samples at the leaf node to be equal to $1$ for both trees.
# 
# Save the resulting accuracy scores to list `acc`. Print the list.

# In[35]:


depth1= 8
depth2 = 32
leaf = 1

max_depth_range = [2**i for i in range(6)]
acc = []
def train_multiple_trees(max_depth_range, leaf):
    
    accuracy_list = []

    for md in max_depth_range:
        score = train_test_DT(X_train, X_test, y_train, y_test, leaf, md)
        accuracy_list.append(float(score))
    
    return accuracy_list
acc = train_multiple_trees(max_depth_range, leaf)


# <b>Task</b>: Visualize the results (Hint: use a `seaborn` lineplot).

# In[36]:


fig = plt.figure()
ax = fig.add_subplot(111)
p = sns.lineplot(x=max_depth_range, y=acc, marker='o', label = 'Full training set')
    
plt.title('Test set accuracy of the DT predictions, for $max\_depth\in\{8, 32\}$')
ax.set_xlabel('max_depth')
ax.set_ylabel('Accuracy')
plt.show()


# <b>Analysis</b>: Experiment with different values for `max_depth`. Add these values to the list `max_depth_range` (i.e. change the values, create a list containing more values), retrain your model and rerun with the visualization cell above. Compare the different accuracy scores.
# 
# Once you find the best value for `max_depth`, experiment with different values for `leaf` and compare the different accuracy scores.
# 
# Is there one model configuration that yields the best score? Record your findings in the cell below.

# <Double click this Markdown cell to make it editable, and record your findings here.>
# 
# <b> Before experimenting with different values for max_depth by assigning the variable to a range, [2**i for i in range(6)], I first used the two values of 8 and 32 to run my visualization; this resulted in a straight line graph, an overall very simplified and inaccurate version of the data. After testing and running my data with more depth values, I was able to get a more defined and accurate graph, which shows a better generalization using more of the data given. </b>
# 

# In[ ]:




