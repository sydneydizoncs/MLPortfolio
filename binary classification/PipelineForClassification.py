#!/usr/bin/env python
# coding: utf-8

# # Assignment 7: Using a Pipeline for Text Transformation and Classification

# In[1]:


import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_roc_curve, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline


# In this assignment, you will practice text vectorization to transform text into numerical feature vectors that can be used to train a classifier. You will then see how to use scikit-learn pipelines to chain together these processes into one step. You will:
# 
# 1. Load the book reviews data set.
# 2. Use a single text column as a feature. 
# 3. Transform features using a TF-IDF vectorizer. 
# 4. Fit a logistic regression model to the transformed features. 
# 5. Evaluate the performance of the model using AUC.
# 6. Set up a scikit-learn pipeline to perform the same tasks above. 
# 7. Execute the pipeline and verify that the performance is the same.
# 8. Add a grid search to the pipeline to find the optimal hyperparameter configuration.
# 9. Evaluate the performance of the optimal configuration using ROC-AUC.
# 
# **<font color='red'>Note: some of the code cells in this notebook may take a while to run</font>**

# ## Part 1: Load the Data Set

# We will work with the book review dataset that you worked with in the sentiment analysis demo.

# In[2]:


filename = os.path.join(os.getcwd(), "data", "bookReviews.csv")
df = pd.read_csv(filename, header=0)


# In[3]:


df.head()


# ## Part 2: Create Training and Test Data Sets

# ### Create Labeled Examples 
# 
# <b>Task</b>: Create labeled examples from DataFrame `df`. We will have one text feature and one label.  
# 
# In the code cell below carry out the following steps:
# 
# * Get the `Positive Review` column from DataFrame `df` and assign it to the variable `y`. This will be our label.
# * Get the column `Review` from DataFrame `df` and assign it to the variable `X`. This will be our feature.
# 

# In[5]:


y = df['Positive Review']
X = df['Review']


# In[6]:


X.head


# In[7]:


X.shape


# ### Split Labeled Examples into Training and Test Sets
# 
# <b>Task</b>: In the code cell below create training and test sets out of the labeled examples. 
# 
# 1. Use scikit-learn's `train_test_split()` function to create the data sets.
# 
# 2. Specify:
#     * A test set that is 20 percent (.20) of the size of the data set.
#     * A seed value of '1234'. 

# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1234)


# ## Part 3: Implement TF-IDF Vectorizer to Transform Text

# <b>Task</b>: Complete the code in the cell below to implement a TF-IDF transformation on the training and test data.
# Use the "Transforming Text For a Classifier" demo as a guide. Follow the following steps:
# 
# 1. Create a `TfidfVectorizer` object and save it to the variable `tfidf_vectorizer`.
# 2. Call `tfidf_vectorizer.fit()` to fit the vectorizer to the training data `X_train`.
# 3. Call the `tfidf_vectorizer.transform()` method to use the fitted vectorizer to transform the training data `X_train`. Save the result to `X_train_tfidf`.
# 4. Call the `tfidf_vectorizer.transform()` method to use the fitted vectorizer to transform the test data `X_test`. Save the result to `X_test_tfidf`.

# In[9]:


# 1. Create a TfidfVectorizer object and save it to the variable 'tfidf_vectorizer'
tfidf_vectorizer = TfidfVectorizer()

# 2. Fit the vectorizer to X_train
tfidf_vectorizer.fit(X_train)

# 3. Using the fitted vectorizer, transform the training data and save the data to variable 'X_train_tfidf'
X_train_tfidf = tfidf_vectorizer.transform(X_train)

# 4. Using the fitted vectorizer, transform the test data and save the data to variable 'X_test_tfidf'
X_test_tfidf = tfidf_vectorizer.transform(X_test)



# In[10]:


print(X_test_tfidf)


# ## Part 4: Fit a Logistic Regression Model to the Transformed Training Data and Evaluate the Model
# <b>Task</b>: Complete the code cell below to train a logistic regression model using the TF-IDF features, and compute the AUC on the test set.
# 
# Follow the following steps:
# 
# 1. Create the `LogisticRegression` model object below and assign to variable `model`. Supply `LogisticRegression()` the following argument: `max_iter=200`.
# 2. Fit the logistic regression model to the transformed training data (`X_train_tfidf` and `y_train`).
# 3. Use the predict_proba() method to make predictions on the test data (`X_test_tfidf`). Save the second column to the variable `probability_predictions`. 
# 4. Use the `roc_auc_score()` function to compute the area under the ROC curve for the test data. Call the
# function with the arguments `y_test` and `probability_predictions`. Save the result to the variable `auc`.
# 5. The 'vocabulary_' attribute of the vectorizer (`tfidf_vectorizer.vocabulary_`) returns the feature space. It returns a dictionary; find the length of the dictionary to get the size of the feature space. Save the result to `len_feature_space`.

# In[11]:


# 1. Create the LogisticRegression model object with max_iter=200
model = LogisticRegression(max_iter=200)

# 2. Fit the model to the transformed training data
model.fit(X_train_tfidf, y_train)

# 3. Use the predict_proba() method to make predictions on the test data
probability_predictions = model.predict_proba(X_test_tfidf)[:, 1]

# 4. Compute the area under the ROC curve for the test data
auc = roc_auc_score(y_test, probability_predictions)

print('AUC on the test data: {:.4f}'.format(auc))

# 5. Compute the size of the resulting feature space
len_feature_space = len(tfidf_vectorizer.vocabulary_)

print('The size of the feature space: {}'.format(len_feature_space))


# ## Part 5: Experiment with Different Document Frequency Values and Analyze the Results

# <b>Task</b>: The cell below will loop over a range of 'document frequency' values. For each value, it will fit a vectorizer specifying `ngram_range=(1,2)`. It will then fit a logistic regression model to the transformed data and evaluate the results.   
# 
# Complete the loop in the cell below by 
# 
# 1. adding a list containing four document frequency values that you would like to use (e.g. `[1, 10, 100, 1000]`)
# 2. adding the code you wrote above inside the loop. 
# 
# Note: This may take a short while to run.

# In[18]:


for min_df in [1, 10, 100, 1000]: # YOUR CODE HERE (add list of four values here): 
    
    print('\nDocument Frequency Value: {0}'.format(min_df))

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df)

    # 2. Fit the vectorizer to X_train
    tfidf_vectorizer.fit(X_train)

    # 3. Using the fitted vectorizer, transform the training data.
    # Save the transformed training data to variable 'X_train_tfidf'
    X_train_tfidf = tfidf_vectorizer.transform(X_train)

    # 4. Using the fitted vectorizer, transform the test data.
    # Save the transformed test data to variable 'X_test_tfidf'
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # 5. Create the LogisticRegression model object and save it to variable 'model'.
    # Call LogisticRegression() with the argument 'max_iter=200'
    model = LogisticRegression(max_iter=200)

    # 6. Fit the model to the transformed training data
    model.fit(X_train_tfidf, y_train)

    # 7. Use the predict_proba() method to make predictions on the transformed test data.
    # Save the second column to the variable 'probability_predictions'
    probability_predictions = model.predict_proba(X_test_tfidf)[:, 1]

    # 8. Using roc_auc_score() function to compute the AUC.
    # Save the result to the variable 'auc'
    auc = roc_auc_score(y_test, probability_predictions)
    
    
    print('AUC on the test data: {:.4f}'.format(auc))
    

    # 9. Compute the size of the resulting feature space. 
    # Save the result to the variable 'len_feature_space'
    len_feature_space = len(tfidf_vectorizer.vocabulary_)


    print('The size of the feature space: {0}'.format(len_feature_space))



# <b>Task</b>: Which document frequency value and feature space produced the best performing model? Do you notice any patterns regarding the number of document frequency values, the feature space and the AUC? Record your findings in the cell below.

# The document frequency value of 1 and the feature space with 143,560 unique words produced the best performing model for this particular dataset and task. Selecting an appropriate document frequency value is crucial for balancing the trade-off between including informative words and reducing the dimensionality of the feature space for efficient and accurate model training.

# ## Part 6: Set up a TF-IDF + Logistic Regression Pipeline
# 
# We will look at a new way to chain together various methods to automate the machine learning workflow. We will use  the scikit-learn `Pipeline` utility. For more information, consult the online [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). First, let's import `Pipeline`.

# In[19]:


from sklearn.pipeline import Pipeline


# The code cell below will use a scikit-learn pipeline to perform TF-IDF vectorization and the fitting of a logistic regression model to the transformed data.
# 
# This will be implemented in the following steps:
# 
# 1. First we will create a list containing the steps to perform in the pipeline. Items in the list will be executed in the order in which they appear.
# 
#     Each item in the list is a tuple consisting of two items: 
#     1. A descriptive name of what is being performed. You can create any name you'd like.
#     2. The code to run.
#     
#     
# 2. Next we will create a Pipeline object and supply it the list of steps using the `step` parameter
# 
# 
# 3. We will use this pipeline as we would any model object and fit this pipeline to the original training data. Note that when calling the `fit()` method on the pipeline object, all of the steps in the pipeline are performed on the data.
# 
# 
# 4. Finally, we will use pipeline object to make predictions on the original test data. When calling the `predict_proba()` method on the pipeline object, all of the steps in the pipeline are performed on the data. 
# 
# 
# <b>Task:</b> In the code cell below, complete step 3 and 4 using the pipeline object  `model_pipeline`.

# In[20]:


print('Begin ML pipeline...')

# 1. Define the list of steps:
s = [
        ("vectorizer", TfidfVectorizer(ngram_range=(1,2), min_df=10)),
        ("model", LogisticRegression(max_iter=200))
    ]

# 2. Define the pipeline:
model_pipeline = Pipeline(steps=s)

# We can use the pipeline the way would would use a model object 
# when fitting the model on the training data and testing on the test data:

# 3. Fit the pipeline to the training data
model_pipeline.fit(X_train, y_train)

# 4. Make predictions on the test data
# Save the second column to the variable 'probability_predictions'
probability_predictions = model_pipeline.predict_proba(X_test)[:, 1]

print('End pipeline')


# Let's compare the performance of our model. 
# 
# <b>Task</b>: In the code cell below, call the function `roc_auc_score()` with arguments `y_test` and `probability_predictions`. Save the results to the variable `auc_score`.
# 

# In[21]:


# Evaluate the performance by computing the AUC

auc_score = roc_auc_score(y_test, probability_predictions)

print('AUC on the test data: {:.4f}'.format(auc_score))


# In some case, scikit-learn gives you the ability to provide a pipeline object as an argument to a function. One such function is `plot_roc_curve()`. You'll see in the online [documentation](https://scikit-learn.org/0.23/modules/generated/sklearn.metrics.plot_roc_curve.html) that this function can take a pipeline (estimator) as an argument. Calling `plot_roc_curve()` with the pipeline and the test data will accomplish the same tasks as steps 3 and 4 in the code cell above.
# 
# Let's import the function and try it out.
# 
# <b>Task:</b> Call `plot_roc_curve()` with the following three arguments:
# 1. The pipeline object `model_pipeline`
# 2.  `X_test`
# 3. `y_test`

# In[22]:


from sklearn.metrics import plot_roc_curve

plot_roc_curve(model_pipeline, X_test, y_test)
plt.title('ROC Curve')
plt.show()


# Note that in newer versions of scikit-learn, this function has been replaced by [RocCurveDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html).

# ## Part 7: Perform a GridSearchCV on the Pipeline to Find the Best Hyperparameters 
# 

# You will perform a grid search on the pipeline object `model_pipeline` to find the hyperparameter configuration for hyperparameter $C$ (for the logistic regression) and for the $ngram\_range$ (for the TF-IDF vectorizer) that result in the best cross-validation score.
# 
# <b>Task:</b> Define a parameter grid to pass to `GridSearchCV()`. Recall that the parameter grid is a dictionary. Name the dictionary `param_grid`.
# 
# The dictionary should contain two key value pairs:
# 
# 1. a key specifying the  $C$ hyperparameter name, and a value containing the list `[0.1, 1, 10]`.
# 2. a key specifying the $ngram\_range$ hyperparameter name, and a value containing the list `[(1,1), (1,2)]`.
# 
# Note that following:
# 
# When running a grid search on a pipelines, the hyperparameter names you specify in the parameter grid are the names of the pipeline items (the descriptive names you provided to the items in the pipeline) followed by two underscores, followed by the actual hyperparameter names. 
# 
# For example, note what we named the pipeline items above:
# 
# ```
# s = [
#         ("vectorizer", TfidfVectorizer(ngram_range=(1,2), min_df=10)),
#         ("model", LogisticRegression(max_iter=200))
#     ]
# ```
# 
# We named the the classifier `model` and the vectorizer `vectorizer`. 
# 
# Since we named our classifier `model`, the hyperparameter name for $C$ that you would specify as they key in `param_grid` is `model__C`. You can find a list containing possible pipeline hyperparameter names you can use by running the code the cell below.

# In[23]:


model_pipeline.get_params().keys()


# In[24]:


param_grid = {
    'model__C': [0.1, 1, 10],  # Hyperparameter C for LogisticRegression
    'vectorizer__ngram_range': [(1, 1), (1, 2)]  # Hyperparameter ngram_range for TfidfVectorizer
}


# <b>Task:</b> Run a grid search on the pipeline.
# 
# 1. Call `GridSearchCV()` with the following arguments:
# 
#     1. Pipeline object `model_pipeline`.
#     2. Parameter grid `param_grid`.
#     3. Specify 3 cross validation folds using the `cv` parameter.
#     4. Specify that the scoring method is `roc_auc` using the `scoring` parameter.
#     5. To monitor the progress of the grid search, supply the argument `verbose=2`.
#     
#     Assign the output to the object `grid`.
#     
#     
# 2. Fit `grid` on the training data (`X_train` and `y_train`) and assign the result to variable `grid_search`.
# 
# 

# In[25]:


print('Running Grid Search...')

# 1. Run a Grid Search with 3-fold cross-validation and assign the output to the 
# object 'grid_LR'.

grid = GridSearchCV(model_pipeline, param_grid, cv=3, scoring='roc_auc', verbose=2)

# 2. Fit the model (grid) on the training data and assign the fitted model to the variable 'grid_search'.
grid_search = grid.fit(X_train, y_train)


print('Done')


# Run the code below to see the best pipeline configuration that was determined by the grid search.

# In[26]:


grid_search.best_estimator_


# <b>Task</b>: Print the best hyperparameters by accessing them by using the `best_params_` attribute.

# In[27]:


print(grid_search.best_params_)


# Recall that in the past, after we obtained the best hyperparameter values from a grid search, we re-trained a model with these values in order to evaluate the performance. This time we will do something different. Just as we can pass a pipeline object directly to `plot_roc_curve()` to evaluate the model, we can pass `grid_search.best_estimator_` to the function `plot_roc_curve()` to evaluate the model. We also pass in the test data (`X_test` and `y_test`). This allows the test data to be passed through the entire pipeline, using the best hyperparameter values.
# 
# 
# <b>Task</b>: In the code cell below plot the ROC curve and compute the AUC by calling the function `plot_roc_curve()` with the arguments `grid_search.best_estimator_` and the test data (`X_test` and  `y_test`). Note that you can simply just pass `grid_search` to the function as well.

# In[28]:


plot_roc_curve(grid_search, X_test, y_test)
plt.title('ROC Curve')
plt.show()


# In[ ]:




