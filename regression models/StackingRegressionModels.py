#!/usr/bin/env python
# coding: utf-8

# # Lab 6: Stacking Regression Models

# In[1]:


import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In this lab assignment, you will:
# 
# 1. Load the Airbnb "listings" data set.
# 2. Use the stacking ensemble method to train four regressors.
# 3. Train and evaluate the same four individual regressors.
# 4. Compare the performance of the stacked ensemble model to that of the individual models.
# 
# **<font color='red'>Note: Some of the code cells in this notebook may take a while to run.</font>**

# ## Part 1: Load the Data Set
# 
# We will work with a preprocessed version of the Airbnb NYC "listings" data set. 
# 
# <b>Task</b>: In the code cell below, use the same method you have been using to load the data using `pd.read_csv()` and save it to DataFrame `df`.
# 
# You will be working with the file named "airbnb_readytofit.csv.gz" that is located in a folder named "data".

# In[2]:


filename = os.path.join(os.getcwd(), "data", "airbnb_readytofit.csv.gz")
df = pd.read_csv(filename, header=0)


# ## Part 2: Create Training and Test Data Sets

# So far, we mostly focused on classification problems. For this exercise, you will focus on a regression problem and predict a continuous outcome.
# 
# Your model will predict the price of a listing; the label is going to be 'price'.
# 
# ### Create Labeled Examples 
# 
# <b>Task</b>: Create labeled examples from DataFrame `df`. 
# In the code cell below carry out the following steps:
# 
# * Get the `price` column from DataFrame `df` and assign it to the variable `y`. This will be our label.
# * Get all other columns from DataFrame `df` and assign them to the variable `X`. These will be our features. 

# In[3]:


y = df['price']
X = df.drop(columns=['price'])


# ### Split Labeled Examples Into Training and Test Sets
# 
# <b>Task</b>: In the code cell below, create training and test sets out of the labeled examples. 
# 
# 1. Use scikit-learn's `train_test_split()` function to create the data sets.
# 
# 2. Specify:
#     * A test set that is 30 percent of the size of the data set.
#     * A seed value of '1234'. 
#     

# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1234)


# ## Part 3: Use the Stacking Ensemble Method to Train Four Regression Models and Evaluate the Performance

# You will use the scikit-learn `StackingRegressor` class. For more information, consult the online [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html).
# 
# First let's import `StackingRegressor`:

# In[5]:


from sklearn.ensemble import StackingRegressor


# In this part of the assignment, we will try to use four models jointly. In the code cell below, we creates a list of tuples, each consisting of a scikit-learn model function and the corresponding shorthand name that we choose:

# In[6]:


estimators = [("DT", DecisionTreeRegressor()),
              ("RF", RandomForestRegressor()),
              ("GBDT", GradientBoostingRegressor()),
              ("LR", LinearRegression())
             ]


# <b>Task</b>: Call `StackingRegressor()` with the following parameters:
# 
# 1. Assign the list `estimators` to the parameter `estimators`.
# 2. Specify a 5-fold cross-validation using the parameter `cv`.
# 3. Use the parameter 'passthrough=False'. 
# 
# Assign the results to the variable `stacking_model`.
# 
# As you read up on the definition of the `StackingRegressor` class, you will notice that by default, the results of each model are combined using a ridge regression (a "final regressor").

# In[7]:


stacking_model = StackingRegressor(
    estimators=estimators,
    cv=5,
    passthrough=False
)


# Let's train and evaluate this ensemble model using cross-validation:

# <b>Task</b>: Use scikit-learn's `cross_val_score()` function on the `stacking_model` model to obtain the 3-fold cross-validation RMSE scores. In the code cell below, perform the following steps:
# 
# 1. Call the function with the following arguments:
# 
#     1. your model object 
#     2. your training data 
#     3. specify the number of folds 
#     4. specify the "scoring method": `scoring = 'neg_root_mean_squared_error'`
# 
# 2. Compute the average RMSE score returned by the 3-fold cross-validation and save the result to `rmse_avg`(Recall that specifying `neg_root_mean_squared_error` will result in negative RMSE values, so you have to multiply each value by -1 to obtain the RMSE scores before obtaining the average RMSE).
# 
# <b>Note</b>: This may take a while to run.

# In[8]:


print('Performing Cross-Validation...')

scores = cross_val_score(stacking_model, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error')
rmse_scores = -scores
rmse_avg = np.mean(rmse_scores)

print('End')
print('average score: {}'.format(rmse_avg))


# <b>Analysis</b>: 
# 1. Does the stacking model perform well? <br>
# 2. Which hyperparameters were used for each one of the models in the stack?<br>
# 
# Record your findings in the cell below.

# <Double click this Markdown cell to make it editable, and record your findings here.>
# 
# <b> ANALYSIS RESPONSE </b>
# 1. The RMSE score, or the error, is 0.645, which is a very low error! Therefore, the stacking model is performing well, with a better predictive performance and a low average error.
# 
# 2. The hyperparameters are as follows:
#     DT: max_depth=8, min_samples_leaf=50
#     RF: max_depth=32, n_estimators_300
#     GBDT:max_depth=2, n_estimators=300
#     LR: No hyperparameters

# ## Part 4: Improve the Performance of the Ensemble Model
# 
# Assume that you decided to further improve your model by tuning a few of the hyperparameters and finding the best ones. Do not run the code cell below, but simply analyze the code:

# In[9]:


"""
params = {
    "DT__max_depth": [2, 4, 8],
    "GBDT__n_estimators":[100,300]
    
}

stack_grid = GridSearchCV(stacking, params, cv=3, verbose=4, scoring='neg_root_mean_squared_error', refit=True, n_jobs=-1)
stack_grid.fit(X_train, y_train)
print(stack_grid.best_params_)
rf_grid.cv_results_['mean_test_score']


print("best parameters:", rf_grid.best_params_)

rmse_stack_cv = -1*rf_grid.best_score_
print("[STACK] RMSE for the best model is : {:.2f}".format(rmse_stack_cv))

"""


# Running the code above is computationally costly (you are welcome to do so on your own time as an ungraded activity). For this lab, we will simply give away the resulting values of the best hyperparameters:<br>
# ```{'DT__max_depth': 8, 'GBDT__n_estimators': 100}```

# <b>Task</b>: Create a new version of the 'estimators' list. You will use the same four regressors, but this time, you will pass the `max_depth` value above to the decision tree model, and the `n_estimators` value above to the gradient boosted decision tree. Save the estimators list to the variable `estimators_best`.

# In[10]:


best_max_depth = 8
best_n_estimators = 100

estimators_best = [("DT", DecisionTreeRegressor(max_depth=best_max_depth)),
              ("RF", RandomForestRegressor()),
              ("GBDT", GradientBoostingRegressor(n_estimators=best_n_estimators)),
              ("LR", LinearRegression())
             ]


# <b>Task</b>: Create a new `StackingRegressor` object with `estimators_best`. Name the model object `stacking_best_model`. Fit `stacking_best_model` to the training data.
# 

# In[11]:


print('Implement Stacking...')

stacking_best_model = StackingRegressor(estimators=estimators_best, cv=5, passthrough=False)
stacking_best_model.fit(X_train, y_train)

print('End')


# <b>Task:</b> Use the `predict()` method to test your ensemble model `stacking_best_model` on the test set (`X_test`). Save the result to the variable `stacking_best_pred`. Evaluate the results by computing the RMSE and R2 score. Save the results to the variables `rmse` and `r2`.
# 
# Complete the code in the cell below to accomplish this.

# In[12]:


# 1. Use predict() to test use the fitted model to make predictions on the test data
# YOUR CODE HERE
stacking_best_pred = stacking_best_model.predict(X_test)

# 2. Compute the RMSE using mean_squared_error()
rmse = mean_squared_error(y_test, stacking_best_pred, squared=False)

# 3. Compute the R2 score using r2_score()
r2 = r2_score(y_test, stacking_best_pred)

           
print('Root Mean Squared Error: {0}'.format(rmse))
print('R2: {0}'.format(r2))                       


# ## Part 5: Fit and Evaluate Individual Regression Models

# ### a. Fit and Evaluate a Linear Regression
# 
# <b>Task:</b> Complete the code below to fit and evaluate a linear regression model:

# In[13]:


# 1. Create the LinearRegression model object below and assign to variable 'lr_model'
lr_model = LinearRegression()

# 2. Fit the model to the training data below
lr_model.fit(X_train, y_train)

# 3.  Call predict() to use the fitted model to make predictions on the test data. Save the results to variable
# 'y_lr_pred'
y_lr_pred = lr_model.predict(X_test)

# 4: Compute the RMSE and R2 (on y_test and y_lr_pred) and save the results to lr_rmse and lr_r2
# YOUR CODE HERE
lr_rmse = mean_squared_error(y_test, y_lr_pred, squared=False)
lr_r2 = r2_score(y_test, y_lr_pred)

print('[LR] Root Mean Squared Error: {0}'.format(lr_rmse))
print('[LR] R2: {0}'.format(lr_r2))


# ### b. Fit and Evaluate a Decision Tree 
# 
# Let's assume you already performed a grid search to find the best model hyperparameters for your decision tree. (We are omitting this step to save computation time.) The best values are: `max_depth=8`, and `min_samples_leaf = 50`. You will train a decision tree with these hyperparameter values.
# 
# <b>Task:</b> Complete the code in the cell below:

# In[14]:


# 1. Create the DecisionTreeRegressor model object using the hyperparameter values above and assign to 
# variable 'dt_model'
# YOUR CODE HERE
dt_model = DecisionTreeRegressor(max_depth=8, min_samples_leaf=50)

# 2. Fit the model to the training data below
# YOUR CODE HERE
dt_model.fit(X_train, y_train)

# 3.  Call predict() to use the fitted model to make predictions on the test data. Save the results to variable
# 'y_dt_pred'
y_dt_pred = dt_model.predict(X_test)

# 4: Compute the RMSE and R2 (on y_test and y_dt_pred) and save the results to dt_rmse and dt_r2
# YOUR CODE HERE
dt_rmse = mean_squared_error(y_test, y_dt_pred, squared=False)
dt_r2 = r2_score(y_test, y_dt_pred)

print('[DT] Root Mean Squared Error: {0}'.format(dt_rmse))
print('[DT] R2: {0}'.format(dt_r2))


# ### c. Fit and Evaluate a Gradient Boosted Decision Tree 
# 
# Let's assume you already performed a grid search to find the best model hyperparameters for your gradient boosted decision tree. (We are omitting this step to save computation time.) The best values are: `max_depth=2`, and `n_estimators = 300`. You will train a GBDT with these hyperparameter values.
# 
# <b>Task</b>: Complete the code in the cell below.

# In[15]:


print('Begin GBDT Implementation...')

# 1. Create the  GradientBoostingRegressor model object below and assign to variable 'gbdt_model'
best_max_depth = 2
best_n_estimators = 300
gbdt_model = GradientBoostingRegressor(max_depth=2, n_estimators=300)

# 2. Fit the model to the training data below
# YOUR CODE HERE
gbdt_model.fit(X_train, y_train)

# 3. Call predict() to use the fitted model to make predictions on the test data. Save the results to variable
# 'y_gbdt_pred'
# YOUR CODE HERE
y_gbdt_pred = gbdt_model.predict(X_test)

# 4. Compute the RMSE and R2 (on y_test and y_gbdt_pred) and save the results to gbdt_rmse and gbdt_r2
# YOUR CODE HERE
gbdt_rmse = mean_squared_error(y_test, y_gbdt_pred, squared=False)
gbdt_r2 = r2_score(y_test, y_gbdt_pred)


print('End')

print('[GBDT] Root Mean Squared Error: {0}'.format(gbdt_rmse))
print('[GBDT] R2: {0}'.format(gbdt_r2))                 




# ### d. Fit and Evaluate  a Random Forest
# 
# Let's assume you already performed a grid search to find the best model hyperparameters for your random forest model. (We are omitting this step to save computation time.) The best values are: `max_depth=32`, and `n_estimators = 300`. 
# You will train a random forest with these hyperparameter values.
# 
# <b>Task</b>: Complete the code in the cell below.

# In[16]:


print('Begin RF Implementation...')

# 1. Create the  RandomForestRegressor model object below and assign to variable 'rf_model'
best_max_depth = 32
best_n_estimators = 300
rf_model = RandomForestRegressor(max_depth=best_max_depth, n_estimators=best_n_estimators)

# 2. Fit the model to the training data below
rf_model.fit(X_train, y_train)

# 3. Call predict() to use the fitted model to make predictions on the test data. Save the results to variable
# 'y_rf_pred'
y_rf_pred = rf_model.predict(X_test)

# 4. Compute the RMSE and R2 (on y_test and y_rf_pred) and save the results to rf_rmse and rf_r2
rf_rmse = mean_squared_error(y_test, y_rf_pred, squared=False)
rf_r2 = r2_score(y_test, y_rf_pred)


print('End')

print('[RF] Root Mean Squared Error: {0}'.format(rf_rmse))
print('[RF] R2: {0}'.format(rf_r2))


# ## Part 6: Visualize Model Performance
# 
# The code cell below will plot the RMSE and R2 score for the stacked ensemble model and each regressor. 
# 
# <b>Task:</b> Complete the code in the cell below.

# In[20]:


RMSE_Results = [rmse, lr_rmse, dt_rmse, gbdt_rmse, rf_rmse]
R2_Results = [r2, lr_r2, dt_r2, gbdt_r2, rf_r2]

rg= np.arange(5)
width = 0.35

# 1. Create bar plot with RMSE results
# YOUR CODE HERE
plt.bar(rg, RMSE_Results, width, label='RMSE')
plt.xlabel('Models')
plt.ylabel('RMSE and R2')

# 2. Create bar plot with R2 results
plt.bar(rg + width, R2_Results, width, label='R2')

# 3. Call plt.xticks() to add labels under the bars indicating which model the pair of RMSE 
# and R2 bars correspond to
plt.xticks(rg+width/2, ['Stacked', 'LR', 'DT', 'GBDT', 'RF'])

# 4. Label the x and y axis of the plot: the x axis should be labeled "Models" and the y axis
# should be labeled "RMSE and R2"
plt.xlabel('Models')
plt.ylabel('RMSE and R2')

plt.ylim([0,1])
plt.title('Model Performance')
plt.legend(loc='upper left', ncol=2)
plt.show()


# <b>Analysis</b>: Compare the performance of the stacking model with the individual models. Is the stacking model performing better?
# Now that you are familiar with the Airbnb data, think about how a regression for price could be improved. What would you change, either at the feature engineering stage, or in the model selection, or at the stage of hyperparameter tuning?
# Record your findings in the cell below.

# <Double click this Markdown cell to make it editable, and record your findings here.>
# 
# <b> ANALYSIS RESPONSE </b>
# 
# The stacking model performed well, with the second improved version resulting in a low RMSE, and even against the other models, resulted in the lowest RMSE of them all. Along with that, the r2 value, which is the coefficient of determination (lets go AP stats), is the highest amongst all of the other models, meaning it is the closest to predicting it accurately! Overall, improving models can lower the error and result in better predictions. 
# 
# The easiest would be to change things around in the feature engineering stage. I would explore and extract additional relevant features from the available data. For example, we could derive new features such as the number of nearby attractions, proximity to public transportation, or amenities provided by the host. Feature engineering helps to capture more relevant information and improve model performance. I could also take a look at the data that we are inputing by handling outliers in the target variable/features. Outliers can drastically impact the model's performance, and addressing them appropriately can lead to better predictions.

# In[ ]:




