#!/usr/bin/env python
# coding: utf-8

# # Lab 2: Building a Modeling Data Set

# In[50]:


import os
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_theme()


# In this lab, you will complete the following tasks to <b>build a modeling data set</b>:
# 
# 1. <b>Load the Airbnb "listings" data set</b> and identify the <b>number of rows & columns</b>
# 2. <b>Remove features</b> that are not currently useful for analysis; <br>
# <b>Modify features</b> to make sure they are machine-comprehensible
# 3. Build a new regression <b>label column</b> by winsorizing outliers
# 4. Replace all <b>missing values</b> with means
# 5. <b>Identify</b> two features with the <b>highest correlation with the label</b>
# 6. Build appropriate <b>bivariate plots</b> between the highest correlated features and the label
# 

# ## Part 1. Load the Data

# We will once again be working with the Airbnb NYC "listings" data set. Use the specified path and name of the file to load the data. Save it as a Pandas DataFrame called `df`.

# In[51]:


# Do not remove or edit the line below:
filename = os.path.join(os.getcwd(), "data", "listings.csv.gz")


# **Task**: load the data and save it to DataFrame `df`.

# In[52]:


df = pd.read_csv(filename, low_memory=False)


# <b>Task</b>: Display the shape of `df` -- that is, the number of rows and columns.

# In[53]:


print(df.shape)


# **Task**: Get a peek at the data by displaying the first few rows, as you usually do.

# In[54]:


df.head()


# ## Part 2. Feature Selection and Engineering

# We won't need the data fields that contain free, unstructured text. For example, we wont need the columns that contain apartment descriptions supplied by the host, customer reviews, or descriptions of the neighborhoods in which a listing is located.

# The code cell below contains a list containing the names of *unstructured text* columns.<br>
# 

# In[55]:


unstr_text_colnames = ['description', 'name', 'neighborhood_overview', 'host_about', 'host_name', 'host_location']


# **Task**: Drop the columns with the specified names, *in place* (that is, make sure this change applies to the original DataFrame `df`, instead of creating a temporary new DataFrame with fewer columns).

# In[56]:


unstr_text_colnames = ['description', 'name', 'neighborhood_overview', 'host_about', 'host_name', 'host_location']
df.drop(columns=unstr_text_colnames, inplace=True)



# **Task**: Display the shape of the data to verify that the new number of columns is what you expected.

# In[57]:


print(df.shape)


# We will furthermore get rid of all the columns which contain website addresses (URLs).<br>
# 
# **Task**: Create a list which contains the names of columns that contain URLs.<br> Save the resulting list to variable `url_colnames`.
# 
# *Tip*: There are different ways to accomplish this, including using Python list comprehensions

# In[58]:


url_colnames = [col for col in df.columns if any(url in str(col) for url in ['http', 'www'])]



# **Task**: Drop the columns with the specified names contained in list `url_colnames` in place (that is, make sure this change applies to the original DataFrame `df`, instead of creating a temporary new DataFrame object with fewer columns).

# In[59]:


df.drop(columns=url_colnames, inplace=True)


# **Task**: Another property of this dataset is that the `price` column contains values that are listed as <br>$<$currency_name$>$$<$numeric_value$>$. For example, it contains values that look like this: `$120`. <br>
# 
# Let's look at the first 15 unique values of this column.<br>
# 
# Display the first 15 unique values of  the `price` column:

# In[60]:


print(df['price'].unique()[:15])


# In order for us to use the prices for modeling, we will have to transform all values of this `price` feature into regular floats.<br>
# We will first need to remove the dollar signs (in this case, the platform forces the currency to be the USD, so we do not need to worry about targeting, say, the Japanese Yen sign, nor about converting the values into USD). Furthermore, we need to remove commas from all values that are in the thousands or above: for example, `$2,500$`. Here is how to do both:

# In[61]:


df['price'] = df['price'].str.replace(',', '')
df['price'] = df['price'].str.replace('$', '')
df['price'] = df['price'].astype(float)


# Let's display the first few unique values again, to make sure they are transformed:

# In[62]:


df['price'].unique()[:15]


# Well done! Our transformed dataset looks like this:

# In[63]:


df.head()


# ## Part 3. Create a (Winsorized) Label Column

# Assume that your goal is to use this dataset to fit a regression model that predicts the price under which a given space is listed.

# **Task**: Create a new version of the `price` column, named `label_price`, in which we replace the top and bottom 1% outlier values with the corresponding percentile value. Add this new column to the DataFrame `df`.

# Remember, you will first need to load the `stats` module from the `scipy` package:

# In[64]:


from scipy import stats


# In[65]:


var1 = df['price'].quantile(0.01)
var2 = df['price'].quantile(0.99)
df['label_price'] = df['price'].clip(var1, var2)


# Let's verify that a new column got added to the DataFrame:

# In[66]:


df.head()


# **Task**: Check that the values of `price` and `label_price` are *not* identical. Do this by subtracting the two columns and printing the *length* of the array (using the `len()` function) of *unique values* of the resulting difference. <br>Note: If all values are identical, the difference would contain only one unique value -- zero. If this is the case, outlier removal did not work.

# In[67]:


diff = df['price'] - df['label_price']
diff2 = len(diff.unique())
print(diff2)
print(df['price'])


# ## Part 4. Replace the Missing Values With Means

# ### a. Identifying missingness

# **Task**: Check if a given value in any data cell is missing, and sum up the resulting values (`True`/`False`) by columns. Save this sum to variable `nan_count`. Print the results.

# In[68]:


nan_count = df.isnull().sum()
nan_count


# Those are more columns than we can eyeball! For this exercise, we don't care about the number of missing values -- we just want to get a list of columns that have *any*. <br>
# 
# <b>Task</b>: From variable `nan_count`, create a new series called `nan_detected` that contains `True`/`False` values that indicate whether the number of missing values is *not zero*:

# In[69]:


nan_detected = nan_count != 0
nan_detected


# Since replacing the missing values with the mean only makes sense for the numerically valued columns (and not for strings, for example), let us create another condition: the *type* of the column must be `int` or `float`.

# **Task**: Create a series that contains `True` if the type of the column is either `int64` or `float64`. Save the result to variable `is_int_or_float`.

# In[70]:


is_int_or_float = df.dtypes.isin([int, float])
is_int_or_float


# <b>Task</b>: Combine the two binary series values into a new series named `to_impute`. It will contain the value `True` if a column contains missing values *and* is of type 'int' or 'float'

# In[71]:


to_impute = nan_detected & is_int_or_float
to_impute


# Finally, let's display a list that contains just the selected column names:

# In[72]:


df.columns[to_impute]


# We just identified and displayed the list of candidate columns for potentially replacing missing values with the column mean.

# Assume that you have decided that it is safe to impute the values for `host_listings_count`, `host_total_listings_count`, `bathrooms`, `bedrooms`, and `beds`:

# In[73]:


to_impute_selected = ['host_listings_count', 'host_total_listings_count', 'bathrooms',
       'bedrooms', 'beds']


# ### b. Keeping record of the missingness: creating dummy variables 

# As a first step, you will now create dummy variables indicating missingness of the values.

# **Task**: Store the `True`/`False` series that indicate missingness of any value in a given column as a new variable called<br> `<original-column-name>_na`. 

# In[78]:


for colname in to_impute_selected:
    varName = colname + "_na"
    df[varName] = df[colname].isnull()


# Check that the DataFrame contains the new variables:

# In[79]:


df.head()


# ### c. Replacing the missing values with mean values of the column

# **Task**: Fill the missing values of the selected few columns with the corresponding mean value.

# In[80]:


for colname in to_impute_selected:
    meanVal = df[colname].mean()
    df[colname].fillna(meanVal, inplace=True)


# Check your results below. The code displays the count of missing values for each of the selected columns. 

# In[81]:


for colname in to_impute_selected:
    print("{} missing values count :{}".format(colname, np.sum(df[colname].isnull(), axis = 0)))


# Why did the `bathrooms` column retain missing values after our imputation?

# **Task**: List the unique values of the `bathrooms` column.

# In[82]:


uniqueVals = df['bathrooms'].unique()
print(uniqueVals)


# The column did not contain a single value (except the `NaN` indicator) to begin with.

# ## Part 5. Identify Features With the Highest Correlation With the Label

# Your next goal is to figure out which features in the data correlate most with the label.<br>
# 
# In the next few cells, we will demonstrate how to use the Pandas `corr()` method to get a list of correlation coefficients between `label` and all other (numerical) features.

# Let's first glance at what the `corr()` method does:

# In[83]:


df.corr().head()


# The result is a computed *correlation matrix*. The values on the diagonal are all equal to 1, and the matrix is symmetrical with respect to the diagonal (note that we are only printing the first five lines of it).<br>
# 
# We only need to observe correlations of all features with *the label* (as opposed to every possible pairwise correlation). <br>
# 
# **Task**: Save the `label_price` column of the correlation matrix to the variable `corrs`:

# In[84]:


corrs = df.corr()['label_price']
corrs


# **Task**: Sort the values of the series we just obtained in the descending order.

# In[85]:


corrs_sorted = corrs.sort_values(ascending=False)
corrs_sorted


# **Task**: In the code cell below, save the *column names* for the top-2 correlation values to the list `top_two_corr` (not counting the correlation of `label` column with itself, nor the `price` column -- which is the `label` column prior to outlier removal). Add the column names to the list in the order in which they appear in the output above. <br>
# Tip: `corrs_sorted` is a Pandas `Series` object, in which column names are the *index*.

# In[86]:


top_two_corr = corrs_sorted.index[1:3].tolist()
top_two_corr


# ## Part 6. Produce Bivariate Plots for the Label and Its Top Correlates

# We will use the `pairplot()` function in `seaborn` to plot the relationships between the two features we identified and the label.

# **Task**: Create a DataFrame `df_sub` that contains only the selected three columns: the label, and the two columns which correlate with it the most.

# In[87]:


# Do not remove or edit the line below:
top_two_corr.append('label_price')


df_sub = df[top_two_corr]


# **Task**: Create a `seaborn` pairplot of the data subset you just created

# In[88]:


sns.pairplot(df_sub)


# This one is not very easy to make sense of: the points overlap, but we do not have visibility into how densely they are stacked together.
# <br>
# 
# **Task**: Repeat the `pairplot` exercise, this time specifying the *kernel density estimator* as the *kind* of the plot.<br>
#     Tip: use `kind = 'kde'` as a parameter of the `pairplot()` function. You could also specify `corner=True` to make sure you don't plot redundant (symmetrical) plots.
#    <br>
#    Note: this one may take a while!

# In[ ]:


sns.pairplot(df_sub, kind='kde', corner=True)


# <b>Analysis:</b> Think about the possible interpretations of these plots. (Recall that our label encodes the listing price). <br>
# What kind of stories does this data seem to be telling? Is the relationship what you thought it would be? Is there anything surprising or, on the contrary, reassuring about the plots?<br>
# For example, how would you explain the relationship between the label and 'accommodates'? Is there a slight tilt to the points cluster, as the price goes up?<br>
# What other patterns do you observe?

# <Double click this Markdown cell to make it editable, and record your findings here.>
