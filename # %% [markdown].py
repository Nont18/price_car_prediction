# %% [markdown]
# # A1: Predicting Car Price
# 
# This data is a **regression problem**, trying to predict car price.
# 
# The followings describe the features.
# 
#  name : brandname of the car        
#  year : released year car
#  selling_price : price for buying the car
#  km_driven : kilometer that the car has been driven      
#  fuel : fuel that the car can used          
#  seller_type : someone that customer buy the car     
#  transmission : manual or anutomatic    
#  owner : the number of owners that how many people used it.          
#  mileage : a term use to express the fuel efficiency of a vehicle.       
#  engine : size of engine         
#  max_power : maximum power output that car can make.     
#  torque : a measurement of your car's ability to do work.            
#  seats : the capacity that one car can carry people.

# %% [markdown]
# ## Importing libraries

# %%
#Import the important libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')

# %%
import matplotlib
np.__version__, pd.__version__, sns.__version__, matplotlib.__version__

# %% [markdown]
# ## 1. Load data

# %%
#Load the data from path that save in computer.
df = pd.read_csv('data/car_price_dataset.csv')

# %%
# print the first rows of data
df.head()

# %%
# Check the shape of your data
df.shape

# %%
# Statistical info Hint: look up .describe()
df.describe()

# %%
# Check Dtypes of your input data
df.info()

# %%
# Check the column names
df.columns

# %% [markdown]
# ## 2. Exploratory Data Analysis
# 
# EDA is an essential step to inspect the data, so to better understand nature of the given data.

# %%
#Let's check the column. What we have in the dataset.
df.columns

# %%
# rename columns named 'name' to 'brand'
df.rename(columns = {'name':'brand' 
                     }, inplace = True)

# %%
#check again
df.columns

# %% [markdown]
# ### 2.1 Univariate analyis
# 
# Single variable exploratory data anlaysis

# %% [markdown]
# #### Countplot

# %%
# Let's see how many developing and developed countries there are
sns.countplot(data = df, x = 'seats')

# %% [markdown]
# #### Distribution plot

# %%
sns.displot(data = df, x = 'seats')

# %% [markdown]
# ### 2.2 Multivariate analysis
# 
# Multiple variable exploratory data analysis

# %% [markdown]
# #### Boxplot

# %%
# Let's try bar plot on "Status"
sns.boxplot(x = df["transmission"], y = df["selling_price"])
plt.ylabel("selling_price")
plt.xlabel("transmission")

# %% [markdown]
# #### Scatterplot

# %%
sns.scatterplot(x = df['selling_price'], y = df['year'], hue=df['transmission'])

# %% [markdown]
# #### Correlation Matrix
# 
# Let's use correlation matrix to find strong factors predicting the life expectancy.  It's also for checking whether certain features are too correlated.

# %%
# Let's check out heatmap
plt.figure(figsize = (15,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")  #don't forget these are not all variables! categorical is not here...

# %% [markdown]
# #### Tips: Label encoding
# 
# Now we would like to change "Developing" and "Developed" to "0" and "1", since machine learning algorithms do not understand text.   Also, correlation matrix and other similar computational tools require label encoding.

# %%
#For the feature owner, map First owner to 1, ..., Test Drive Car to 5
from sklearn.preprocessing import LabelEncoder
label_mapping = {
    'First Owner': 1,
    'Second Owner': 2,
    'Third Owner': 3,
    'Fourth & Above Owner': 4,
    'Test Drive Car': 5
}

categorical_data = df['owner']

df['owner'] = [label_mapping[label] for label in categorical_data]

# %%
#For the feature fuel, remove all rows with CNG and LPG because CNG and LPG use a different mileage system 
# i.e., km/kg which is different from kmfeaturepl for Diesel and Petrol
df = df[~df['fuel'].isin(['CNG', 'LPG'])]

# %%
#For the feature mileage, remove “kmpl” and convert the column to numerical type (e.g., float).
# Extract numeric mileage values by splitting and converting to float
df['mileage'] = df['mileage'].str.split().str[0].astype(float)

# %%
# Remove "CC" and convert to float
df['engine'] = df['engine'].str.replace(' CC', '').astype(float)

# %%
# Remove " bhp" and convert to float, handling N/A values
df['max_power'] = df['max_power'].str.replace(' bhp', '').astype(float)

# %%
# Extract the first word and update the column
df['brand'] = df['brand'].apply(lambda x: x.split()[0])

# %%
#convert string to numeric
label_mapping = {
    'Diesel': 1,
    'Petrol': 2
}

categorical_data1 = df['fuel']

df['lable_fuel'] = [label_mapping[label] for label in categorical_data1]

# %%
# Drop the 'torque' feature
df = df.drop(columns=['torque'])

# %%
# Remove rows with 'Test Drive Cars' in the 'make' column
df = df[df['owner'] != 5]

# %%

df['log_selling_price'] = np.log(df['selling_price'])

# %%
#Check again
df.head(45)

# %%
# Let's check out heatmap
plt.figure(figsize = (15,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")  #don't forget these are not all variables! categorical is not here...

# %% [markdown]
# #### Predictive Power Score
# 
# This is another way to check the predictive power of some feature.  Unlike correlation, `pps` actually obtained from actual prediction.  For more details:
#     
# - The score is calculated using only 1 feature trying to predict the target column. This means there are no interaction effects between the scores of various features. Note that this is in contrast to feature importance
# - The score is calculated on the test sets of a 4-fold crossvalidation (number is adjustable via `ppscore.CV_ITERATIONS`)
# - All rows which have a missing value in the feature or the target column are dropped
# - In case that the dataset has more than 5,000 rows the score is only calculated on a random subset of 5,000 rows with a fixed random seed (`ppscore.RANDOM_SEED`). You can adjust the number of rows or skip this sampling via the API. However, in most scenarios the results will be very similar.
# - There is no grid search for optimal model parameters
# 
# We can install by doing <code>pip install ppscore</code>

# %%
import ppscore as pps

# before using pps, let's drop country and year
dfcopy = df.copy()
dfcopy.drop(['year'], axis='columns', inplace=True)

#this needs some minor preprocessing because seaborn.heatmap unfortunately does not accept tidy data
matrix_df = pps.matrix(dfcopy)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')

#plot
plt.figure(figsize = (15,8))
sns.heatmap(matrix_df, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)

# %% [markdown]
# ## 3. Feature Engineering
# 
# We gonna skip for this tutorial.  But we can certainly try to combine some columsn to create new features.

# %% [markdown]
# ## 4. Feature selection

# %%
#x is our strong features
X = df[['max_power','engine','mileage','seats', 'km_driven', 'owner', 'lable_fuel']]

#y is simply the life expectancy col
y = df["log_selling_price"]

# %% [markdown]
# ### Train test split

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42) #It can be 0.2-0.4 for test_size

# %% [markdown]
# ## 5. Preprocessing

# %% [markdown]
# ### Null values

# %%
#check for null values
X_train[['max_power']].isna().sum()

# %%
X_test[['max_power']].isna().sum()

# %%
X_train[['engine']].isna().sum()

# %%
X_test[['engine']].isna().sum()

# %%
X_train[['mileage']].isna().sum()

# %%
X_test[['mileage']].isna().sum()

# %%
X_train[['seats']].isna().sum()

# %%
X_test[['seats']].isna().sum()

# %%
X_train[['km_driven']].isna().sum()

# %%
X_test[['km_driven']].isna().sum()

# %%
X_train[['owner']].isna().sum()

# %%
X_test[['owner']].isna().sum()

# %%
X_train[['lable_fuel']].isna().sum()

# %%
X_test[['lable_fuel']].isna().sum()

# %%
y_train.isna().sum()

# %%
y_test.isna().sum()

# %% [markdown]
# ##### Plot the graph then calculated mean and median for imputation. (filling missing value)

# %%
sns.displot(data=df, x='max_power')

# %%
df['max_power'].mean(),df['max_power'].median()

# %%
sns.displot(data=df, x='engine')

# %%
df['engine'].mean(),df['engine'].median()

# %%
sns.displot(data=df, x='mileage')

# %%
df['mileage'].mean(),df['mileage'].median()

# %%
sns.displot(data=df, x='seats')

# %%
df['seats'].mean(),df['seats'].median()

# %%
sns.displot(data=df, x='km_driven')

# %%
df['km_driven'].mean(),df['km_driven'].median()

# %%
sns.displot(data=df, x='owner')

# %%
df['owner'].mean(),df['owner'].median()

# %%
sns.displot(data=df, x='lable_fuel')

# %%
df['lable_fuel'].mean(),df['lable_fuel'].median()

# %%
sns.displot(y_train)

# %%
df['log_selling_price'].mean(),df['log_selling_price'].median()

# %% [markdown]
# Mean: Use the mean to fill missing values if the data is approximately normally distributed and does not have significant outliers. The mean is sensitive to extreme values, so if your data has outliers, using the mean might result in skewed imputations.
# 
# Median: Use the median to fill missing values if your data has outliers or is skewed. The median is a robust measure of central tendency and is less affected by extreme values compared to the mean.

# %%
#let's fill the training set first!

X_train['max_power'].fillna(X_train['max_power'].median(), inplace=True)
X_train['engine'].fillna(X_train['engine'].median(), inplace=True)
X_train['mileage'].fillna(X_train['mileage'].median(), inplace=True)
X_train['seats'].fillna(X_train['seats'].median(), inplace=True)
X_train['km_driven'].fillna(X_train['km_driven'].median(), inplace=True)
X_train['owner'].fillna(X_train['owner'].mean(), inplace=True)
X_train['lable_fuel'].fillna(X_train['lable_fuel'].median(), inplace=True)

# %%
#let's fill the testing set with the training distribution first!

X_test['max_power'].fillna(X_train['max_power'].median(), inplace=True)
X_test['engine'].fillna(X_train['engine'].median(), inplace=True)
X_test['mileage'].fillna(X_train['mileage'].median(), inplace=True)
X_test['seats'].fillna(X_train['seats'].median(), inplace=True)
X_test['km_driven'].fillna(X_train['km_driven'].median(), inplace=True)
X_test['owner'].fillna(X_train['owner'].mean(), inplace=True)
X_test['lable_fuel'].fillna(X_train['lable_fuel'].median(), inplace=True)

# %%
#same for y
y_train.fillna(y_train.median(), inplace=True)
y_test.fillna(y_train.median(), inplace=True)

# %%
#check again
X_train[['max_power']].isna().sum()

# %%
X_test[['max_power']].isna().sum()

# %%
X_train[['engine']].isna().sum()

# %%
X_test[['engine']].isna().sum()

# %%
X_train[['mileage']].isna().sum()

# %%
X_test[['mileage']].isna().sum()

# %%
X_train[['seats']].isna().sum()

# %%
X_test[['seats']].isna().sum()

# %%
X_train[['km_driven']].isna().sum()

# %%
X_test[['km_driven']].isna().sum()

# %%
X_train[['owner']].isna().sum()

# %%
X_test[['owner']].isna().sum()

# %%
X_train[['lable_fuel']].isna().sum()

# %%
X_test[['lable_fuel']].isna().sum()

# %%
y_train.isna().sum(), y_test.isna().sum()

# %% [markdown]
# ### Checking Outliers

# %%
# Create a dictionary of columns.
col_dict = {'max_power':1,'engine':2,'mileage':3,'seats':4, 'km_driven':5, 'owner':6, 'lable_fuel':7}

# Detect outliers in each variable using box plots.
plt.figure(figsize=(20,30))

for variable,i in col_dict.items():
                     plt.subplot(5,4,i)
                     plt.boxplot(X_train[variable])
                     plt.title(variable)

plt.show()

# %%
def outlier_count(col, data = X_train):
    
    # calculate your 25% quatile and 75% quatile
    q75, q25 = np.percentile(data[col], [75, 25])
    
    # calculate your inter quatile
    iqr = q75 - q25
    
    # min_val and max_val
    min_val = q25 - (iqr*1.5)
    max_val = q75 + (iqr*1.5)
    
    # count number of outliers, which are the data that are less than min_val or more than max_val calculated above
    outlier_count = len(np.where((data[col] > max_val) | (data[col] < min_val))[0])
    
    # calculate the percentage of the outliers
    outlier_percent = round(outlier_count/len(data[col])*100, 2)
    
    if(outlier_count > 0):
        print("\n"+15*'-' + col + 15*'-'+"\n")
        print('Number of outliers: {}'.format(outlier_count))
        print('Percent of data that is outlier: {}%'.format(outlier_percent))

# %%
for col in X_train.columns:
    outlier_count(col)

# %% [markdown]
# ### Scaling

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# feature scaling helps improve reach convergence faster
scaler = StandardScaler() # for standadization use StandardScaler() , for normalization use MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

#x = (x - mean) / std
#why do we want to scale our data before data analysis / machine learning

#allows your machine learning model to catch the pattern/relationship faster
#faster convergence

#how many ways to scale
#standardardization <====current way
# (x - mean) / std
#--> when your data follows normal distribution

#normalization <---another way
# (x - x_min) / (x_max - x_min)
#---> when your data DOES NOT follow normal distribution (e.g., audio, signal, image) We will use nomalization when mean is a bad.

# %%
# Let's check shapes of all X_train, X_test, y_train, y_test
print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of y_test: ", y_test.shape)

# %% [markdown]
# ## 6. Modeling
# 
# Let's define some algorithms and compare them using cross-validation.
# 
# [Scikit-Learn](http://scikit-learn.org) provides quick access to a huge pool of machine learning algorithms.
# 
# Before using sklearn, there is **one thing you need to know**, i.e., the **data shape that sklearn wants**.
# 
# To apply majority of the algorithms, sklearn requires two inputs, i.e., $\mathbf{X}$ and $\mathbf{y}$.
# 
# -  $\mathbf{X}$, or the **feature matrix** *typically* has the shape of ``[n_samples, n_features]``
# -  $\mathbf{y}$, or the **target/label vector** *typically* has the shape of ``[n_samples, ]`` or ``[n_samples, n_targets]`` depending whether that algorithm supports multiple labels
# 
# Note 1:  if you $\mathbf{X}$ has only 1 feature, the shape must be ``[n_samples, 1]`` NOT ``[n_samples, ]``
# 
# Note 2:  sklearn supports both numpy and pandas, as long as the shape is right.  For example, if you use pandas, $\mathbf{X}$ would be a dataframe, and $\mathbf{y}$ could be a series or dataframe.
# 
# Tips:  it's always better to look at sklearn documentation before applying any algorithm.

# %% [markdown]
# ### Much better: Cross validation + Grid search

# %% [markdown]
# To find the appropriate algorithms, We need to choose what algorithms good for prediction. This step calls Cross validation.
# Algorithms have named for using such as 
# 
# 1.Linear Regression      
# 2.SVR      
# 3.KNeighbors Regressor      
# 4.Decision-Tree Regressor      
# 5.Random-Forest Regressor     

# %%
from sklearn.linear_model import LinearRegression  #we are using regression models
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Libraries for model evaluation

# models that we will be using, put them in a list
algorithms = [LinearRegression(), SVR(), KNeighborsRegressor(), DecisionTreeRegressor(random_state = 0), 
              RandomForestRegressor(n_estimators = 100, random_state = 0)]

# The names of the models
algorithm_names = ["Linear Regression", "SVR", "KNeighbors Regressor", "Decision-Tree Regressor", "Random-Forest Regressor"]

# %%
from sklearn.linear_model import LinearRegression  #we are using regression models
from sklearn.metrics import mean_squared_error, r2_score

lr = LinearRegression()
lr.fit(X_train, y_train)
yhat = lr.predict(X_test)

print("MSE: ", mean_squared_error(y_test, yhat))
print("r2: ", r2_score(y_test, yhat))

# %% [markdown]
# Let's do some simple cross-validation here....

# %% [markdown]
# The next step is spilt the training set to 5 fold for one iteration. There will be Test set(validation set) and Training set. In each iteration, we will get mse for one value and next iteration

# %%
from sklearn.model_selection import KFold, cross_val_score

#lists for keeping mse
train_mse = []
test_mse = []

#defining splits
kfold = KFold(n_splits=5, shuffle=True)

for i, model in enumerate(algorithms):
    scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error') #Higher is better.
    print(f"{algorithm_names[i]} - Score: {scores}; Mean: {scores.mean()}")

# %% [markdown]
# Hmm...it seems random forest do very well....how about we grid search further to find the best version of the model.

# %% [markdown]
# ### Grid Search

# %%
from sklearn.model_selection import GridSearchCV

param_grid = {'bootstrap': [True], 'max_depth': [5, 10, None],
              'n_estimators': [5, 6, 7, 8, 9, 10, 11, 12, 13, 15]}

rf = RandomForestRegressor(random_state = 1)

grid = GridSearchCV(estimator = rf, 
                    param_grid = param_grid, 
                    cv = kfold, 
                    n_jobs = -1, 
                    return_train_score=True, 
                    refit=True,
                    scoring='neg_mean_squared_error')

# Fit your grid_search
grid.fit(X_train, y_train);  #fit means start looping all the possible parameters

# %%
grid.best_params_

# %%
# Find your grid_search's best score
best_mse = grid.best_score_

# %%
best_mse  # ignore the minus because it's neg_mean_squared_error

# %% [markdown]
# ## 7. Testing
# 
# Of course, once we do everything.  We can try to shoot with the final test set.  We should no longer do anything like improving the model.  It's illegal!  since X_test is the final final test set.

# %%
yhat = grid.predict(X_test)
mean_squared_error(y_test,yhat)

# %%
print(yhat)

# %%
y_pred_original = np.exp(yhat)

# %%
print(y_pred_original)

# %% [markdown]
# ## 8. Analysis:  Feature Importance
# 
# Understanding why is **key** to every business, not how low MSE we got.  Extracting which feature is important for prediction can help us interpret the results.  There are several ways: algorithm, permutation, and shap.  Note that these techniques can be mostly applied to most algorithms. 
# 
# Most of the time, we just apply all, and check the consistency.

# %% [markdown]
# #### Algorithm way
# 
# Some ML algorithms provide feature importance score after you fit the model

# %%
#stored in this variable
#note that grid here is random forest
rf = grid.best_estimator_

rf.feature_importances_

# %%
#let's plot
plt.barh(X.columns, rf.feature_importances_)

# %%
#hmm...let's sort first
sorted_idx = rf.feature_importances_.argsort()
plt.barh(X.columns[sorted_idx], rf.feature_importances_[sorted_idx])
plt.xlabel("Random-Forest Regressor Importance")

# %% [markdown]
# #### Permutation way
# 
# This method will randomly shuffle each feature and compute the change in the model’s performance. The features which impact the performance the most are the most important one.
# 
# *Note*: The permutation based importance is computationally expensive. The permutation based method can have problem with highly-correlated features, it can report them as unimportant.

# %%
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(rf, X_test, y_test)

#let's plot
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Random-Forest Regressor Importance")

# %% [markdown]
# #### Shap way
# 
# The SHAP interpretation can be used (it is model-agnostic) to compute the feature importances. It is using the Shapley values from game theory to estimate the how does each feature contribute to the prediction. It can be easily installed (<code>pip install shap</code>) 

# %%
import shap

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# %%
#shap provides plot
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names = X.columns)

# %% [markdown]
# ## 9. Inference
# 
# To provide inference service or deploy, it's best to save the model for latter use.

# %%
import pickle

# save the model to disk
filename = 'model/price_car_prediction.pkl'
pickle.dump(grid, open(filename, 'wb'))

# %%
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

# %%
#let's try to create one silly example
df[['max_power','engine','mileage','seats', 'km_driven', 'owner', 'lable_fuel','selling_price']].loc[1]

# %%
sample = np.array([[103.52,1498.00,21.14,5.00,120000.00,2,1]])

# %%
selling_price = loaded_model.predict(sample)
selling_price = np.exp(selling_price)
print(selling_price)

# %% [markdown]
# # Report

# %% [markdown]
# 


