# -*- coding: utf-8 -*-
"""GLM Model 26.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AXGuxEm465CnT2ccLFrPRsPtr5IwBwTM

# Overview
The following notebook outlines the process that one of our data scientists utilized to build a "segmentation" model for our marketing department.  The dependent variable is whether a customer purchased a product (y=1).or not (y=0). The implemented model will help the marketing department decide which customers receive an advertisement for the product.

## Reading in the Libraries
"""

import numpy as np
import pandas as pd
import sys
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
#loading visualization library
# import bokeh

import collections as ccc

# print("python version " + sys.version)
# print('numpy version ' + np.__version__)
# print('pandas version ' + pd.__version__)
# print('sklern version ' + '0.23.1')
# print('bokeh version ' + bokeh.__version__)
# print('statsmodels version ' + '0.9.0')

raw_train=pd.read_csv('exercise_26_train.csv')
raw_test=pd.read_csv('exercise_26_test.csv')
#Desribing the target variable
from collections import Counter
Counter(raw_train.y)

raw_train.head()

# Overview of data types
print("object dtype:", raw_train.columns[raw_train.dtypes == 'object'].tolist())
print("int64 dtype:", raw_train.columns[raw_train.dtypes == 'int'].tolist())
print("The rest of the columns have float64 dtypes.")

# Investigate Object Columns
def investigate_object(df):
    """
    This function prints the unique categories of all the object dtype columns.
    It prints '...' if there are more than 13 unique categories.
    """
    col_obj = df.columns[df.dtypes == 'object']

    for i in range(len(col_obj)):
        if len(df[col_obj[i]].unique()) > 13:
            print(col_obj[i]+":", "Unique Values:", np.append(df[col_obj[i]].unique()[:13], "..."))
        else:
            print(col_obj[i]+":", "Unique Values:", df[col_obj[i]].unique())

    del col_obj
investigate_object(raw_train)

"""# Feature Engineering"""



train_val = raw_train.copy(deep=True)

#1. Fixing the money and percents#
train_val['x12'] = train_val['x12'].str.replace('$','')
train_val['x12'] = train_val['x12'].str.replace(',','')
train_val['x12'] = train_val['x12'].str.replace(')','')
train_val['x12'] = train_val['x12'].str.replace('(','-')
train_val['x12'] = train_val['x12'].astype(float)
train_val['x63'] = train_val['x63'].str.replace('%','')
train_val['x63'] = train_val['x63'].astype(float)

#Dropping Nan values from the dataset(If any)

train_val.dropna(subset=['y'],inplace=True)
# 2. Creating the train/val/test set
x_train, x_val, y_train, y_val = train_test_split(train_val.drop(columns=['y']), train_val['y'], test_size=0.1, random_state=13)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=4000, random_state=13)

# 3. smashing sets back together
train = pd.concat([x_train, y_train], axis=1, sort=False).reset_index(drop=True)
val = pd.concat([x_val, y_val], axis=1, sort=False).reset_index(drop=True)
test = pd.concat([x_test, y_test], axis=1, sort=False).reset_index(drop=True)

# 3. With mean imputation from Train set

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
train_imputed = pd.DataFrame(imputer.fit_transform(train.drop(columns=['y', 'x5', 'x31',  'x81' ,'x82'])), columns=train.drop(columns=['y', 'x5', 'x31', 'x81', 'x82']).columns)
std_scaler = StandardScaler()
train_imputed_std = pd.DataFrame(std_scaler.fit_transform(train_imputed), columns=train_imputed.columns)

# 3 create dummies

dumb5 = pd.get_dummies(train['x5'], drop_first=True, prefix='x5', prefix_sep='_', dummy_na=True)
train_imputed_std = pd.concat([train_imputed_std, dumb5], axis=1, sort=False)

dumb31 = pd.get_dummies(train['x31'], drop_first=True, prefix='x31', prefix_sep='_', dummy_na=True)
train_imputed_std = pd.concat([train_imputed_std, dumb31], axis=1, sort=False)

dumb81 = pd.get_dummies(train['x81'], drop_first=True, prefix='x81', prefix_sep='_', dummy_na=True)
train_imputed_std = pd.concat([train_imputed_std, dumb81], axis=1, sort=False)

dumb82 = pd.get_dummies(train['x82'], drop_first=True, prefix='x82', prefix_sep='_', dummy_na=True)
train_imputed_std = pd.concat([train_imputed_std, dumb82], axis=1, sort=False)
train_imputed_std = pd.concat([train_imputed_std, train['y']], axis=1, sort=False)

del dumb5, dumb31, dumb81, dumb82
train.head()

#Showing the imputer statistics
imputer.statistics_

#Showing the variance
train_imputed.var()

train_imputed_std.head()

"""# Visualizing the Correlations
As part of the exploratory analysis, we want to look at a heatmap to see if there are any high pairwise correlations.  If we see a few number of variables correlated with the target, then we will use an L2 penalty.  If we see a lot of variables correlated with y then we will use an L2 penalty.
"""


"""# Initial Feature Selection
Looking at the correlation map from above, we can see there are very few variables associated with the dependent variable.  Thus, we will use an L1 penalty to for feature selection. Interestingly enough, we see a that some variables have heavy correlation amongst themselves.
"""

exploratory_LR = LogisticRegression(penalty='l1', fit_intercept=False, solver='liblinear')
exploratory_LR.fit(train_imputed_std.drop(columns=['y']), train_imputed_std['y'])
exploratory_results = pd.DataFrame(train_imputed_std.drop(columns=['y']).columns).rename(columns={0:'name'})
exploratory_results['coefs'] = exploratory_LR.coef_[0]
exploratory_results['coefs_squared'] = exploratory_results['coefs']**2
var_reduced = exploratory_results.nlargest(25,'coefs_squared')

"""# Preliminary Model
## Starting with the train set
The L1 process creates biased parameter estimates.  As a result, we will build a final model without biased estimators.
"""

variables = var_reduced['name'].to_list()

logit = sm.Logit(train_imputed_std['y'], train_imputed_std[variables])
# fit the model
result = logit.fit()
result.summary()

"""## Prepping the validation set"""

val_imputed = pd.DataFrame(imputer.transform(val.drop(columns=['y', 'x5', 'x31', 'x81' ,'x82'])), columns=train.drop(columns=['y','x5', 'x31', 'x81', 'x82']).columns)
val_imputed_std = pd.DataFrame(std_scaler.transform(val_imputed), columns=train_imputed.columns)

dumb5 = pd.get_dummies(val['x5'], drop_first=True, prefix='x5', prefix_sep='_', dummy_na=True)
val_imputed_std = pd.concat([val_imputed_std, dumb5], axis=1, sort=False)

dumb31 = pd.get_dummies(val['x31'], drop_first=True, prefix='x31', prefix_sep='_', dummy_na=True)
val_imputed_std = pd.concat([val_imputed_std, dumb31], axis=1, sort=False)

dumb81 = pd.get_dummies(val['x81'], drop_first=True, prefix='x81', prefix_sep='_', dummy_na=True)
val_imputed_std = pd.concat([val_imputed_std, dumb81], axis=1, sort=False)

dumb82 = pd.get_dummies(val['x82'], drop_first=True, prefix='x82', prefix_sep='_', dummy_na=True)
val_imputed_std = pd.concat([val_imputed_std, dumb82], axis=1, sort=False)
val_imputed_std = pd.concat([val_imputed_std, val['y']], axis=1, sort=False)

val_imputed_std.head()

"""## Prepping the test set"""

test_imputed = pd.DataFrame(imputer.transform(test.drop(columns=['y', 'x5', 'x31', 'x81' ,'x82'])), columns=train.drop(columns=['y','x5', 'x31', 'x81', 'x82']).columns)
test_imputed_std = pd.DataFrame(std_scaler.transform(test_imputed), columns=train_imputed.columns)

# 3 create dummies

dumb5 = pd.get_dummies(test['x5'], drop_first=True, prefix='x5', prefix_sep='_', dummy_na=True)
test_imputed_std = pd.concat([test_imputed_std, dumb5], axis=1, sort=False)

dumb31 = pd.get_dummies(test['x31'], drop_first=True, prefix='x31', prefix_sep='_', dummy_na=True)
test_imputed_std = pd.concat([test_imputed_std, dumb31], axis=1, sort=False)

dumb81 = pd.get_dummies(test['x81'], drop_first=True, prefix='x81', prefix_sep='_', dummy_na=True)
test_imputed_std = pd.concat([test_imputed_std, dumb81], axis=1, sort=False)

dumb82 = pd.get_dummies(test['x82'], drop_first=True, prefix='x82', prefix_sep='_', dummy_na=True)
test_imputed_std = pd.concat([test_imputed_std, dumb82], axis=1, sort=False)
test_imputed_std = pd.concat([test_imputed_std, test['y']], axis=1, sort=False)

val_imputed_std.columns

Outcomes_train = pd.DataFrame(result.predict(train_imputed_std[variables])).rename(columns={0:'probs'})
Outcomes_train['y'] = train_imputed_std['y']
print('The C-Statistics is ',roc_auc_score(Outcomes_train['y'], Outcomes_train['probs']))
Outcomes_val = pd.DataFrame(result.predict(val_imputed_std[variables])).rename(columns={0:'probs'})
Outcomes_val['y'] = val_imputed_std['y']
print('The C-Statistics is ',roc_auc_score(Outcomes_val['y'], Outcomes_val['probs']))
Outcomes_test = pd.DataFrame(result.predict(test_imputed_std[variables])).rename(columns={0:'probs'})
Outcomes_test['y'] = test_imputed_std['y']
print('The C-Statistics is ',roc_auc_score(Outcomes_test['y'], Outcomes_test['probs']))
Outcomes_train['prob_bin'] = pd.qcut(Outcomes_train['probs'], q=20)

Outcomes_train.groupby(['prob_bin'])['y'].sum()

"""# Finalizing the Model
In the code above, we identified that the model generalized well; the AUC was similar for each of the partitions of the training data.  Moving forward, we want to
1. refit the model using all of the training data
2. check the coefficients against the preliminary model
3. assess the lift and ask for a cutoff from the business partner.
"""

train_and_val = pd.concat([train_imputed_std, val_imputed_std])
all_train = pd.concat([train_and_val, test_imputed_std])
variables = var_reduced['name'].to_list()
final_logit = sm.Logit(all_train['y'], all_train[variables])
# fit the model
final_result = final_logit.fit()
final_result.summary()

Outcomes_test_final = pd.DataFrame(final_result.predict(test_imputed_std[variables])).rename(columns={0:'phat'})
Outcomes_test_final['business_outcome'] = test_imputed_std['y']
print('The C-Statistics is ',roc_auc_score(Outcomes_test_final['business_outcome'], Outcomes_test_final['phat']))
Outcomes_test_final['prob_bin'] = pd.qcut(Outcomes_test_final['phat'], q=20)
Outcomes_test_final.groupby(['prob_bin'])['business_outcome'].sum()


# Define the threshold for classifying as an event (75th percentile)
threshold = Outcomes_test_final['phat'].quantile(0.75)
# Classify observations in the top 5 bins as an event
Outcomes_test_final['business_outcome'] = (Outcomes_test_final['phat'] > threshold).astype(int)
# Assign the predicted probability to the variable 'phat'
Outcomes_test_final['phat'] = Outcomes_test_final['phat']
# Display the DataFrame with the new columns
print(Outcomes_test_final[['business_outcome', 'phat']])

Outcomes_test_final
# variables

# Add a new column 'prob_bin' representing quantiles with labels [1, 2, ..., 20]
Outcomes_test_final['prob_bin'] = pd.qcut(Outcomes_test_final['phat'], q=20, labels=False) + 1

# Display the count of 'y' for each quantile
quantile_counts = Outcomes_test_final.groupby(['prob_bin'])['business_outcome'].sum()
print(quantile_counts)

def get_result(raw_test,final_model=final_result):
  variables = final_model.params.index.tolist()
  test_val = raw_test.copy(deep=True)

  #1. Fixing the money and percents#
  test_val['x12'] = test_val['x12'].str.replace('$','')
  test_val['x12'] = test_val['x12'].str.replace(',','')
  test_val['x12'] = test_val['x12'].str.replace(')','')
  test_val['x12'] = test_val['x12'].str.replace('(','-')
  test_val['x12'] = test_val['x12'].astype(float)
  test_val['x63'] = test_val['x63'].str.replace('%','')
  test_val['x63'] = test_val['x63'].astype(float)

  #Dropping Nan values from the dataset(If any)

  # test_val.dropna(subset=['y'],inplace=True)

  # x_train, x_val, y_train, y_val = train_test_split(train_val.drop(columns=['y']), train_val['y'], test_size=0.1, random_state=13)
  # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=4000, random_state=13)

  # # 3. smashing sets back together
  # train = pd.concat([x_train, y_train], axis=1, sort=False).reset_index(drop=True)
  # val = pd.concat([x_val, y_val], axis=1, sort=False).reset_index(drop=True)
  # test = pd.concat([x_test, y_test], axis=1, sort=False).reset_index(drop=True)

  # 3. With mean imputation from Train set

  imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

#   test_imputed = pd.DataFrame(imputer.fit_transform(test_val.drop(columns=[ 'x5', 'x31',  'x81' ,'x82'])), columns=test_val.drop(columns=[ 'x5', 'x31', 'x81', 'x82']).columns)
#   std_scaler = StandardScaler()
#   test_imputed_std = pd.DataFrame(std_scaler.fit_transform(test_imputed), columns=test_imputed.columns)

  # 3 create dummies

  months=['January','February','March','April','May','June','July','August','September','October','November','December']
  country=['germany','america','asia','japan']
  week= ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
  gender=['Male,Female']



  # All possible categories for each variable
  categories = {
    'x5': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
    'x31': ['asia', 'germany', 'japan', 'america'],
    'x81': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
    'x82': ['Male', 'Female']
    }

# Create dummy variables
#   for column, cats in categories.items():
#         dummies = pd.get_dummies(test_imputed_std[column])
#         # Create missing columns for categories not in test_data
#         for cat in cats:
#             dummy_col = f"{column}_{cat}"
#             if dummy_col not in dummies.columns:
#                 dummies[dummy_col] = 0
#         # Drop original column and concatenate
#         test_imputed_std = test_imputed_std.drop(column, axis=1).join(dummies)

  for column, cats in categories.items():
    for cat in cats:
        test_val[f'{column}_{cat}'] = (test_val[column] == cat).astype(int)
    test_val.drop(column, axis=1, inplace=True)
  
#   dumb5 = pd.get_dummies(test_val['x5'], drop_first=True, prefix='x5', prefix_sep='_', dummy_na=True)
#   test_imputed_std = pd.concat([test_imputed_std, dumb5], axis=1, sort=False)

#   dumb31 = pd.get_dummies(test_val['x31'], drop_first=True, prefix='x31', prefix_sep='_', dummy_na=True)
#   test_imputed_std = pd.concat([test_imputed_std, dumb31], axis=1, sort=False)

#   dumb81 = pd.get_dummies(test_val['x81'], drop_first=True, prefix='x81', prefix_sep='_', dummy_na=True)
#   test_imputed_std = pd.concat([test_imputed_std, dumb81], axis=1, sort=False)

#   dumb82 = pd.get_dummies(test_val['x82'], drop_first=True, prefix='x82', prefix_sep='_', dummy_na=True)
#   test_imputed_std = pd.concat([test_imputed_std, dumb82], axis=1, sort=False)
  # train_imputed_std = pd.concat([train_imputed_std, train['y']], axis=1, sort=False)

#   del dumb5, dumb31, dumb81, dumb82


  test_imputed = pd.DataFrame(imputer.fit_transform(test_val), columns=test_val.columns)
  std_scaler = StandardScaler()
  test_imputed_std = pd.DataFrame(std_scaler.fit_transform(test_imputed), columns=test_imputed.columns)

  print(test_imputed_std.head())





  #Test
  Outcomes_test_final = pd.DataFrame(final_model.predict(test_imputed_std[variables])).rename(columns={0:'phat'})
  # Outcomes_test_final['business_outcome'] = test_imputed_std['y']
  # print('The C-Statistics is ',roc_auc_score(Outcomes_test_final['business_outcome'], Outcomes_test_final['phat']))
  Outcomes_test_final['business_outcome'] = Outcomes_test_final['phat'].round().astype(int)

  Outcomes_test_final['prob_bin'] = pd.qcut(Outcomes_test_final['phat'], q=20)
  Outcomes_test_final.groupby(['prob_bin'])['business_outcome'].sum()

  # Define the threshold for classifying as an event (75th percentile)
  threshold = Outcomes_test_final['phat'].quantile(0.75)
  # Classify observations in the top 5 bins as an event
  Outcomes_test_final['business_outcome'] = (Outcomes_test_final['phat'] > threshold).astype(int)
  # Assign the predicted probability to the variable 'phat'



  return Outcomes_test_final[['business_outcome', 'phat']],variables
# print(raw_test)
print("#################################################")
print(get_result(raw_test))

# variable_names = final_result.params.index.tolist()
# print(variable_names)

"""## Debrief
In the final discussion with the business partner, the partner was thrilled with the rank-order ability of the model.  Based on a combination of capacity and accuracy, the partner would like to classify any observation that would fall in the top 5 bins as an event; for simplicity we will say the cutoff is at the 75th percentile.  For the API, please return the predicted outcome (variable name is business_outcome), predicted probability (variable name is phat), and all model inputs; the variables should be returned in alphabetical order in the API return.
"""