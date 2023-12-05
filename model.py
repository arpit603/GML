
#Importing the necessary libraries

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
import bokeh

import collections as ccc

#importing the data

raw_train=pd.read_csv('exercise_26_train.csv')
raw_test=pd.read_csv('exercise_26_test.csv')

#Desribing the target variable
from collections import Counter
Counter(raw_train.y)


#preppingthe Train data for model processing

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


#Feature Selection


exploratory_LR = LogisticRegression(penalty='l1', fit_intercept=False, solver='liblinear')
exploratory_LR.fit(train_imputed_std.drop(columns=['y']), train_imputed_std['y'])
exploratory_results = pd.DataFrame(train_imputed_std.drop(columns=['y']).columns).rename(columns={0:'name'})
exploratory_results['coefs'] = exploratory_LR.coef_[0]
exploratory_results['coefs_squared'] = exploratory_results['coefs']**2
var_reduced = exploratory_results.nlargest(25,'coefs_squared')




#Prepping the validation data
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



#Prepping the Incoming test data
#the input data is in a dataframe named "test"
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




#Final Model Fit

train_and_val = pd.concat([train_imputed_std, val_imputed_std])
all_train = pd.concat([train_and_val, test_imputed_std])
variables = var_reduced['name'].to_list()
final_logit = sm.Logit(all_train['y'], all_train[variables])
# fit the model
final_result = final_logit.fit()



#Result Prediction 
#Train 
Outcomes_train_final = pd.DataFrame(final_result.predict(all_train[variables])).rename(columns={0:'phat'})
Outcomes_train_final['business_outcome'] = all_train['y']
print('The C-Statistics is ',roc_auc_score(Outcomes_train_final['business_outcome'], Outcomes_train_final['phat']))
Outcomes_train_final['prob_bin'] = pd.qcut(Outcomes_train_final['phat'], q=20)
Outcomes_train_final.groupby(['prob_bin'])['business_outcome'].sum()



#Test
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


#to return in the API call 

#1. Outcomes_test_final
#2. variables

#the variables should be returned in alphabetical order in the API return.

#final function 
def get_result(raw_train,raw_test):

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

  exploratory_LR = LogisticRegression(penalty='l1', fit_intercept=False, solver='liblinear')
  exploratory_LR.fit(train_imputed_std.drop(columns=['y']), train_imputed_std['y'])
  exploratory_results = pd.DataFrame(train_imputed_std.drop(columns=['y']).columns).rename(columns={0:'name'})
  exploratory_results['coefs'] = exploratory_LR.coef_[0]
  exploratory_results['coefs_squared'] = exploratory_results['coefs']**2
  var_reduced = exploratory_results.nlargest(25,'coefs_squared')

    #Prepping the validation data
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



  #Prepping the Incoming test data
  #the input data is in a dataframe named "test"
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

  train_and_val = pd.concat([train_imputed_std, val_imputed_std])
  all_train = pd.concat([train_and_val, test_imputed_std])
  variables = var_reduced['name'].to_list()
  final_logit = sm.Logit(all_train['y'], all_train[variables])
  # fit the model
  final_result = final_logit.fit()



  #Result Prediction 
  #Train 
  Outcomes_train_final = pd.DataFrame(final_result.predict(all_train[variables])).rename(columns={0:'phat'})
  Outcomes_train_final['business_outcome'] = all_train['y']
  print('The C-Statistics is ',roc_auc_score(Outcomes_train_final['business_outcome'], Outcomes_train_final['phat']))
  Outcomes_train_final['prob_bin'] = pd.qcut(Outcomes_train_final['phat'], q=20)
  Outcomes_train_final.groupby(['prob_bin'])['business_outcome'].sum()



  #Test
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
  
  # Display the DataFrame with the new columns
  #print(Outcomes_test_final[['business_outcome', 'phat']])
  
  return Outcomes_train_final[['business_outcome', 'phat']],variables


