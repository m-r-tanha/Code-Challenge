# -*- coding: utf-8 -*-
"""
Created on Jul 15 02:16:36 2021

@author: Mohammad.Tanhatalab
"""
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import re
from datetime import datetime
import xlrd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import mean
from numpy import absolute
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor 
from sklearn.compose import TransformedTargetRegressor
import xgboost as xg
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import GridSearchCV

path = r"C:\1 Research\Interview\Braincourt 2021.06\challange\md_raw_dataset.csv"
target_pat = r"C:\1 Research\Interview\Braincourt 2021.06\challange\md_target_dataset.csv"
df_raw = pd.read_csv(path,  delimiter=';')
df_target = pd.read_csv(target_pat,  delimiter=';')
#%%
# =============================================================================
# Map Raw Data to Target
# =============================================================================
df_raw['Unnamed: 0'].astype(float) # Raw Index
df_target.iloc[:,0].astype(float)
df_raw['groups'].astype(float)
df_target['groups'].astype(float)


df_raw.insert (2,'Index_group', 
               [(str(df_raw.loc[i]['Unnamed: 0'].astype(int)) + '_' + str(df_raw.loc[i]['groups'].astype(int))) 
                for i in range (df_raw.shape[0])])
df_target.insert (1,'Index_group', 
               [(str(df_target.loc[i]['index'].astype(int)) + '_' + str(df_target.loc[i]['groups'].astype(int))) 
                for i in range (df_target.shape[0])])


Converted_Detaset = pd.merge(df_raw, df_target, how="inner", on=["Index_group"] )
#%%
# =============================================================================
# Remove Columns with high Missing values
# =============================================================================
df_target.isna().sum()
Converted_Detaset.drop(['etherium_before_start', 'start_critical_subprocess1','raw_kryptonite', 'pure_seastone', 'opened'],
            axis = 1, inplace = True)

# Some date values are in text type like (43223.56) they should detect and convert to date type
# the detection has been done by regex
column = [ 'expected_start', 'start_process', 'start_subprocess1',
       'predicted_process_end', 'process_end', 'subprocess1_end',
       'reported_on_tower']

for col in column:
    for row in range(Converted_Detaset.shape[0]):
        if (re.match('[3-4][0-9]{4}.[0-9]',str(Converted_Detaset.loc[row:row+1][col]))):
            Converted_Detaset.loc[row:row+1][col] = datetime(*xlrd.xldate_as_tuple(float(Converted_Detaset.loc[row:row+1][col]), 0)).strftime('%Y-%m-%d %H:%M:%S')
            Converted_Detaset.loc[row:row+1][col] = datetime.strptime(Converted_Detaset.loc[row:row+1][col], "%Y/%m/%d %H:%M:%S").strftime('%d/%m/%Y %H:%M:%S')
        elif (re.match('[0-9]{4}/[0-9]{2}/[0-9]',str(Converted_Detaset.loc[row:row+1][col]))):
            Converted_Detaset.loc[row:row+1][col] = datetime.strptime(str(Converted_Detaset.loc[row:row+1][col]), "%Y/%m/%d %H:%M:%S").strftime('%d/%m/%Y %H:%M:%S')
            
Converted_Detaset['expected_start'] = pd.to_datetime(Converted_Detaset['expected_start'] , format='%d/%m/%Y %H:%M')
Converted_Detaset['start_process'] = pd.to_datetime(Converted_Detaset['start_process'] , format='%d/%m/%Y %H:%M')
Converted_Detaset['start_subprocess1'] = pd.to_datetime(Converted_Detaset['start_subprocess1'] , format='%d/%m/%Y %H:%M')
Converted_Detaset['predicted_process_end'] = pd.to_datetime(Converted_Detaset['predicted_process_end'] , format='%d/%m/%Y %H:%M')
Converted_Detaset['process_end'] = pd.to_datetime(Converted_Detaset['process_end'] , format='%d/%m/%Y %H:%M')
Converted_Detaset['subprocess1_end'] = pd.to_datetime(Converted_Detaset['subprocess1_end'] , format='%d/%m/%Y %H:%M')
Converted_Detaset['reported_on_tower'] = pd.to_datetime(Converted_Detaset['reported_on_tower'] , format='%d/%m/%Y %H:%M')
Converted_Detaset.reset_index(inplace = True, drop=True)
#%%
# =============================================================================
# Find the day gap between two dates
# =============================================================================

def minuts_between(t1, t2):

    M1 = str((t2-t1))
    M = str(abs(t2-t1))
    H=re.findall('[0-9]{2}:[0-9]{2}:[0-9]{2}', M)
    h = M1.split(' ')[3]
    if(len(H)!= 0):
        Time = H[0].split(':')
#         print(Time)
        Gap = int(Time[0])*3600+int(Time[1])*60+int(Time[2])
        if h == '-1':
            Gap=Gap*(-1) 
    else:
        Gap = 0
#     print(Gap)
    return Gap
#%%
# =============================================================================
#                                Create New KPI
# =============================================================================

Converted_Detaset.insert(2, 
                         'Expected_Start_Process', 
                         [ minuts_between(Converted_Detaset.loc[i:i]['expected_start'],
                                          Converted_Detaset.loc[i:i]['start_process'])
                         for i in range(Converted_Detaset.shape[0])])

Converted_Detaset.insert(2, 
                         'start_time_of_subprocess1', 
                         [ minuts_between(Converted_Detaset.loc[i:i]['start_process'],
                                          Converted_Detaset.loc[i:i]['start_subprocess1'])
                         for i in range(Converted_Detaset.shape[0])])

Converted_Detaset.insert(2, 
                         'process_duration', 
                         [ minuts_between(Converted_Detaset.loc[i:i]['start_process'],
                                          Converted_Detaset.loc[i:i]['process_end'])
                         for i in range(Converted_Detaset.shape[0])])

Converted_Detaset.insert(2, 
                         'expected_end_process', 
                         [ minuts_between(Converted_Detaset.loc[i:i]['process_end'],
                                          Converted_Detaset.loc[i:i]['predicted_process_end'])
                         for i in range(Converted_Detaset.shape[0])])

Converted_Detaset.insert(2, 
                         'subprocess_duration', 
                         [ minuts_between(Converted_Detaset.loc[i:i]['start_process'],
                                          Converted_Detaset.loc[i:i]['subprocess1_end'])
                         for i in range(Converted_Detaset.shape[0])])

Converted_Detaset.insert(2, 
                         'whole_process', 
                         [ minuts_between(Converted_Detaset.loc[i:i]['start_process'],
                                          Converted_Detaset.loc[i:i]['reported_on_tower'])
                         for i in range(Converted_Detaset.shape[0])])
#%%
# =============================================================================
# Use Ordinal Encoding to change catagorical columns to numerical values
# =============================================================================

enc = OrdinalEncoder()
Converted_Detaset[["super_hero_group","crystal_type", "crystal_supergroup", "Cycle"]] = enc.fit_transform(Converted_Detaset[["super_hero_group","crystal_type", "crystal_supergroup", "Cycle"]])
#%%
# =============================================================================
# Define new DataSet with new KPIs (Features)
# =============================================================================

X = Converted_Detaset[['Unnamed: 0','whole_process', 'subprocess_duration',
       'expected_end_process', 'process_duration', 'start_time_of_subprocess1',
       'Expected_Start_Process', 'super_hero_group', 'tracking',
       'place', 'tracking_times', 'crystal_type', 'Unnamed: 7',
       'human_behavior_report', 'human_measure', 'crystal_weight',
       'expected_factor_x', 'previous_factor_x', 'first_factor_x',
       'expected_final_factor_x', 'final_factor_x', 'previous_adamantium',
       'Unnamed: 17', 'chemical_x', 'argon', 'crystal_supergroup',
       'Cycle','groups_x']]
Y = Converted_Detaset[['target']]

#%%
# =============================================================================
# KNN Imputer
# =============================================================================
from numpy import isnan
from sklearn.impute import KNNImputer

# define imputer
imputer = KNNImputer()
# fit on the dataset
imputer.fit(X)
# transform the dataset
X = imputer.transform(X)
# summarize total missing


#%%
# =============================================================================
# Remove Columns That Have A Low Variance
# =============================================================================

print(X.shape, Y.shape)
# define the transform
transform = VarianceThreshold(threshold=.15)
# transform the input data
X_sel = transform.fit_transform(X)
print(X_sel.shape)
X_sel = pd.DataFrame(X_sel)
X = X_sel.copy()

X_Orgin = X.copy()
Y_Orgin = Y.copy()

#%%
# =============================================================================
# Find Hyperparameters (best number of estimators in XGBoost)
# =============================================================================
#import xgboost as xg
##regressor = DecisionTreeRegressor(random_state = 0)
#for i in range (1, 700, 100):
#    xgb_r = xg.XGBRegressor(objective ='reg:linear',
#                      n_estimators = i, seed = 123)
#    pipeline = Pipeline(steps=[('normalize', MinMaxScaler()), ('model', xgb_r)])
#    # prepare the model with target scaling
#    model = TransformedTargetRegressor(regressor=pipeline, transformer=MinMaxScaler())
#    # evaluate model
#    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#    scores = cross_val_score(model, X, Y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
#    # convert scores to positive
#    scores = absolute(scores)
#    # summarize the result
#    s_mean = mean(scores)
#    print('n_estimators: %d , Mean MAE: %.3f' % (i, s_mean))
#%%
# =============================================================================
# Tune the Number of Selected Features
# =============================================================================
#cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)

#regressor = DecisionTreeRegressor(random_state = 0)
xgb_r = xg.XGBRegressor(objective ='reg:linear',
                  n_estimators = 50, seed = 123)
pipeline = Pipeline(steps=[('normalize', MinMaxScaler()), ('model', xgb_r)])
model = TransformedTargetRegressor(regressor=pipeline, transformer=MinMaxScaler())

fs = SelectKBest(score_func=mutual_info_regression)
pipeline = Pipeline(steps=[('sel',fs), ('lr', model)])
# define the grid
grid = dict()
grid['sel__k'] = [i for i in range(X.shape[1]-15, X.shape[1]+1)]
# define the grid search
search = GridSearchCV(pipeline, grid, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv)
# perform the search
results = search.fit(X, Y)
# summarize best
print('Best MAE: %.3f' % results.best_score_)
print('Best Config: %s' % results.best_params_)

means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
for np.mean, param in zip(means, params):
    print('>%.3f with: %r' % (np.mean, param))
##    
# =============================================================================
# Final Projext run on X_test
# =============================================================================

#%%
X = X_Orgin.copy()
Y = Y_Orgin.copy()

X = X.sample(frac = 1)
Y = X.iloc[:,-1]
X = X.iloc[:,0:-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.015, random_state=10)

#regressor = DecisionTreeRegressor(random_state = 0)
xgb_r = xg.XGBRegressor(objective ='reg:linear',
                  n_estimators = 360, seed = 123)
pipeline = Pipeline(steps=[('normalize', MinMaxScaler()), ('model', xgb_r)])
model = TransformedTargetRegressor(regressor=pipeline, transformer=MinMaxScaler())
model.fit(X_train,Y_train)
prediction_test = model.predict(X_test)
plt.figure(figsize=(17, 6))

plt.plot(np.arange(X_test.shape[0]), np.array(Y_test), 
        label='Actual Values')     
plt.plot(np.arange(X_test.shape[0]),np.array(prediction_test),
        label='Predicted Values')
plt.legend(loc='upper left')
#%%
# =============================================================================
# AdaBoostRegressor run on X_test
# =============================================================================
#%%
#Sample = Converted_Detaset[Converted_Detaset['groups_y']<3]
#X_Sample = Sample[['Unnamed: 0','whole_process', 'subprocess_duration',
#       'expected_end_process', 'process_duration', 'start_time_of_subprocess1',
#       'Expected_Start_Process', 'super_hero_group', 'tracking',
#       'place', 'tracking_times', 'crystal_type', 'Unnamed: 7',
#       'human_behavior_report', 'human_measure', 'crystal_weight',
#       'expected_factor_x', 'previous_factor_x', 'first_factor_x',
#       'expected_final_factor_x', 'final_factor_x', 'previous_adamantium',
#       'Unnamed: 17', 'chemical_x', 'argon', 'crystal_supergroup',
#       'Cycle','groups_x']]
#Y_Sample = Sample[['target']]
##xgb_r = xg.XGBRegressor(objective ='reg:linear',
##                  n_estimators = 360, seed = 123)
##pipeline = Pipeline(steps=[('normalize', MinMaxScaler()), ('model', xgb_r)])
##model = TransformedTargetRegressor(regressor=pipeline, transformer=MinMaxScaler())
##model.fit(X_Sample,Y_Sample)
#
#prediction_test = model.predict(X_Sample)
#plt.figure(figsize=(17, 6))
#
#plt.plot(np.arange(X_Sample.shape[0]), np.array(Y_Sample), 
#        label='Actual Values')     
#plt.plot(np.arange(X_Sample.shape[0]),np.array(prediction_test),
#        label='Predicted Values')
#plt.legend(loc='upper left')