# -*- coding: utf-8 -*-
"""
Created on Jul 15 02:16:36 2021

@author: Mohammad.Tanhatalab
"""
# External libraries
import pandas as pd
import numpy as np
import re
from datetime import datetime
import xlrd
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold
import xgboost as xg
from sklearn.pipeline import Pipeline

# =============================================================================
#  DataPrepration Class Definition
# =============================================================================
class DataPrepration:
    
    ''' Data preparation class includes below Function:
            0. Get Data
            1. Map the Target Values to Raw Data
            2. Create New KPIs
            3. Delete Columns with a high proportion of missing values
            4. Convert Catagorical columns to Numerical values
            5. Selected Features
                * Select Features basd on thier variance
                * Remove Duplicate Columns
            6. Use Imputation to reGenerate the missed data
            7. Shuffling the Dataset '''
        
    def __init__(self, raw_path, target_path):
        self.raw_path = raw_path
        self.target_path = target_path
        
    def Get_Data (self):
        ''' Load Raw and Target CSV files, since the index column is the first  
            unknown column, so the "Unnamed: 0" column rename to index '''
        df_raw = pd.read_csv(self.raw_path,  delimiter=';')
        df_target = pd.read_csv(self. target_path,  delimiter=';')
        df_raw.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
        return df_raw, df_target
    # =========================================================================
    #     Establish a correspondence Between Raw and Target
    # =========================================================================
    def Mapping_Raw_to_Target(self, df_raw, df_target, col1, col2):
        
        ''' Raw and Target dataset are not matched one-to one correspondence.
            By combination of index and groups in Raw and Target dataset, 
            a new Key has been created, as Index_Groups to map and link 
            the Target to the Raw data.'''
         
        df_raw[col1].astype(float) # Raw Index
        df_raw[col2].astype(float) #Raw Group
        df_target[col1].astype(float) # Target Index
        df_target[col2].astype(float) #Target Group
       
        # 
        df_raw.insert (2,'Index_group', 
                       [(str(df_raw.loc[i][col1].astype(int)) + '_' +
                         str(df_raw.loc[i][col2].astype(int))) 
                        for i in range (df_raw.shape[0])])
        df_target.insert (1,'Index_group', 
                       [(str(df_target.loc[i][col1].astype(int)) + '_' +
                         str(df_target.loc[i][col2].astype(int))) 
                        for i in range (df_target.shape[0])])
            
        Converted_Dataset = pd.merge(df_raw, df_target, how="inner", 
                                     on=["Index_group"] )
        Converted_Dataset.drop('index_y',axis = 1, inplace = True)
        Converted_Dataset.drop('groups_y',axis = 1, inplace = True)
        
        return Converted_Dataset

    # =============================================================================
    # Remove Columns with high Missing values
    # =============================================================================
    def Drop_missed_Column(self, df, col_to_del):
        '''Function to delete Columns with a high proportion of missing values,
           or one of the Duplicate columns '''
        for col in col_to_del:
            df.pop(col)
        return df           
    # =============================================================================
    #   To Unify time feature 
    # =============================================================================
    def Unifying_timestamp_col(self, df, col_to_convert):
        ''' Unifying all types of Date and time to %d/%m/%Y %H:%M
            in this part the Regex has been used to find the different types of Date time'''

        for col in col_to_convert:
            for row in range(df.shape[0]):
                if (re.match('[3-4][0-9]{4}.[0-9]',str(df.loc[row:row+1][col]))):
                    df.loc[row:row+1][col] = datetime(*xlrd.xldate_as_tuple(float(df.loc[row:row+1][col]), 0)).strftime('%Y-%m-%d %H:%M:%S')
                    df.loc[row:row+1][col] = datetime.strptime(df.loc[row:row+1][col], "%Y/%m/%d %H:%M:%S").strftime("%Y/%m/%d %H:%M:%S")
                elif (re.match('[0-9]{4}/[0-9]{2}/[0-9]',str(df.loc[row:row+1][col]))):
                    df.loc[row:row+1][col] = datetime.strptime(str(df.loc[row:row+1][col]), "%Y/%m/%d %H:%M:%S").strftime("%Y/%m/%d %H:%M:%S")
                        
        for col in col_to_convert:
    
            df[col] = pd.to_datetime(df[col] , format='%d/%m/%Y %H:%M')
    
        return df    
    # =============================================================================
    # Find the day gap between two dates
    # ============================================================================= 
    def Second_Between(self, t1, t2):
        ''' Calculate the time period between two times, based on Second'''
        Distance_sign = str((t2-t1)) # calculate of gap between two times. 
        Distance = str(abs(t2-t1)) # It change to str to use Regex
        Date_type = re.findall('[0-9]{2}:[0-9]{2}:[0-9]{2}', Distance)
        Sign = Distance_sign.split(' ')[3]
        if(len(Date_type)!= 0):
            Time = Date_type[0].split(':')
            Gap = int(Time[0])*3600+int(Time[1])*60+int(Time[2])
            if Sign == '-1':
                Gap=Gap*(-1) 
        else:
            Gap = 0

        return Gap
    # =============================================================================
    #                                Create New KPI
    # =============================================================================
    def New_KPI(self, df, new_kpi, col1, col2):
        '''Create new features (KPI) from timestamp values, 
           it helps us to make the better of timestamp values'''
        # Insert new KPI to the Dataset   
        df.insert(2, new_kpi, 
                  [ self.Second_Between(df.loc[i:i][col1],
                                                      df.loc[i:i][col2])
                                     for i in range(df.shape[0])])
        return df
    # =============================================================================
    # Use Ordinal Encoding to change catagorical columns to numerical values
    # =============================================================================
    def Catagorical_to_Numerical (self, df, col_to_numerical):
        ''' Relpace the Categoricall Features with Numerical 
            by OrdinalEncoder function'''
        enc = OrdinalEncoder()
        df[col_to_numerical] = enc.fit_transform(df[col_to_numerical])
        return df
    # =============================================================================
    # KNN Imputer
    # =============================================================================
    def KNN_Imputer(self, df):
        ''' Replacing NAN (missing) values with 
            reasonable data with KNN Algorithm.'''
        imputer = KNNImputer() # fit on the dataset
        imputer.fit(df) 
        X = imputer.transform(df)  # transform the dataset
        return X    
    # =============================================================================
    # Remove Columns That Have A Low Variance
    # =============================================================================      
    def Del_Low_Variance(self, df):
        '''To make the better performance low variance features have been removed.
           Based on tunning hyperparameters the threshold = 0.15 select 26 features 
           with better performance'''
        # define the transform: With this threshold 26 features has been selected
        transform = VarianceThreshold(threshold=.15)   
        X_sel = transform.fit_transform(df)  # transform the input data
        X = pd.DataFrame(X_sel)
        return X

#%%
# =============================================================================
# =============================================================================
# #                            XGBoost Class
# =============================================================================
# =============================================================================
class XGBoost_Train_Predict:
    ''' In this Class, th eModel has been trained with XGBoost method'''
    
    def __init__(self, X, Y, X_train, X_test, Y_train, Y_test):
        self.X = X
        self.Y = Y
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
    
    def Model_Fit (cls, X_train,  Y_train):
        xgb_r = xg.XGBRegressor(objective ='reg:linear',
                          n_estimators = 360, seed = 123,verbosity = 0)
        pipeline = Pipeline(steps=[('normalize', MinMaxScaler()), ('model', xgb_r)])
        model = TransformedTargetRegressor(regressor=pipeline, transformer=MinMaxScaler())
        model.fit(cls.X_train, cls.Y_train)
        return model
    
#    def Evaluate (cls, X, Y):
#        from sklearn.model_selection import KFold
#        from sklearn.model_selection import cross_val_score
#        kfold = KFold(n_splits=10, random_state=None)
#        results = cross_val_score(model, X, Y, cv=kfold)
#        print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
        
    def Model_Predict(cls, X_test, model):
        prediction_test = model.predict(cls.X_test)
        return prediction_test

#%%
# =============================================================================
#  Main Project
# =============================================================================
# Read and load the Feature and Target Data        
raw_path = r"C:\1 Research\Interview\Braincourt 2021.06\challange\md_raw_dataset.csv"
target_path = r"C:\1 Research\Interview\Braincourt 2021.06\challange\md_target_dataset.csv"
data_pre = DataPrepration(raw_path,target_path)
df_raw, df_target = data_pre.Get_Data()

# Join the Target and Feature Data 
Converted_Dataset = data_pre.Mapping_Raw_to_Target(df_raw, df_target, 'index', 'groups')

# Drop the high missed columns
col_to_del = ['etherium_before_start', 'start_critical_subprocess1','raw_kryptonite', 'pure_seastone', 'opened']
Converted_Dataset = data_pre.Drop_missed_Column(Converted_Dataset, col_to_del)

# Unify the Date time columns
col_to_convert = [ 'expected_start', 'start_process', 'start_subprocess1',
                   'predicted_process_end', 'process_end', 'subprocess1_end',
                   'reported_on_tower']
Converted_Dataset = data_pre.Unifying_timestamp_col(Converted_Dataset, col_to_convert)

# Create New KPIs (Feature)
Converted_Dataset = data_pre.New_KPI(Converted_Dataset, 'Expected_Start_Process', 'expected_start', 'start_process')
Converted_Dataset = data_pre.New_KPI(Converted_Dataset, 'start_time_of_subprocess1', 'start_process', 'start_subprocess1' )
Converted_Dataset = data_pre.New_KPI(Converted_Dataset, 'process_duration','start_process', 'process_end')
Converted_Dataset = data_pre.New_KPI(Converted_Dataset, 'expected_end_process', 'process_end', 'predicted_process_end' )
Converted_Dataset = data_pre.New_KPI(Converted_Dataset, 'subprocess_duration', 'start_process', 'subprocess1_end' )
Converted_Dataset = data_pre.New_KPI(Converted_Dataset, 'whole_process', 'start_process', 'reported_on_tower')

# Convert Categorical to Numerical columns
col_to_numerical = ["super_hero_group","crystal_type", "crystal_supergroup", "Cycle"]
Converted_Dataset = data_pre.Catagorical_to_Numerical (Converted_Dataset, col_to_numerical)

# Remove the old and un-used columns from the DataSet
col_to_del = [ 'when', 'expected_start', 'start_process', 'start_subprocess1',
                   'predicted_process_end', 'process_end', 'subprocess1_end',
                   'reported_on_tower']
Converted_Dataset = data_pre.Drop_missed_Column(Converted_Dataset, col_to_del)

# Extract the feature and Target from the Dataset
X = Converted_Dataset.iloc[:,0:-1]
Y = Converted_Dataset.iloc[:,-1]

# Replace the missing values by calculated values
X = data_pre.KNN_Imputer(X)

# Remove low Variance columns from the Dataset
X = data_pre.Del_Low_Variance(X)

# Suffel the Dataset and split into the Train and Test Data
X = X.sample(frac = 1)
Y = X.iloc[:,-1]
X = X.iloc[:,0:-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.015, random_state=10)

# Define XGBoost from Class of XGBoost_Train_Predict
XGB = XGBoost_Train_Predict(X, Y, X_train, X_test, Y_train, Y_test)

# Create Model based on Train Data
model = XGB.Model_Fit(X_train,  Y_train)

# Predict X_Test
prediction_test = XGB.Model_Predict(X_test, model)

# Plot the Acual (Y_Test) and Predicted values
plt.figure(figsize=(17, 6))
plt.plot(np.arange(Y_test.shape[0]), np.array(Y_test), 
        label='Actual Values')     
plt.plot(np.arange(Y_test.shape[0]),np.array(prediction_test),
        label='Predicted Values')
plt.legend(loc='upper left')
plt.show()
# Evaluate the Algorithm and show Accuracy 
#XGB.Evaluate(X,Y)
