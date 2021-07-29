
import pandas as pd
import numpy as np
import re
from datetime import datetime
# import xlrd
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold
import xgboost as xg
from sklearn.pipeline import Pipeline
# Check core SDK version number
import azureml.core
from azureml.core import Workspace, Dataset

print('SDK version:', azureml.core.VERSION)

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
    # Define Class Variable
    df_raw = pd. DataFrame()
    df_target = pd. DataFrame()
    
    def __init__(self, raw_name, target_name):
        # Define Instance variable
        self.raw_name = raw_name
        self.target_name = target_name
        
    def Get_Data (self):
        ''' Load Raw and Target CSV files, since the index column is the first  
            unknown column, so the "Unnamed: 0" column rename to index '''

        subscription_id = 'ec0ad9e6-b046-4a60-81a6-c379f32c659e'
        resource_group = 'AML1'
        workspace_name = 'AMLocode-Challenge'

        workspace = Workspace(subscription_id, resource_group, workspace_name)


        df_target = Dataset.get_by_name(workspace, name=self.target_name)
        df_target = df_target.to_pandas_dataframe()

        df_raw = Dataset.get_by_name(workspace, name=self.raw_name)
        df_raw = df_raw.to_pandas_dataframe()
        df_raw.rename(columns={'Column1': 'index'}, inplace=True)
        return df_raw, df_target

    # =========================================================================
    #     Establish a correspondence Between Raw and Target
    # =========================================================================
    def Mapping_Raw_to_Target(self, df_raw, df_target, col1, col2):
        
        ''' Raw and Target dataset are not matched one-to one correspondence.
            By combination of index and groups in Raw and Target dataset, 
            a new Key has been created, as Index_Groups to map and link 
            the Target to the Raw data.'''
         
#        df_raw[col1].astype(float) # Raw Index
#        df_raw[col2].astype(float) #Raw Group
#        df_target[col1].astype(float) # Target Index
#        df_target[col2].astype(float) #Target Group
#       
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
        if type(col_to_del) ==list:
            for col in col_to_del:
                df.pop(col)
        elif type(col_to_del) == str:
                df.pop(col_to_del)
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
        Distance_sign = str((t2-t1)) # Find To be forward or to be behind. 
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
    #     Remove Duplicate Columns
    # =============================================================================
    def Transpose (self, df):
        ''' To remove duplicate columns'''
        
        Trans = df.T
        Trans.drop_duplicates( inplace = True)
        Trans = Trans.T
        return Trans 
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
    # =========================================================================
    #     Split Dataset to Train and Test
    # =========================================================================   
    def Split (self, df, split_percent):
        ''' Split the Dataset into Test and Train'''
        # X_train, X_test, Y_train, Y_test = train_test_split(df, test_size=split_percent, random_state=10)
        X_train, X_test = train_test_split(Converted_Dataset, test_size = split_percent, random_state=223)
        return X_train, X_test

# =============================================================================
#  Main Project
# =============================================================================

target_name = 'Target'
raw_name = 'Raw data'
data_pre = DataPrepration(raw_name,target_name)
df_raw, df_target = data_pre.Get_Data()
# Converted_Dataset.drop("Index_group",axis = 1, inplace =  True
# Join the Target and Feature Data 
Converted_Dataset = data_pre.Mapping_Raw_to_Target(df_raw, df_target, 'index', 'groups')
print('Mapping Raw and Target dataset has done')

# Drop the high missed columns
col_to_del = ['etherium_before_start', 'start_critical_subprocess1','raw_kryptonite', 'pure_seastone', 'opened']
Converted_Dataset = data_pre.Drop_missed_Column(Converted_Dataset, col_to_del)
print("Drop the high missed columns has done")


# Unify the Date time columns
col_to_convert = [ 'expected_start', 'start_process', 'start_subprocess1',
                   'predicted_process_end', 'process_end', 'subprocess1_end',
                   'reported_on_tower']
Converted_Dataset = data_pre.Unifying_timestamp_col(Converted_Dataset, col_to_convert)
print ("Unify the Date time columns has done")

# Create New KPIs (Feature)
Converted_Dataset = data_pre.New_KPI(Converted_Dataset, 'Expected_Start_Process', 'expected_start', 'start_process')
Converted_Dataset = data_pre.New_KPI(Converted_Dataset, 'start_time_of_subprocess1', 'start_process', 'start_subprocess1' )
Converted_Dataset = data_pre.New_KPI(Converted_Dataset, 'process_duration','start_process', 'process_end')
Converted_Dataset = data_pre.New_KPI(Converted_Dataset, 'expected_end_process', 'process_end', 'predicted_process_end' )
Converted_Dataset = data_pre.New_KPI(Converted_Dataset, 'subprocess_duration', 'start_process', 'subprocess1_end' )
Converted_Dataset = data_pre.New_KPI(Converted_Dataset, 'whole_process', 'start_process', 'reported_on_tower')
print ('Create Nea Features has done')

# Convert Categorical to Numerical columns
col_to_numerical = ["super_hero_group","crystal_type", "crystal_supergroup", "Cycle"]
Converted_Dataset = data_pre.Catagorical_to_Numerical (Converted_Dataset, col_to_numerical)
print('Convert Categorical to Numerical columns has done')

# Remove the old and un-used columns from the DataSet
col_to_del = [ 'when', 'expected_start', 'start_process', 'start_subprocess1',
                   'predicted_process_end', 'process_end', 'subprocess1_end',
                   'reported_on_tower']
Converted_Dataset = data_pre.Drop_missed_Column(Converted_Dataset, col_to_del)
print('Remove un-used columns has done')

X_train, X_test = data_pre.Split(Converted_Dataset, 0.1)
print ('Split dataset to train and test has done')


# =============================================================================
# Config and run AML
# =============================================================================
import logging
x_train, x_test = train_test_split(Converted_Dataset, test_size=0.2, random_state=223)
automl_settings = {
    "iteration_timeout_minutes": 10,
    "experiment_timeout_hours": 0.3,
    "enable_early_stopping": True,
    "primary_metric": 'spearman_correlation',
    "featurization": 'auto',
    "verbosity": logging.INFO,
    "n_cross_validations": 5
}

from azureml.train.automl import AutoMLConfig

automl_config = AutoMLConfig(task='regression',
                             debug_log='automated_ml_errors.log',
                             training_data=x_train,
                             label_column_name="target",
                             **automl_settings)

from azureml.core.workspace import Workspace
ws = Workspace.from_config()
from azureml.core.experiment import Experiment
experiment = Experiment(ws, "CodeChallenge")
local_run = experiment.submit(automl_config, show_output=True)

from azureml.widgets import RunDetails
RunDetails(local_run).show()

best_run, fitted_model = local_run.get_output()
print(best_run)
print(fitted_model)

import pandas_profiling
profile = pandas_profiling.ProfileReport(Converted_Dataset, title="Pandas Profiling Report")
profile.to_widgets()
profile.to_notebook_iframe()

sum_actuals = sum_errors = 0

for actual_val, predict_val in zip(y_actual, y_predict):
    abs_error = actual_val - predict_val
    if abs_error < 0:
        abs_error = abs_error * -1

    sum_errors = sum_errors + abs_error
    sum_actuals = sum_actuals + actual_val

mean_abs_percent_error = sum_errors / sum_actuals
print("Model MAPE:")
print(mean_abs_percent_error)
print()
print("Model Accuracy:")
print(1 - mean_abs_percent_error)