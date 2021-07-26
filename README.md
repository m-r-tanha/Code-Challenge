# Challenge
#
#

## Project Steps
1. **Data preparation**
    1. Map the Target Values to Raw Data
    2. Create New KPIs
    3. Delete Columns with a high proportion of missing values
    4. Convert Catagorical columns to Numerical values
    5. Selected Features
        - Select Features basd on thier variance
        - Remove Duplicate Columns
    7. Use Imputation to reGenerate the missed data
    8. Shuffling the Dataset
2. **Regression Model**
    1. Find the Best Algorithm for this problem
        - Run and evaluate different Regression Algorithms in **ML Azure** to find the Best Model
    2. Train the Model
    3. Use K-Fold
3. **Hyperparameters Tuning**
    1. Find the best XGBoost parametres
    2. Tune the Number of Selected Features
4. **Evaluate the Models**
    1. Mean Absolute Error
    2. Root Mean Squared Error
    3. Relative Squared Error
    4. Relative Absolute Error
    5. Coefficient of Determination
5. **Make Predictions**
    1. Predict and depict on X_test


## Map Raw Data to Target
The Target and Raw data Indexs are not "one-to-one correspondence"(the raw data has **9592** rows, but the target has **9589** rows), thus with a combination of index and groups, a new Key has been created, as **Index_Groups** to map and link the Target to the Raw data. Finally, the 8544 unique rows have been found:
```python
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
```
## Delete Columns with a high proportion of missing values
By reviewing the statistics for this column noting, there are quite a few values in this columns. They will limit their usefulness in predicting the target; so you might want to exclude it from training. These features are: 'etherium_before_start', 'start_critical_subprocess1','raw_kryptonite', and 'pure_seastone'.
![image](https://user-images.githubusercontent.com/42087252/126125435-6ad53acc-976a-4ad8-96ff-c6c26dfb72cc.png)
```python
df_target.isna().sum()
Converted_Detaset.drop(['etherium_before_start', 'start_critical_subprocess1','raw_kryptonite', 'pure_seastone', 'opened'],
            axis = 1, inplace = True)
```


## Unify the Feature date types
Some date values are in text type like (43223.56), or in different format date. they should detect and convert to date type the detection has been done by regex
```python
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
```
## Use Ordinal Encoding to change catagorical columns to numerical values
The Raw data include some **Categoricall Features**, and they should be convered to **Numerical** by OrdinalEncoder() function:
```python
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
Converted_Detaset[["super_hero_group","crystal_type", "crystal_supergroup", "Cycle"]] =  enc.fit_transform(Converted_Detaset[["super_hero_group","crystal_type", "crystal_supergroup", "Cycle"]])
```
## Create New KPIs
To make the better use of timing data or features they have been changed to the new KPIs, they have been defined based on duration, like below KPIs:

- <a href="https://www.codecogs.com/eqnedit.php?latex=Expected&space;\:&space;to\,&space;Start&space;\,&space;Duration\rightarrow&space;\left&space;(&space;Expected\,&space;Time&space;\circleddash&space;Start\,&space;Time&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Expected&space;\:&space;to\,&space;Start&space;\,&space;Duration\rightarrow&space;\left&space;(&space;Expected\,&space;Time&space;\circleddash&space;Start\,&space;Time&space;\right&space;)" title="Expected \: to\, Start \, Duration\rightarrow \left ( Expected\, Time \circleddash Start\, Time \right )" /></a>

- <a href="https://www.codecogs.com/eqnedit.php?latex=SubProcess\,&space;Delay&space;\,&space;After&space;\,&space;Start&space;\rightarrow&space;Start\,&space;SubProcess1&space;\circleddash&space;Start\,&space;Process" target="_blank"><img src="https://latex.codecogs.com/gif.latex?SubProcess\,&space;Delay&space;\,&space;After&space;\,&space;Start&space;\rightarrow&space;Start\,&space;SubProcess1&space;\circleddash&space;Start\,&space;Process" title="SubProcess\, Delay \, After \, Start \rightarrow Start\, SubProcess1 \circleddash Start\, Process" /></a>

- <a href="https://www.codecogs.com/eqnedit.php?latex=Process\,&space;Duration&space;\,&space;\rightarrow&space;Process\,&space;End&space;\circleddash&space;Start\,&space;Process" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Process\,&space;Duration&space;\,&space;\rightarrow&space;Process\,&space;End&space;\circleddash&space;Start\,&space;Process" title="Process\, Duration \, \rightarrow Process\, End \circleddash Start\, Process" /></a>

- <a href="https://www.codecogs.com/eqnedit.php?latex=Expected\,to\,&space;End\,&space;Duration&space;\rightarrow&space;Expected\,&space;End&space;\circleddash&space;Process\,&space;End" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Expected\,to\,&space;End\,&space;Duration&space;\rightarrow&space;Expected\,&space;End&space;\circleddash&space;Process\,&space;End" title="Expected\,to\, End\, Duration \rightarrow Expected\, End \circleddash Process\, End" /></a>

- <a href="https://www.codecogs.com/eqnedit.php?latex=SubProcess1&space;\,Duration&space;\rightarrow&space;SubProcess1\,&space;End&space;\circleddash&space;Start\,&space;SubProcess" target="_blank"><img src="https://latex.codecogs.com/gif.latex?SubProcess1&space;\,Duration&space;\rightarrow&space;SubProcess1\,&space;End&space;\circleddash&space;Start\,&space;SubProcess" title="SubProcess1 \,Duration \rightarrow SubProcess1\, End \circleddash Start\, SubProcess" /></a>

- <a href="https://www.codecogs.com/eqnedit.php?latex=Total\,&space;Process&space;\,Duration&space;\rightarrow&space;Time\,&space;to\,&space;Report&space;\circleddash&space;Start\,&space;Process" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Total\,&space;Process&space;\,Duration&space;\rightarrow&space;Time\,&space;to\,&space;Report&space;\circleddash&space;Start\,&space;Process" title="Total\, Process \,Duration \rightarrow Time\, to\, Report \circleddash Start\, Process" /></a></br>

```python
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
```
## Define new DataSet with new KPIs (Features)
```python
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
```
## Shuffling the Dataset
This is the tricky part, and it helps to have better performance, Since the original DataSet is somehow a periodical dataset, although it is not time-series, before dividing it into train and test, the dataset should be shuffled randomly. 
```python
X = X.sample(frac = 1)
Y = X.iloc[:,-1]
X = X.iloc[:,0:-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.015, random_state=10)
```
##  Use Imputation to reGenerate the missed data
If the rows with NAN values had been deleted, we had missed much information. For example, in this challenge, 10% (8544 - 7623 = 921) of data is missed if we use deleting missing values approach. In the below code, replacing NAN values with reasonable data is its goal.
![image](https://user-images.githubusercontent.com/42087252/126125435-6ad53acc-976a-4ad8-96ff-c6c26dfb72cc.png)
- Replacing the missing values:
```python
# define imputer
imputer = IterativeImputer()
# fit on the dataset
imputer.fit(X)
# transform the dataset
Xtrans = imputer.transform(X)
```
## Remove Columns that have the low Variance
variables with very few numerical values can also cause errors or unexpected results. On the other hand, the features with low variance may not contribute to the skill of a model. And often removing these feature help to have beter performance. Feature Count = 26 has the best performance.

![image](https://user-images.githubusercontent.com/42087252/126230820-892fa4b2-495d-4db9-bca9-244b385cc3d0.png)


```python
print(X.shape, Y.shape)
# define the transform
transform = VarianceThreshold(threshold=.2)
# transform the input data
X_sel = transform.fit_transform(X)
print(X_sel.shape)
X_sel = pd.DataFrame(X_sel)
```

## Find the best Regression Method by ML Azure
To share my **Azure Machine Learning** experience, this part of challenge has been done be Azure. and it goes without saying that **Boosted Decision Tree Regression** has the best performance. the following table shows the evalution.


![image](https://github.com/m-r-tanha/RHI-Challenge/blob/main/Regression%20Model.png)

|Evaluation Metrics| Mean Absolute Error | Root Mean Squared Error | Relative Squared Error | Relative Absolute Error | Coefficient of Determination | 
| :---:        |     :---:      |         :---: | :---:        |     :---:      |         :---: |
| Poisson Regression  | 0.077|	0.105|	0.245|	0.432|	0.755|
| Decision Forest Regression|	0.031|	0.044|	0.044|	0.175|	0.956|
|Neural Network Regression|	0.059|	0.079|	0.141|	0.332|	0.859|
|**Boosted Decision Tree Regression**	|**0.015**	|**0.021**	|**0.010**	|**0.086**	|**0.990**|
|Linear Regression|	0.056|	0.076|	0.129|	0.316|	0.871|

The following Boxplot depicts, Not only the "Boosted Decision Tree Regression" method has the lowest second Quartile but also does it perform persistently.

![image](https://user-images.githubusercontent.com/42087252/126204484-2791f4b6-6eb8-48b2-9807-4f7542f40eb1.png)

##  Find Hyperparameters (best number of estimators in XGBoost)
```python
regressor = DecisionTreeRegressor(random_state = 0)
for i in range (1, 700, 100):
    xgb_r = xg.XGBRegressor(objective ='reg:linear',
                      n_estimators = i, seed = 123)
    pipeline = Pipeline(steps=[('normalize', MinMaxScaler()), ('model', xgb_r)])
    # prepare the model with target scaling
    model = TransformedTargetRegressor(regressor=pipeline, transformer=MinMaxScaler())
    # evaluate model
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, Y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # convert scores to positive
    scores = absolute(scores)
    # summarize the result
    s_mean = mean(scores)
    print('n_estimators: %d , Mean MAE: %.3f' % (i, s_mean))
```
## Tune the Number of Selected Features
```python
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)

regressor = DecisionTreeRegressor(random_state = 0)
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
# summarize all
means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
for mean, param in zip(means, params):
    print('>%.3f with: %r' % (mean, param))
```
# Predict the sample data 
This code can be optimized and perfom in higher accuracy. But as a test I present it to prove my experience in Machine Learning Field.
```python

X = X.sample(frac = 1)
Y = X.iloc[:,-1]
X = X.iloc[:,0:-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01, random_state=1)

regressor = DecisionTreeRegressor(random_state = 0)
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

```

![image](https://github.com/m-r-tanha/RHI-Challenge/blob/main/X_Test2.png)


# Suggestion for next Steps/Further Improvements
1. Gathering or generat data in highr index values.
![image](https://user-images.githubusercontent.com/42087252/126552640-a8495a00-55e3-4313-97ec-f2915b356773.png)
2. I didn't work on **Outlier**, so it should be evaluated and work in improved code.
3. in above graph, prediction in a few indexs are far from the actual values, As a result, these area should be considerd, to have less difference between prediction and acual.
4. This challange can be done as Time-Series System and Deep Learning

