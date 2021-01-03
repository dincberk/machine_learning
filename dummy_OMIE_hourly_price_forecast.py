import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import xlrd
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot	
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
from math import sqrt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
import xlwings as xw
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from keras.layers import Dropout
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import time
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 
from statsmodels.tsa.seasonal import seasonal_decompose 
from pmdarima import auto_arima                        
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from statsmodels.tools.eval_measures import rmse
import warnings
from pandas import json_normalize
import requests
import datetime as dt
from datetime import timedelta
from datetime import datetime
from matplotlib import pyplot as plt
import time
import sys
import requests
from openpyxl import load_workbook
import xlwings as xw
import json
from sklearn.metrics import accuracy_score
import catboost
from catboost import CatBoostRegressor
import glob
import os
print("Setup Complete")
starting_time=time.time()

#####

CODE BEFORE THIS POINT IS A CONNECTION TO SEVERAL EXTERNAL APIs

CODE BELOW HERE INCLUDES 3 NON-MODIFIED DUMMY MACHINE LEARNING ALGORITHMS: Catboost, XGBoost, DecisionTree

#####

appended_df=[]
frames=[ree_generation_capacity,ree_p48,M_spot_prices]

appended_df=pd.concat(frames,axis=1,sort=True)
appended_df.index=pd.to_datetime(appended_df.index, format='%Y-%m-%d %H:%M',utc=True)

appended_df
#ID NUMBERS OF ESIOS for the given variables to retrieve them with a HTTPS requrest to RED ELECTRICA's API
#10010: Wind p48
#84: SolarPV p48
#10027: Total Demand p48 ,'510','514'
#472:Hydro availability
#474: Nuclear availability
#475: Fossil hard availability
#477: CCGT availability
#10001: Total availability
#552: Powernext France power price real
#2584: OMEI Spain power price real

X_forecast_data
#28223: M wind production forecast
#28220: M solar production forecast
#38848: E demand forecast
#41377: France price forecast  

appended_df=appended_df.tz_localize(None)
tomorrow=pd.Timestamp(dt.datetime.now()+timedelta(days=1),tz=None)
tomorrow=tomorrow.replace(hour=0,minute=0,second=0,microsecond=0)
ree_past=appended_df.loc[appended_df.index < tomorrow]
ree_future=appended_df.loc[appended_df.index >= tomorrow]
ree_future=ree_future.filter(items=['472','474','475','477','10001'])
M_forecast_data=M_forecast_data.tz_localize(None)
M_forecast_data=M_forecast_data.loc[M_forecast_data.index >= tomorrow]
test_frames=[M_forecast_data,ree_future]
test_data=pd.concat(test_frames,axis=1)

test_data['2584']=''
test_data['2584']='1'
test_data['2584']=test_data['2584'].astype(int)

#Historical data includes a column where LOCKDOWN dates of Madrid/Spain is present. For the future, it is represented as 0 in the test data
test_data['Covid']=''
test_data['Covid']='0'
test_data['Covid']=test_data['Covid'].astype(int)
train_data=ree_past
train=train_data.rename(columns={'472':'HydroAva','474':'NucAva','475':'FosHardAva','477':'CcgtAva','10001':'TotalAva','10010':'Wind','84':'SolarPV','10027':'Demand','552':'FR_SPOT','2584':'ES_SPOT'})
test=test_data.rename(columns={'28223':'Wind','28220':'SolarPV','38848':'Demand','41377':'FR_SPOT','472':'HydroAva','474':'NucAva','475':'FosHardAva','477':'CcgtAva','10001':'TotalAva','2584':'ES_SPOT'})
train_live=train

train=pd.read_csv('train_data_file.csv',delimiter=';',decimal=',',error_bad_lines=False)
train.index=pd.to_datetime(train.index, format='%Y-%m-%d %H:%M',utc=True)
train=train.set_index('datetime')

train=train.append(train_live,ignore_index=False)
train=train[['ES_SPOT','FR_SPOT','Demand','Wind','SolarPV','TotalAva','HydroAva','NucAva','CcgtAva','FosHardAva','Covid']]
test=test[['ES_SPOT','FR_SPOT','Demand','Wind','SolarPV','TotalAva','HydroAva','NucAva','CcgtAva','FosHardAva','Covid']]

test_cutoff=pd.Timestamp(dt.datetime.now()+timedelta(days=10),tz=None)
test=test.loc[test.index < test_cutoff]


#Data editing----------------------------

test['ES_SPOT']=np.nan_to_num(test['ES_SPOT'])
train=train.dropna()
test=test.dropna()

target_column_train=['ES_SPOT']
predictors_train = list(set(list(train.columns))-set(target_column_train))
X_train = train[predictors_train]
y_train = train[target_column_train]

#EXLUDING SPOT PRICE FROM TEST DATA
target_column_test = ['ES_SPOT'] 
predictors_test = list(set(list(test.columns))-set(target_column_test))
X_test = test[predictors_test]
y_test = test[target_column_test]

def tree():
    target_column_train=['ES_SPOT']
    predictors_train = list(set(list(train.columns))-set(target_column_train))
    X_train = train[predictors_train]
    y_train = train[target_column_train]
    #EXLUDING SPOT PRICE FROM TEST DATA
    target_column_test = ['ES_SPOT'] 
    predictors_test = list(set(list(test.columns))-set(target_column_test))
    X_test = test[predictors_test]
    y_test = test[target_column_test]
    #Decision TREE---------------------------------------------------
    dtree = DecisionTreeRegressor( min_samples_leaf=0.001, random_state=3)
    dtree.fit(X_train, y_train)

    DecisionTreeRegressor(criterion='mae', max_features=None,
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=0.001,
            min_samples_split=10, min_weight_fraction_leaf=0.0,
            presort=False, random_state=3, splitter='best')
    pred_train_tree= dtree.predict(X_train)
    train_mae=mean_absolute_error(y_train,pred_train_tree)
    #print(r2_score(y_train, pred_train_tree))
    print('Decision Tree Train MAE is: {:.3f} '.format(train_mae))
    pred_test_tree= dtree.predict(X_test)
    #test_mae=mean_absolute_error(y_test,pred_test_tree)
    #print(r2_score(y_test, pred_test_tree))
    #print('Decision Tree Test MAE is: {:.3f} '.format(test_mae))
    #Converting array to DataFrame-----------------------------------
    y_test=pd.DataFrame(y_test,columns=['Real'])
    pred_test_tree=pd.DataFrame(pred_test_tree,columns=['Decision Tree Prediction'])
    y_test.index=test.index
    pred_test_tree.index=test.index
    #y_test['Prediction']=pred_test_tree['Prediction']
    #test.index=pd.to_datetime(test.index, format='%Y-%m-%d %H:%M',utc=True)
    #test.index=test.index.strftime('%d')
    file_path='forecasts.xlsx'
    wb=xw.Book(file_path)
    dt=wb.sheets['Outputs']
    dt.range('C1').options(pd.DataFrame, ignore_index=False).value=pred_test_tree   
    dt.range('T4').value=train_mae

tree()


def boost():
    predictors_train = list(set(list(train.columns))-set(target_column_train))
    X_train = train[predictors_train]
    y_train = train[target_column_train]
    #EXLUDING SPOT PRICE FROM TEST DATA
    target_column_test = ['ES_SPOT'] 
    predictors_test = list(set(list(test.columns))-set(target_column_test))
    X_test = test[predictors_test]
    y_test = test[target_column_test]

    reg = xgb.XGBRegressor(random_state=0,n_estimators=100,max_depth=5,min_child_weight=1,scale_pos_weight=1)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=10,
        verbose=True,eval_metric='mae')
    
    #_ = plot_importance(reg, height=0.9)
    pred_train_xgboost=reg.predict(X_train)
    pred_test_xgboost=reg.predict(X_test)
    train_mae=mean_absolute_error(y_train,pred_train_xgboost)
    #print(r2_score(y_train, pred_train_xgboost))
    print('Xgboost Train MAE is: {:.3f} '.format(train_mae))
    #test_mae=mean_absolute_error(y_test,pred_test_xgboost)
    #print(r2_score(y_test, pred_test_xgboost))
    #print('Xgboost Test MAE is: {:.3f} '.format(test_mae))
    pred_test_xgboost=pd.DataFrame(pred_test_xgboost,columns=['Xgboost Prediction'])
    pred_test_xgboost.index=y_test.index

    file_path='forecasts.xlsx'
    wb=xw.Book(file_path)
    xg=wb.sheets['Outputs']
    xg.range('E1').options(pd.DataFrame, ignore_index=False).value=pred_test_xgboost
    xg.range('U4').value=train_mae

boost()

def cat():
    predictors_train = list(set(list(train.columns))-set(target_column_train))
    X_train = train[predictors_train]
    y_train = train[target_column_train]
    #EXLUDING SPOT PRICE FROM TEST DATA
    target_column_test = ['ES_SPOT'] 
    predictors_test = list(set(list(test.columns))-set(target_column_test))
    X_test = test[predictors_test]
    y_test = test[target_column_test]

    model = CatBoostRegressor(iterations=5000,learning_rate=0.1,depth=6,eval_metric='RMSE')
    model.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],verbose=False)
    
    #_ = plot_importance(reg, height=0.9)
    pred_train_catboost=model.predict(X_train)
    pred_test_catboost=model.predict(X_test)
    train_mae=mean_absolute_error(y_train,pred_train_catboost)
    #print(r2_score(y_train, pred_train_xgboost))
    print('Catboost Train MAE is: {:.3f} '.format(train_mae))
    #test_mae=mean_absolute_error(y_test,pred_test_xgboost)
    #print(r2_score(y_test, pred_test_xgboost))
    #print('Xgboost Test MAE is: {:.3f} '.format(test_mae))
    pred_test_catboost=pd.DataFrame(pred_test_catboost,columns=['Catboost Prediction'])
    pred_test_catboost.index=y_test.index

    file_path='forecasts.xlsx'
    wb=xw.Book(file_path)
    xg=wb.sheets['Outputs']
    xg.range('G1').options(pd.DataFrame, ignore_index=False).value=pred_test_catboost
    xg.range('V4').value=train_mae
    
cat()


