# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 09:25:12 2023

@author: Wai Yip LIEW (liewwy19@gmail.com)
"""

# %% 
#0. Import Packages
import numpy as np
import pandas as pd
import os, datetime, pickle
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

#   Variables
SEED = 142857
CSV_PATH = os.path.join(os.getcwd(),'Datasets','cases_malaysia_train.csv')
CSV_PATH_TEST = os.path.join(os.getcwd(),'Datasets','cases_malaysia_test.csv')
LOG_PATH = os.path.join(os.getcwd(),'logs',datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
SAVED_MODEL_DIR = os.path.join(os.getcwd(),'Saved_model')

#   Functions
def load_data(data_path, data_col='cases_new'):
    '''
        This function doing data loading and data cleaning.
        Return 2D numpy array of the loaded data
    '''
    # load data from csv
    loaded_data = pd.read_csv(data_path)

    # convert dbtype object to numeric 
    print('[INFO] Dtype BEFORE (data cleaning):', loaded_data[data_col].dtype)
    loaded_data[data_col] = pd.to_numeric(loaded_data['cases_new'],errors='coerce')
    print('[INFO] Dtype AFTER (data cleaning):', loaded_data[data_col].dtype)

    # to replace NaNs using Interpolation approach
    print('[INFO] NaN Counts BEFORE (Interpolation):',loaded_data[data_col].isna().sum())
    loaded_data[data_col] = loaded_data[data_col].interpolate(method='polynomial',order=2).astype('int64')
    print('[INFO] NaN Counts AFTER (Interpolation):',loaded_data[data_col].isna().sum())

    # keep only the data_col and perform expand dimension
    loaded_data = loaded_data[data_col].values[::,None]

    return loaded_data 

def split_data(input_dataset, win_size=30):
    '''
        This function split data based on windows size.
        Return X and y (both in Numpy Array format)
    '''
    X = []
    y = []
    
    for i in range(win_size, len(input_dataset)):
        X.append(input_dataset[i-win_size:i])
        y.append(input_dataset[i])

    X = np.array(X)
    y = np.array(y)

    return X,y

# %%
#1. Data Preparation
#   data loading
data = load_data(CSV_PATH)

#   data inspection
'''
    Observation:
    - 680 entries
    - 31 columns
    - feature => cases_new (Dtype = Object)
    - cases_new => 12 NAN (after converted to numeric) 

'''
       
#   quick visual check of training data
plt.figure()
plt.plot(data)
ax = plt.show()

#2. Features Selection
#   None needed as we only focus on one column ==> cases_new
#   data is now numpy array with shape (680,1)


# %%
#3. Data Preprocessing
#   apply MinMaxScaler to train dataset
mms = MinMaxScaler()
data = mms.fit_transform(data)

# %%
#   data spliting
X,y = split_data(data)
#%%
#   Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True,random_state=SEED)

# %%
#4. Model Development
#   model creation
model = Sequential()
model.add(LSTM(64,return_sequences=True, input_shape=(X_train.shape[1:])))
model.add(Dropout(0.3))  # added to reduce overfitting
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(0.3)) # added to reduce overfitting
model.add(LSTM(64))
model.add(Dropout(0.5)) # added to reduce overfitting
model.add(Dense(1))

#   generate model summary
tf.keras.utils.plot_model(model,show_shapes=True)

# %%
#   model compilation
model.compile(optimizer='adam',loss='mse',metrics=['mse','mae'])

#   define callback functions
tb = TensorBoard(log_dir=LOG_PATH)
es = EarlyStopping(monitor='mae',patience=8, verbose=1,restore_best_weights=True) 
mc = ModelCheckpoint(os.path.join(SAVED_MODEL_DIR,'best_model_chkpt.h5'), monitor='mae', mode='min', verbose=1, save_best_only=True)  

#   model training
EPOCHS = 50
BATCH_SIZE = 64
history = model.fit(X_train,y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,callbacks=[tb,es,mc])
# %%
#5. Model Analysis/ Evaluation
#   Load test data
test_data = load_data(CSV_PATH_TEST)

#   mms tranform test data
test_data = mms.transform(test_data)

# concatenate test and train data
win_size = X_train.shape[1]
test_data = np.concatenate((data[-win_size:],test_data)) 

#%%
#   test data splitting
X_test2,y_test2 = split_data(test_data)

#   model deployment
predicted_new_cases = model.predict(X_test2)

# %%
#   inversing the normalized data
inversed_predicted = np.round(mms.inverse_transform(predicted_new_cases))
inversed_actual = mms.inverse_transform(y_test2)

# %%
#   visualization
plt.figure()
plt.plot(inversed_actual, color='red', label='Actual Cases')
plt.plot(inversed_predicted, color='blue', label='Predicted Cases')
plt.legend()
plt.xlabel('Time (Day)')
plt.ylabel('Daily New Covid-19 Cases')
ax = plt.show()

# %%
#   metrics
print(f'mape: {mean_absolute_percentage_error(inversed_actual, inversed_predicted):.6f}') # actual, predicted
print(f'mae: {mean_absolute_error(inversed_actual, inversed_predicted):.2f}')
print(f'rmse: {mean_squared_error(inversed_actual, inversed_predicted,squared=False):.2f}')

# %%
#6. Model Saving
#   save trained model
model.save(os.path.join(SAVED_MODEL_DIR,'model.h5')) # save train model

#   save scaler model
with open(os.path.join(SAVED_MODEL_DIR,'mms.pkl'),'wb') as f:
    pickle.dump(mms,f)


# %%
# @author: Wai Yip LIEW (liewwy19@gmail.com)