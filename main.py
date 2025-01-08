##import pathlib

##import numpy as np
##import matplotlib.pyplot as plt
import tensorflow as tf
import keras as ks
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import time
import pandas as pd
train_csv_data = pd.read_csv('./Data/train_energy_data.csv')
test_csv_data = pd.read_csv('./Data/test_energy_data.csv')
BUILDING_TYPE = ['Residential','Commercial','Industrial']
USE_CASE = ['Weekday','Weekend']
Building_mapping = {BUILDING_TYPE[0]: 0, BUILDING_TYPE[1]: 1, BUILDING_TYPE[2]: 2}  
use_case_mapping = {USE_CASE[0]: 0, USE_CASE[1]: 1}

train_csv_data['Building Type'] = train_csv_data['Building Type'].replace(Building_mapping)
train_csv_data['Day of Week'] = train_csv_data['Day of Week'].replace(use_case_mapping)

test_csv_data['Building Type'] = test_csv_data['Building Type'].replace(Building_mapping)
test_csv_data['Day of Week'] = test_csv_data['Day of Week'].replace(use_case_mapping)

train_x = train_csv_data.drop(columns=['Energy Consumption'])
train_y = train_csv_data['Energy Consumption']


test_x = test_csv_data.drop(columns=['Energy Consumption'])
test_y = test_csv_data['Energy Consumption']

train_x = train_x.to_numpy()
train_y = train_y.to_numpy().reshape(-1, 1)
test_x = test_x.to_numpy()
test_y = test_y.to_numpy().reshape(-1, 1)

feature_scaler = StandardScaler()
train_x_scaled = feature_scaler.fit_transform(train_x)
test_x_scaled = feature_scaler.transform(test_x)
label_scaler = MinMaxScaler()
train_y_scaled = label_scaler.fit_transform(train_y.reshape(-1, 1))
test_y_scaled = label_scaler.transform(test_y.reshape(-1, 1))
try:
    model = tf.keras.models.load_model('./Data/model.keras', compile=True)
except:
    log_dir = './logs/fit/3*100'
    Layers = 100
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(Layers, activation='relu',input_shape=(6,)))
    model.add(tf.keras.layers.Dense(Layers, activation='relu'))
    model.add(tf.keras.layers.Dense(Layers, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))





    early_stop = tf.keras.callbacks.EarlyStopping(monitor='MAE', patience=5,
                                              restore_best_weights=True, mode='min')

    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['MAE'])
    model.fit(train_x_scaled, train_y_scaled, epochs=200, batch_size=2, verbose=1, callbacks=[early_stop, tensorboard])



model.summary()
loss,MAE = model.evaluate(test_x_scaled,test_y_scaled,batch_size=2)

y_min = min(test_y)
y_max = max(test_y)

MAE_original = MAE * (y_max - y_min) / 2
accuracy = 1-(MAE_original/(y_max-y_min))
print("MAE when rescalled is : {MAE_original} And Relative accuracy is : {accuracy}".format(MAE_original=MAE_original, accuracy=accuracy))




