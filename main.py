##DataSet taken from Kaggle.com under same name https://www.kaggle.com/datasets/govindaramsriram/energy-consumption-dataset-linear-regression/


import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pandas as pd
#import csv data and replaces words with certain numbers to start machine learning
train_csv_data = pd.read_csv('./Data/train_energy_data.csv')
test_csv_data = pd.read_csv('./Data/test_energy_data.csv')
BUILDING_TYPE = ['Residential','Commercial','Industrial']
USE_CASE = ['Weekday','Weekend']
Building_mapping = {BUILDING_TYPE[0]: 0, BUILDING_TYPE[1]: 1, BUILDING_TYPE[2]: 2}  
use_case_mapping = {USE_CASE[0]: 0, USE_CASE[1]: 1}
##uses pandas replace function to replace occurences
train_csv_data['Building Type'] = train_csv_data['Building Type'].replace(Building_mapping)
train_csv_data['Day of Week'] = train_csv_data['Day of Week'].replace(use_case_mapping)

test_csv_data['Building Type'] = test_csv_data['Building Type'].replace(Building_mapping)
test_csv_data['Day of Week'] = test_csv_data['Day of Week'].replace(use_case_mapping)

##divides the labels from the data
train_x = train_csv_data.drop(columns=['Energy Consumption'])
train_y = train_csv_data['Energy Consumption']

test_x = test_csv_data.drop(columns=['Energy Consumption'])
test_y = test_csv_data['Energy Consumption']


# makes it a numpy array so TensorFlow can use it
train_x = train_x.to_numpy()
train_y = train_y.to_numpy().reshape(-1, 1)
test_x = test_x.to_numpy()
test_y = test_y.to_numpy().reshape(-1, 1)

#use standard scaler to normalize everything
feature_scaler = StandardScaler()
train_x_scaled = feature_scaler.fit_transform(train_x)
test_x_scaled = feature_scaler.transform(test_x)

#uses minmax scaler to put all labels to [-1,1] So that MAE Makes more sense
label_scaler = MinMaxScaler()
train_y_scaled = label_scaler.fit_transform(train_y.reshape(-1, 1))
test_y_scaled = label_scaler.transform(test_y.reshape(-1, 1))
#trys to load preexisting model or creates a new one with log
try:
    model = tf.keras.models.load_model('./model.keras', compile=True)
except:
    #creats log with dir of layers * nodes
    log_dir = './logs/fit/3*100'
    Layers = 100
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    #creates a model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(Layers, activation='relu',input_shape=(6,)))
    model.add(tf.keras.layers.Dense(Layers, activation='relu'))
    model.add(tf.keras.layers.Dense(Layers, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))




    #early stop function
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='MAE', patience=5,
                                              restore_best_weights=True, mode='min')
    #trains model
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['MAE'])
    model.fit(train_x_scaled, train_y_scaled, epochs=200, batch_size=2, verbose=1, callbacks=[early_stop, tensorboard])



#gives the summary
model.summary()
loss,MAE = model.evaluate(test_x_scaled,test_y_scaled,batch_size=2)

y_min = min(test_y)
y_max = max(test_y)
#gives scaled accuracy and MAE
MAE_original = MAE * (y_max - y_min) / 2
accuracy = 1-(MAE_original/(y_max-y_min))
print("MAE when rescalled is : {MAE_original} And Relative accuracy is : {accuracy}".format(MAE_original=MAE_original, accuracy=accuracy))
model.save('./model.keras')



