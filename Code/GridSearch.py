# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 20:49:57 2019

@author: Andrei
"""

from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.model_selection import GridSearchCV
import json

def create_model(optimizer = 'adam', loss = 'mse'):
        model = Sequential()
        config = json.load(open('config.json','r'))
        for layer in config['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else 0
            dropout = layer['dropout'] if 'dropout' in layer else 0
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else False
            input_timestamps = layer['input_timestamps'] if 'input_timestamps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else 0 
            
            if layer['type'] == 'dense':
                model.add(Dense(neurons, activation = activation))
            if layer['type'] == 'lstm':
                model.add(LSTM(neurons, input_shape=(input_timestamps, input_dim), return_sequences = return_seq))
            if layer['type'] == 'dropout':
                model.add(Dropout(dropout))
              
        model.compile(loss = loss, optimizer = optimizer,metrics=['accuracy'])
        
        return model
    
    
def grid_search(data, config):
        print("Grid search started...")
        X_train, y_train = data.get_training_data(seq_len = config['data']['seq_len'],
        preprocess = config['data']['preprocess'])
        model = KerasClassifier(build_fn= create_model, verbose=0)
        grid_param = { 
                'loss' : ['mse'],
                'optimizer': ['adam','SGD'],
                'epochs': [3],
                'batch_size': [16,32]
                }
        
        gd_sr = GridSearchCV(estimator=model,  
                     param_grid=grid_param,
                     cv = 3,
                     n_jobs=1)
        
        gd_sr.fit(X_train, y_train)
        print("Grid search ended...")
        best_parameters = gd_sr.best_params_  
        print(best_parameters)
        return best_parameters 