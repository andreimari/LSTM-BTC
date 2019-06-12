# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 14:29:47 2019

@author: Andrei
"""

import numpy as np
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime as dt
import os

class Model():
    
    def __init__(self):
        self.model = Sequential()
        
    def load_model(self, filename):
        self.model = load_model('01052019-165453-e2.h5')
    
    def build_model(self, config, loss = None, optimizer = None):
        for layer in config['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else 0
            dropout = layer['dropout'] if 'dropout' in layer else 0
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else False
            input_timestamps = layer['input_timestamps'] if 'input_timestamps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else 0 
            
            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation = activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timestamps, input_dim), return_sequences = return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(rate = dropout))
                
        if config['model']['select'] == "yes":
            self.model.compile(loss = loss, optimizer = optimizer)
        elif config['model']['select'] == "no":
            self.model.compile(loss = config['model']['loss'], optimizer = config['model']['optimizer'])
        
        
    def train(self, X, y, epochs, batch_size, save_dir):
        callbacks = [EarlyStopping(monitor = 'loss', patience =2), ModelCheckpoint(filepath = 'model1', monitor = 'val_loss', save_best_only=True)]
        self.model.fit(X, y, epochs = epochs, batch_size = batch_size, callbacks = callbacks)
        self.model.save("model1")
   
    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
        save_name = os.path.join(save_dir,'%s-e%s.h5' % (dt.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [ModelCheckpoint(filepath = save_name, monitor = 'loss', save_best_only = True)]
        self.model.fit_generator(data_gen, steps_per_epoch = steps_per_epoch, epochs = epochs, callbacks = callbacks, workers=1)
        
    def predict_point(self,data):
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size, ))
        return predicted 
    
    def predict_full(self, data, window_size):
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[np.newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame,[window_size-2], predicted[-1], axis = 0)
        return predicted
    
    def predict_seq_full(self, data,seq_len):
        pred_seq = []
        for i in range(int(len(data)/seq_len)):
            curr_frame = data[i*seq_len]
            predicted = []
            for j in range(seq_len):
                predicted.append(self.model.predict(curr_frame[np.newaxis,:,:])[0,0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [seq_len-2],predicted[-1], axis=0)
            pred_seq.append(predicted)
        return pred_seq
            
            
            
            
            
    