# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 22:42:48 2019
@author: Andrei
"""
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

class DataSimple():
    
    #methods used for transforming the initial data standardising and normalising
    std = StandardScaler()
    nrm = MinMaxScaler(feature_range=(-1, 1))
    
    #initialise all the components needed for data processing 
    def __init__(self, filename, divide, data,seq_len):
        data_frame = pd.read_csv(filename)
        split = int(len(data_frame)*divide)
        self.data_train = data_frame.get(data).values[:split]
        self.data_test = data_frame.get(data).values[split:]
        self.time = data_frame.get("time").values[split+seq_len:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)
        self.len_train_windows = None
    
    #get data for testing the model 
    def get_test_data(self, seq_len, preprocess):
        X_test = []
        y_test = []
        #take windows of seq length and preprocess them 
        for i in range(self.len_test - seq_len):
            X, y = self.get_window(i, seq_len,preprocess,"testing")
        #append the results     
            X_test.append(X)
            y_test.append(y)
        return np.array(X_test), np.array(y_test)
       
      ##get data for training the model 
    def get_training_data(self, seq_len,preprocess):
        X_train = []
        y_train = []
        #take windows of seq length and preprocess them 
        for i in range(self.len_train - seq_len):
        #append the results     
            X, y = self.get_window(i,seq_len,preprocess,"training")
            X_train.append(X)
            y_train.append(y)
        return np.array(X_train), np.array(y_train)
            
    #generate training data for the out of memory option 
    def generate_train_batch(self, seq_len, batch_size, preprocess):
        i=0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    yield np.array(x_batch), np.array(y_batch)
                    i=0
                x, y = self.get_window(i,seq_len,preprocess,"training")
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)
    
    #make the choice between normalise and standardise 
    def preprocess_data(self, preprocess,data):
        if preprocess == "normalise":
            data = self.normalise_data(data,hint=1)
            return data
        elif preprocess == "standardise":
            data = self.standardise_data(data,hint=1)
            return data
    
    #transform the data in x(for training, testing) and y(the desired output)
    def get_window(self, i, seq_len,preprocess,hint):
        if hint == "training":
            window = self.data_train[i:i+seq_len]
        elif hint == "testing" :
            window = self.data_test[i:i+seq_len]
        window = self.preprocess_data(preprocess, window)
        x = window[:-1]
        y = window[-1, [0]]
        return x,y 
         
    #function which standardises the window data 
    def standardise_data(self,data,hint=1):
        if hint == 1:
            data = DataSimple.std.fit_transform(data)
        else:
            data = np.append(data,data,axis=1)
            data = DataSimple.std.inverse_transform(data)
        return data
     
    #function which normalises the window data 
    def normalise_data(self,data,hint=1):
        if hint == 1:
            data = DataSimple.nrm.fit_transform(data)
        else:
            data = np.append(data,data,axis=1)
            data = DataSimple.nrm.inverse_transform(data)
        return data
          
    
      
         
                
            
            
        
        
    
        
        
        
    
        
        
        