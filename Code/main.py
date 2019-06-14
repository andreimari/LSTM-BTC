# -*- coding: utf-8 -*-

"""
Created on Mon Apr 15 20:04:14 2019

@author: Andrei
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Model import Model
from DataProcessingSimple import DataSimple
from sklearn.metrics import mean_squared_error
from DataAquisition import getDataDaily,getDataHourly
from GridSearch import grid_search
import json
import math
import os


def plot_simple(predicted_data,real_data):
    figure = plt.figure(facecolor = 'white')
    sub = figure.add_subplot(111)
    sub.plot(real_data, label = 'Real Data')
    plt.plot(predicted_data, label = 'Predicted data')
    plt.ylabel('Price')
    plt.legend()
    plt.show


def plot_multiple_points(predicted_data, real_data, prediction_len):
    figure = plt.figure(facecolor = 'white')
    sub = figure.add_subplot(111)
    sub.plot(real_data, label = 'Real data')
    for i,data in enumerate(predicted_data):
        padding = [None for p in range(i*prediction_len)]
        plt.plot(padding + data, label = 'Predicted data')
        plt.legend()
    plt.show()


def inMemory(data,model,config,epochs = None, batch_size = None):
    X, y = data.get_training_data(seq_len = config['data']['seq_len'],
        preprocess = config['data']['preprocess'])
    
    if config['model']['select'] == "yes":
        model.train(X, y, epochs = epochs,
                    batch_size = batch_size,
                    save_dir = config['model']['save_dir'])
    elif config['model']['select'] == "no":
         model.train(X, y, epochs = config['training']['epochs'],
                    batch_size = config['training']['batch'],
                    save_dir = config['model']['save_dir'])


def outMemory(data,model,config,epochs = None, batch_size = None):
    if config['model']['select'] == "yes":
        steps_per_epoch = math.ceil((data.len_train-config['data']['seq_len']) / batch_size)
        data_gen = data.generate_train_batch(seq_len = config['data']['seq_len'],
                   batch_size = batch_size, preprocess = config['data']['preprocess'])

        model.train_generator(data_gen,
                              epochs = epochs,
                              batch_size = batch_size,
                              steps_per_epoch = steps_per_epoch,
                              save_dir = config['model']['save_dir'])
    
    elif config['model']['select'] == "no":
        steps_per_epoch = math.ceil((data.len_train-config['data']['seq_len']) / config['training']['batch'])
        data_gen = data.generate_train_batch(seq_len = config['data']['seq_len'],
                   batch_size = config['training']['batch'], preprocess = config['data']['preprocess'])

        model.train_generator(data_gen,
                              epochs = config['training']['epochs'],
                              batch_size = config['training']['batch'],
                              steps_per_epoch = steps_per_epoch,
                              save_dir = config['model']['save_dir'])
    
        
    
def make_prediction(model, config, X_test, y_test):
    prediction_type = config["data"]["prediction_type"]
    if prediction_type == "single":
        prediction = model.predict_point(X_test)
    elif prediction_type == "full":
        prediction = model.predict_full(X_test,config['data']['seq_len'] )
    elif prediction_type == "multi_sequences":
        prediction = model.predict_seq_full(X_test,config['data']['seq_len'])
    
    prediction = np.array(prediction)

    if config["data"]["preprocess"] == "standardise":
        prediction = DataSimple.standardise_data(None,data = np.array(prediction[:,np.newaxis]),hint=2)
        y_test = DataSimple.standardise_data(None, data = y_test, hint=2)
    else:
        prediction = DataSimple.normalise_data(None,data = np.array(prediction[:,np.newaxis]),hint=2)
        y_test = DataSimple.normalise_data(None, data = y_test, hint=2)
    
    
    if prediction_type == "single":
        plot_simple(prediction[:,0], y_test[:,0])
        print(math.sqrt(mean_squared_error(y_test[:,0],prediction[:,0])))
    elif prediction_type == "full":
        plot_simple(prediction[:,0], y_test[:,0])
        print(math.sqrt(mean_squared_error(y_test[:,0],prediction[:,0])))
    elif prediction_type == "multi_sequences":
        plot_multiple_points(prediction[:,0].tolist(),y_test[:,0],config['data']['seq_len'])
        
        
    return prediction[:,0], y_test[:,0]


def main():
    config = json.load(open('config.json','r'))
    
    if config['data']['time'] == 'hourly':
        getDataHourly()
    elif config['data']['time'] =='daily':
        getDataDaily()  
    
    data = DataSimple(
            os.path.join('data',config['data']['filename']),
            config['data']['train_test_split'],
            config['data']['columns'],
            config['data']['seq_len'])
    
    time = data.time
    
    model = Model()
    
    if config["model"]["select"] == "yes":
        parameters = grid_search(data, config)
        loss = parameters['loss']
        optimizer = parameters['optimizer']
        epochs = parameters['epochs']
        batch_size = parameters['batch_size']
        model.build_model(config,loss = loss,optimizer = optimizer)
        if config['training']['mode'] == 'in':
            inMemory(data,model,config,epochs,batch_size)
        elif config['training']['mode'] == 'out':
            outMemory(data,model,config,epochs,batch_size)
    elif config["model"]["select"] == "no":
         model.build_model(config)
         if config['training']['mode'] == 'in':
            inMemory(data,model,config)
         elif config['training']['mode'] == 'out':
            outMemory(data,model,config)   
            
    X_test, y_test = data.get_test_data(
            seq_len = config['data']['seq_len'],
            preprocess = config['data']['preprocess'])
    
    results = make_prediction(model, config, X_test, y_test)
    real_data = results[1]
    predicted = results[0]
    
    dataset = pd.DataFrame({'Time':time, 'RealData':real_data,'Predicted':predicted})
    #dataset.to_csv("Data/result.csv", index=False)
    dataset.to_json('Data/result.json')
    
    
    return dataset

if __name__ == "__main__":
    main()
    