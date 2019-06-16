# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 01:40:11 2019

@author: Andrei
"""
import requests
import pandas as pd
from datetime import datetime,timedelta
import numpy as np


'''method in which we get the data from an API on a daily basis and store it in a CSV file'''
def getDataDaily():

    '''api request allows for max 2000 items'''
    now = datetime.today()
    then = now - timedelta(days=2001)
    stamp = int(datetime.timestamp(then))
    api_req = 'https://min-api.cryptocompare.com/data/histoday?fsym=BTC&tsym=USD&limit=2000&toTs=' + str(stamp)
    
    '''get the data from 2 api GET requests and store them in a data list'''
    r = requests.get(api_req)
    data = r.json()['Data']
    r = requests.get('https://min-api.cryptocompare.com/data/histoday?fsym=BTC&tsym=USD&limit=2000')
    data.extend(r.json()['Data'])
    
    '''after storing the data in a list we convert it to a dataframe to delete all null values'''   
    data = pd.DataFrame(data)
    data.replace(0, np.nan, inplace=True)
    data= data.dropna()
    data.reset_index(drop=True, inplace=True)
    
    '''change from timestamp format to year/month/day format'''
    df = []
    for column in data['time']:
            df.append({'time':datetime.fromtimestamp(column).date()})

    '''change the order of the columns in the list and also change the name of the columns'''
    data= data[['time','close','open','low','high','volumeto','volumefrom']]    
    data['volumeBTC'] = data.pop('volumeto')
    data['volumeUSD'] = data.pop('volumefrom')
    
    '''update the current data list with the new values for the time columns and create the csv file'''
    data.update(df) 
    data.to_csv('Code/Data/data.csv',index=False)


'''method in which we get the data from an API on a hourly basis and store it in a CSV file'''
def getDataHourly():
 
    '''make an api request to get the latest timestamp'''
    data = []
    stamp = None
    ok = True
    r = requests.get('https://min-api.cryptocompare.com/data/histohour?fsym=BTC&tsym=USD&limit=2000')
    x = r.json()['TimeTo'] - r.json()['TimeFrom']
    y = r.json()['TimeTo']

    '''see how far the timestamp can go until it reaches null values'''
    while ok==True:
        stamp = r.json()['TimeTo'] - x -3600
        api_req = 'https://min-api.cryptocompare.com/data/histohour?fsym=BTC&tsym=USD&limit=2000&toTs='+str(stamp)
        r = requests.get(api_req)
        if r.json()['Data'][0]['close'] == 0 and r.json()['Data'][0]['open']== 0:
            ok = False
    
    '''after the lastes timestamp is found we go backwards and store the data in a list'''
    while ok == False:
         r = requests.get(api_req)
         data.extend(r.json()['Data'])
         stamp = r.json()['TimeTo'] + x + 3600
         api_req = 'https://min-api.cryptocompare.com/data/histohour?fsym=BTC&tsym=USD&limit=2000&toTs='+str(stamp)
         if r.json()['TimeTo']==y:
             ok = True
             
    '''convert the list to dataframe and then delete the null values'''  
    data = pd.DataFrame(data)         
    data.replace(0, np.nan, inplace=True)
    data= data.dropna()
    data.reset_index(drop=True, inplace=True)
    
    '''change from timestamp format to year/month/day format'''
    df = []
    for column in data['time']:
            df.append({'time':datetime.fromtimestamp(column).date()})

    '''change the order of the columns in the list and also change the name of the columns'''
    data= data[['time','close','open','low','high','volumeto','volumefrom']]    
    data['volumeBTC'] = data.pop('volumeto')
    data['volumeUSD'] = data.pop('volumefrom')
     
    '''update the current data list with the new values for the time columns and create the csv file'''
    data.update(df) 
    data.to_csv('Code/Data/data.csv',index=False)

    
    
    
    
    
    
    