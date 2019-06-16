# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 23:24:21 2019

@author: Andrei
"""

import requests

r = requests.post('https://forecasting-btc.herokuapp.com/titanic-survival-classification-model/predict')
print("Andrei")
print(r.content)
