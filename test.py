# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 23:24:21 2019

@author: Andrei
"""
import gunicorn
import requests

r = requests.post('http://localhost:5000/titanic-survival-classification-model/predict')
print("Andrei")
print(r.content)
print (gunicorn.__version__)