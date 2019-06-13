# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:13:51 2019

@author: Andrei
"""

from flask import Flask 
from flask_restful import Api, Resource, reqparse 

app = Flask(__name__)
api = Api(app)