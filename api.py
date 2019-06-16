
from flask import Blueprint, jsonify, request
from Code.Data import *
from Code.DataAquisition import *
from Code.DataProcessingSimple import *
from Code.GridSearch import *
from Code.main import main

predict_api = Blueprint('predict_api', __name__)

@predict_api.route('/predict', methods=['POST'])
def apicall():
    return main()
    