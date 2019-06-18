from flask import Flask, jsonify, request, render_template,flash, redirect
import os
import pandas as pd 
import time
import tensorflow as tf
from werkzeug import secure_filename
import matplotlib.pyplot as plt
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from keras import backend as K
from Code.Data import *
from Code.DataAquisition import *
from Code.DataProcessingSimple import *
from Code.GridSearch import *
from Code.main import main

UPLOAD_FOLDER = 'Upload'
ALLOWED_EXTENSIONS = set(['json'])

 
application = Flask(__name__)
application.secret_key = 'super secret key'
application.config['SESSION_TYPE'] = 'filesystem'
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
graph = tf.get_default_graph()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Loading home page
@application.route('/',defaults={'page': 'upload'})
def show(page):
        return render_template('upload.html') 
   

@application.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect('http://localhost:5000/')
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect('http://localhost:5000/')
    if file and allowed_file(file.filename):
        file.save(os.path.join(application.config['UPLOAD_FOLDER'], "config.json"))
        return redirect('http://localhost:5000/results')
    else:
        return redirect('http://localhost:5000/')
    return "not good"
    
# Handling 400 Error
@application.errorhandler(400)
def bad_request(error=None):

    message = {
        'status': 400,
        'message': 'Bad Request: ' + request.url + '--> Please check your data payload...',
    }
    
    resp = jsonify(message)
    resp.status_code = 400
    
    return resp

@application.route('/results')
def plot_png():
    main()
    return render_template("results.html")
    
    
# run application
if __name__ == "__main__":
    application.run(debug=True)