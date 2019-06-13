from flask import Flask, render_template, request, send_file
from flask_restful import Api, Resource, reqparse 

from werkzeug import secure_filename
app = Flask(__name__)
api = Api(app) 

@app.route('/home')
def upload_file1():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      print(f.filename)
      return f.filename
		
if __name__ == '__main__':
   app.run(host= '0.0.0.0', debug = True)