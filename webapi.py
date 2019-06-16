from flask import Flask, jsonify, request, render_template, abort
from Code import main
from api import predict_api


application = Flask(__name__)
application.register_blueprint(predict_api, url_prefix='/titanic-survival-classification-model')

# Loading home page
@application.route('/',defaults={'page': 'upload'})
@application.route('/<page>')
def show(page):
        return render_template('upload.html') 
   
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

# run application
if __name__ == "__main__":
    application.run()