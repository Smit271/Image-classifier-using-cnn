#!/usr/bin/env python3
import os
import flask
import uuid
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
from time import gmtime, strftime
import tensorflow as tf
import numpy as np


app = flask.Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = tf.keras.models.load_model('./model_best.h5')
classes = ['building', 'forest', 'glacier', 'mountain', 'sea', 'street']

def get_output(image_path):
    test_image = tf.keras.preprocessing.image.load_img(image_path, color_mode='rgb', target_size=(150, 150),interpolation='bicubic')
    
    test_image = tf.keras.preprocessing.image.img_to_array(test_image, data_format="channels_last") / 255.
    test_image = np.expand_dims(test_image, axis=0)
    scores = model.predict(test_image)
    preds = np.argmax(scores, axis = 1)
    
    return classes[preds[0]]

@app.route('/', methods=['GET'])
@cross_origin()
def any():      
	
	return render_template('index.html')
	
	
@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        in_image = request.files["in_image"]
        
        filename = str(uuid.uuid4()) + '_' + str(strftime("%Y_%m_%d-%H_%M_%S", gmtime()))
        in_image.save(os.path.join('uploads', secure_filename(f"{filename}.{in_image.filename.split('.')[-1]}")))
	    
        output = get_output(os.path.join('uploads', f"{filename}.{in_image.filename.split('.')[-1]}"))
	    
        os.remove(os.path.join('uploads', f"{filename}.{in_image.filename.split('.')[-1]}"))
        output.capitalize()
	    
    return flask.render_template("predict.html", value=output)
    
if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5555, debug=True)
