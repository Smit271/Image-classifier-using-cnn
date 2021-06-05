#!/usr/bin/env python3
import os
import flask
import uuid
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
from time import gmtime, strftime
import tensorflow as tf
import numpy as np


app = flask.Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SECRET_KEY'] = 'sjbcxzsdc15xz6czc'


model = tf.keras.models.load_model('./model_best.h5')
classes = ['building', 'forest', 'glacier', 'mountain', 'sea', 'street']

def get_output(image_path):
    test_image = tf.keras.preprocessing.image.load_img(image_path, color_mode='rgb', target_size=(150, 150),interpolation='bicubic')
    
    test_image = tf.keras.preprocessing.image.img_to_array(test_image, data_format="channels_last") / 255.
    test_image = np.expand_dims(test_image, axis=0)
    scores = model.predict(test_image)
    preds = np.argmax(scores, axis = 1)
    
    return classes[preds[0]]

@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def home():
	if request.method == 'POST':
		upload_image()
	return render_template('index.html')
	
	
def upload_image():
    if request.files:
        
        output = "NO"
        in_image = request.files["in_image"]
        filename = str(uuid.uuid4()) + '_' + str(strftime("%Y_%m_%d-%H_%M_%S", gmtime()))
        print(filename)
        print(in_image.filename.split('.')[-1])
        filename = filename +"."+in_image.filename.split('.')[-1]
        in_image.save(os.path.join('uploads/', secure_filename(filename)))
        if os.path.exists(os.path.join('uploads/',filename)):
            output = get_output(os.path.join('uploads/', filename))
            print("This is output")
            print(output)
            if output=="NO":
                flash(f'Please upload image first!', 'danger')
                return render_template("index.html")
            # print(output)
        if output!="NO":
            flash(f'This is sign of {output.capitalize()}!', 'success')
            return render_template("index.html")
    else:
        return render_template("index.html")
    return render_template('index.html')

@app.route('/about')
def about():      
	return render_template('about.html')


@app.route('/other_work')
def other():      
	return render_template('other.html')


if __name__ == "__main__":
	app.run(debug=True)
