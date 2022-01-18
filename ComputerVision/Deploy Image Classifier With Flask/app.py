# some utilities
import os
import numpy as np
from util import base64_to_pil

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect

#tensorflow
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2


# Variables 
# Change them if you are using custom model or pretrained model with saved weigths
Model_json = ".json"
Model_weigths = ".h5"


# Declare a flask app
app = Flask(__name__)

def get_ImageClassifierModel():
    model = MobileNetV2(weights='imagenet')

    # Loading the pretrained model
    # model_json = open(Model_json, 'r')
    # loaded_model_json = model_json.read()
    # model_json.close()
    # model = model_from_json(loaded_model_json)
    # model.load_weights(Model_weigths)

    return model  
    


def model_predict(img, model):
    '''
    Prediction Function for model.
    Arguments: 
        img: is address to image
        model : image classification model
    '''
    img = img.resize((224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    '''
    Render the main page
    '''
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    '''
    predict function to predict the image
    Api hits this function when someone clicks submit.
    '''
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)
        
        # initialize model
        model = get_ImageClassifierModel()

        # Make prediction
        preds = model_predict(img, model)

        pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode

        result = str(pred_class[0][0][1])               # Convert to string
        result = result.replace('_', ' ').capitalize()
        
        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=pred_proba)
    return None


if __name__ == '__main__':
    # app.run(port=5002)
    app.run()
