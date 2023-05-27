import os

import tensorflow as tf
import keras
from keras.models import load_model
from pathlib import Path
from flask import Flask, jsonify, request
from src.uploadFile import *
from src.downloadModel import *
from src.fluency import *

app = Flask(__name__)

pathFluency = "./models/model_ielts_fluency_real_data.h5"

my_file = Path(pathFluency)
if not my_file.is_file():
    # file exists check
    downloadModel("https://storage.googleapis.com/ielts-capstone/model_ielts_fluency_real_data.h5", dest_folder="../models/")

loaded_model_fluency = load_model(pathFluency)

@app.route("/")
def hello_world():
    """Example Hello World route."""
    name = os.environ.get("NAME", "World")
    return f"Hello IELTS New {name}!"

#  POST Method
@app.route("/fluency", methods=['POST'])
def fluency():
    file_to_upload = request.files['file']
    audioUrl = uploadFile(file_to_upload)

    # Using data from Cloudinary with (secure_url)
    mfccs, rmse, spectral_flux, zcr = feature_extraction(audioUrl) 
    number_of_features = 3 + 30
    datasetcheck = np.empty((0,number_of_features))
    extracted_features = np.hstack([mfccs, rmse, spectral_flux, zcr])
    datasetcheck = np.vstack([datasetcheck, extracted_features])
    datasetcheck = np.expand_dims(datasetcheck, axis=1)
    pred = loaded_model_fluency.predict(datasetcheck)
    classes = np.argmax(pred, axis = 1).tolist()

    return jsonify({"Fluency Class": classes[0]})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))