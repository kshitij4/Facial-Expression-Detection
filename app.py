from flask import Flask, render_template, request
import cv2
from keras.models import load_model
import keras
import numpy as np
from tensorflow.keras.models import model_from_json
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods = ['GET', 'POST'])
def after():
    img = request.files['file1']
    img.save('static/file.jpg')

    image = cv2.imread('static/file.jpg', 0)
    image = cv2.resize(image, (48, 48))
    image = np.reshape(image, (1, 48, 48, 1))



    json_file = open('model_json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("model_weights1.h5")

    ##model = keras.models.load_model("model_weights.h5" )
    prediction = model.predict(image)

    #label_map = np.argmax(prediction)
    #final_preciction = label_map[prediction]




    return render_template('after.html', data =prediction)


if __name__ == "__main__":
    app.run(debug=True)
