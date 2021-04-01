from flask import Flask, render_template, request
import cv2
from keras.models import load_model
import keras
import numpy as np
from tensorflow.keras.models import model_from_json
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

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
    model.load_weights("model_weights.h5")

    ##model = keras.models.load_model("model_weights.h5" )
    prediction = model.predict(image)

    pred = list(prediction[0])
    anger = int(pred[0]*100)
    disgust = int(pred[1]*100)
    fear = int(pred[2]*100)
    happy = int(pred[3]*100)
    neutral = int(pred[4]*100)
    sad = int(pred[5]*100)
    surprised = int(pred[6]*100)
    maxNum = max(pred)
    #for i in range(7):

    indx = pred.index(maxNum)
    emotion = ""
    if indx == 0:
        emotion = "The person is angry ヽ(ಠ_ಠ)ノ"
    elif indx == 1:
        emotion = "The person is feeling disgust 「(°ヘ°)"
    elif indx == 2:
        emotion = "The person is in fear (⋋▂⋌)"
    elif indx == 3:
        emotion = "The person is happy ヽ(•‿•)ノ"
    elif indx == 4:
        emotion = "The person is neutral ┐(￣ー￣)┌"
    elif indx == 5:
        emotion = "The person is sad (︶︿︶) "
    elif indx == 6:
        emotion = "The person is surprised (⊙＿⊙')" 

    color = "#653533"

    return render_template('after.html', data =emotion, angerlvl = anger, disgustlvl = disgust, fearlvl = fear, happylvl = happy, neutrallvl = neutral, sadlvl = sad, surprisedlvl = surprised)
    

    


if __name__ == "__main__":
    app.run(debug=True)
