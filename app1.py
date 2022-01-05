from flask import Flask, render_template, Response, url_for, make_response, jsonify ,request
import cv2
import os
import time
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

import tensorflow as tf
# load model
model = model_from_json(open("fer.json", "r").read())
# load weights
model.load_weights('fer.h5')


from camera import VideoCamera

app = Flask(__name__, static_folder='static')

facec = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor = 0.6
font = cv2.FONT_HERSHEY_SIMPLEX

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)
set_session(session)

camera = cv2.VideoCapture(0)  # use 0 for web camera


class FacialExpressionModel(object):
    EMOTIONS_LIST = {0:'with_mask',1:'without_mask'}

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.compile()
        #self.loaded_model._make_predict_function()

    def predict_emotion(self, img):
        global session
        set_session(session)
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]






model = FacialExpressionModel("mask_detection.json", "mask_detection.h5")


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y + h, x:x + w]

            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 0, 0), 2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

from werkzeug.debug import DebuggedApplication


time.time()

english_bot = ChatBot("Chatterbot", storage_adapter="chatterbot.storage.SQLStorageAdapter")
trainer = ChatterBotCorpusTrainer(english_bot)
trainer.train("chatterbot.corpus.english")


#define app routes

@app.route("/get")


def get_bot_response(englishBot=None):
    userText = request.args.get('msg')
    return str(english_bot.get_response(userText))




def create_app():
    # Insert whatever else you do in your Flask app factory.

    if app.debug:
        app.wsgi_app = DebuggedApplication(app.wsgi_app, evalex=True)

    return app
obj=VideoCamera()
imgs=obj.get_frame()

emotionsList = []
#emotion=model.predict_emotion(imgs)
#emotionsList.append(emotion)
print(emotionsList)

if __name__ == '__main__':
    app.run()
