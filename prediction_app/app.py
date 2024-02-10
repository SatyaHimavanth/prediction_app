import os
from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)
model = load_model("age_race_gender_model.h5")

gender_code = {0: 'male', 1: 'female'}

race_code = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'}

def preprocess_image(image):
    image = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (198, 198))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_age_race_gender(image):
    image = preprocess_image(image)
    age_pred, race_pred, gender_pred = model.predict(image)
    age_pred = age_pred[0][0] * 100  # Assuming age was normalized between 0 and 1
    race_pred = np.argmax(race_pred)
    gender_pred = np.argmax(gender_pred)
    return age_pred, race_pred, gender_pred

@app.route("/", methods=["GET", "POST"])
def index():
    age = None
    if request.method == "POST":
        if "image" in request.files:
            image = request.files["image"]
            if image.filename == "":
                return render_template("index.html", error="No file selected!")
            if image:
                age, race, gender = predict_age_race_gender(image)
                return render_template("index.html", age=age, race=race_code[race], gender=gender_code[gender])
    return render_template("index.html", age=age)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
