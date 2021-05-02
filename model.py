import numpy as np
from flask import Flask, render_template
import flask
from tensorflow.keras.preprocessing.image import img_to_array
import io
from tensorflow import keras
from PIL import Image
import base64

app = Flask(__name__)
model = None

CLASSES = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", \
           "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup", \
           "Vegetable/Fruit"]


def load_model():
    global model
    model = keras.models.load_model("food11.model")


def prepare_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image[:, :, 0] = image[:, :, 0] - 123.68
    image[:, :, 1] = image[:, :, 1] - 116.779
    image[:, :, 2] = image[:, :, 2] - 103.939

    return image


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=["POST"])
def predict():
    output = "Invalid"

    if flask.request.files.get("image"):
        # read the image in PIL format
        image = flask.request.files["image"].read()
        data=io.BytesIO(image)
        image = Image.open(data)

        # preprocess the image and prepare it for classification
        image = prepare_image(image)

        # classify the input image and then initialize the list
        # of predictions to return to the client
        preds = model.predict(image)
        predIdxs = np.argmax(preds, axis=1)

        output = "The class is " + CLASSES[predIdxs[0]]
        encoded_img_data=base64.b64encode(data.getvalue())

        return render_template('result.html', output=output,img_data=encoded_img_data.decode('utf-8'))

    return render_template('result.html', output=output)


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run()
