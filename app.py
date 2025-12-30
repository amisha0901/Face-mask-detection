from flask import Flask, jsonify, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from config import config

model = tf.keras.models.load_model("lib/face_mask_detection.keras")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index2.html")

# @app.route("/classify", methods = ["POST"])
# def get_prediction():
#     file = request.files["image"]
#     input_image = image.load_img(file, target_size = config.image_size)
#     image_array = image.img_to_array(input_image)/255  
#     test_array = image_array.reshape(1, 64, 64, 3)
#     prediction = model.predict(test_array)
#     output = sorted(list(zip(list(config.class_indices), list(prediction[0]))), key = lambda x:x[1])[-1]
    
#     return output



# @app.route("/classify", methods=["POST"])
# def get_prediction():
#     file = request.files["image"]

#     # Convert FileStorage → BytesIO
#     img_bytes = BytesIO(file.read())

#     # Load image
#     input_image = image.load_img(
#         img_bytes,
#         target_size=config.image_size
#     )

#     # Preprocess
#     image_array = image.img_to_array(input_image) / 255.0
#     print(image_array)
#     # test_array = image_array.reshape(1, 64, 64, 3)
#     # print(test_array)
    

#     # Predict
#     prediction = model.predict(image_array)

#     # Get class with highest probability
#     class_labels = list(config.class_indices)
#     predicted_index = np.argmax(prediction[0])
#     predicted_label = class_labels[predicted_index]
#     confidence = float(prediction[0][predicted_index])

#     return jsonify({
#         "prediction": predicted_label,
#         "confidence": round(confidence, 4)
#     })

@app.route("/classify", methods = ["POST"])
def get_prediction():
    file = request.files["image"]

    img_bytes = BytesIO(file.read())

    input_image = image.load_img(
        img_bytes,
        target_size=config.image_size
    )

    image_array = image.img_to_array(input_image) / 255.0
    test_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(test_array, verbose=0)

    predicted_index = int(np.argmax(prediction[0]))
    predicted_label = config.class_indices[predicted_index]
    confidence = float(prediction[0][predicted_index]) * 100

    return jsonify({
        "success": True,
        "predictions": [
            {
                "description": predicted_label.replace("_", " ").title(),
                "confidence": round(confidence, 2)
            }
        ]
    })



if __name__ == "__main__":
    app.run(port = config.port, debug = True)