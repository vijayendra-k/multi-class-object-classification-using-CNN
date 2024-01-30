import base64
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('object_recognition_model.h5')
class_labels = ['Bicycle', 'Cat', 'Dog', 'Female', 'Male']

def predict_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]

    confidence = predictions[0][predicted_class_index]

    # Convert the image to a base64-encoded string
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    return predicted_class, round(100*confidence,2), encoded_string

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        image_path = 'uploaded_image.jpg'
        file.save(image_path)
        predicted_class, confidence, encoded_string = predict_image(image_path)

        return render_template('result.html', result_class=predicted_class, confidence=confidence, encoded_image=encoded_string)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
