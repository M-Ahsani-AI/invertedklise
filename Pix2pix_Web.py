from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
import base64
from PIL import Image
import io
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

model = None
MODEL_PATH = r"C:\\Users\\moham\\OneDrive\\Documents\\CODE\\ImageProcessing\\pix2pix_checkpoints"

def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully!")
        else:
            print(f"Model file not found: {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")

def preprocess_image(image_data, is_base64=False):
    try:
        if is_base64:
            image_data = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            image = np.array(image)
        else:
            image = cv2.imread(image_data)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (256, 256))
        image = (image / 127.5) - 1.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def postprocess_image(prediction):
    try:
        prediction = (prediction[0] + 1.0) * 127.5
        prediction = np.clip(prediction, 0, 255).astype(np.uint8)
        return prediction
    except Exception as e:
        print(f"Error postprocessing image: {e}")
        return None

def image_to_base64(image_array):
    try:
        img = Image.fromarray(image_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

def process_image_from_file(file):
    # Simpan file sementara
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(file_path)

    input_image = preprocess_image(file_path)
    if input_image is None:
        os.remove(file_path)
        return None, None

    prediction = model(input_image, training=False)
    result_image = postprocess_image(prediction)

    # Load original image for base64
    original_img = cv2.imread(file_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img = cv2.resize(original_img, (256, 256))

    os.remove(file_path)

    original_b64 = image_to_base64(original_img)
    result_b64 = image_to_base64(result_image)

    return original_b64, result_b64

def process_image_from_base64(image_data):
    input_image = preprocess_image(image_data, is_base64=True)
    if input_image is None:
        return None, None

    prediction = model(input_image, training=False)
    result_image = postprocess_image(prediction)

    original_b64 = image_data
    result_b64 = image_to_base64(result_image)

    return original_b64, result_b64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    original_b64, result_b64 = process_image_from_file(file)
    if original_b64 is None or result_b64 is None:
        return jsonify({'error': 'Error processing image'}), 400

    return jsonify({
        'success': True,
        'original_image': original_b64,
        'converted_image': result_b64
    })

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    original_b64, result_b64 = process_image_from_file(file)
    if original_b64 is None or result_b64 is None:
        return jsonify({'error': 'Error processing image'}), 400

    return jsonify({
        'success': True,
        'original_image': original_b64,
        'converted_image': result_b64
    })

@app.route('/camera', methods=['POST'])
def camera_capture():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    original_b64, result_b64 = process_image_from_base64(data['image'])
    if original_b64 is None or result_b64 is None:
        return jsonify({'error': 'Error processing image'}), 400

    return jsonify({
        'success': True,
        'original_image': original_b64,
        'converted_image': result_b64
    })

# Load model on start
load_model()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)