from flask import Flask, request, render_template, url_for, redirect
import os
from werkzeug.utils import secure_filename # Import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Define paths
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'BreastInsight.h5' # Assuming it's in the root, adjust if not
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the Keras model
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Set model to None if loading fails
    # Handle model loading error appropriately, e.g., exit or use a dummy model

# Define class names (as in the notebook, ensure this order matches model's output)
class_names = ['benign', 'malignant', 'normal']

def preprocess_image(image_path, target_size=(224, 224)):
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(target_size)
        img_array = np.asarray(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not loaded. Cannot predict.", 500

    if 'image' not in request.files:
        return redirect(url_for('index')) # Or show an error message

    file = request.files['image']

    if file.filename == '':
        return redirect(url_for('index')) # Or show an error message

    # Proceed only if file is valid
    filename = secure_filename(file.filename) # Do this once
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filepath)
        
        # Check if file was actually saved, though file.save() should raise error if it fails.
        # This is an extra check.
        if not os.path.exists(filepath):
            # This case should ideally not be reached if file.save() works as expected
            print(f"File not found after save: {filepath}")
            return "Error saving file.", 500

        preprocessed_img = preprocess_image(filepath)
        
        if preprocessed_img is None:
            if os.path.exists(filepath):
                os.remove(filepath)
            return "Error preprocessing image", 500

        predictions = model.predict(preprocessed_img)
            predicted_class_index = np.argmax(predictions, axis=1)[0] # Get the index
            
            if predicted_class_index < len(class_names):
                predicted_class_name = class_names[predicted_class_index]
            else:
                predicted_class_name = "Unknown" # Handle out-of-bounds index

            # Optionally, remove the uploaded file after prediction
            # For now, we'll keep it to display it in results, but in a real app, you might remove it.
            # os.remove(filepath) 
            
            # Pass the filename to result.html to display the image
            return render_template('result.html', prediction=predicted_class_name, image_filename=filename)
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Optionally, remove the uploaded file in case of error too
            if os.path.exists(filepath):
                os.remove(filepath)
            return "Error during prediction", 500
            
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
