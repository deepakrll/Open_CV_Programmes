import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.feature import local_binary_pattern
from google.colab import drive
import os

def mount_drive():
    drive.mount('/content/drive', force_remount=True)

def load_trained_model(model_path):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    try:
        model = load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print("Error loading model:", e)
        return None

def preprocess_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (128, 128))  # Resize to match model input
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0  # Normalize
    return np.expand_dims(image, axis=0)

def extract_texture_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Unable to read image for texture analysis.")
        return None
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
    return hist.reshape(1, -1)

def detect_defects(image_path, model):
    if model is None:
        print("Error: No model loaded. Please check the model path.")
        return
    image = preprocess_image(image_path)
    texture_features = extract_texture_features(image_path)
    if image is None or texture_features is None:
        return
    
    predictions = model.predict(image)
    defect_classes = ["No Defect", "Small Bubble or Uneven Paint", "Scratches or Streaks", "Blotches or Paint Runs"]
    predicted_class = defect_classes[np.argmax(predictions)]
    
    # Edge detection for better defect visualization
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Display results
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f"Original Image - Predicted: {predicted_class}")
    ax[0].axis("off")
    
    ax[1].imshow(edges, cmap='gray')
    ax[1].set_title("Edge Detection (Highlights Defects)")
    ax[1].axis("off")
    
    plt.show()
    print("Detected Defect:", predicted_class)

# Mount Google Drive
mount_drive()

# Define paths
model_path = "/content/drive/My Drive/Colab Notebooks/defect_detection_model.h5"
image_path = "/content/drive/My Drive/Colab Notebooks/images/painting_defect_1.jpg"

# Load the trained CNN model
model = load_trained_model(model_path)

# Run detection only if model is loaded successfully
if model:
    detect_defects(image_path, model)
