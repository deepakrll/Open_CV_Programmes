import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive

def mount_drive():
    drive.mount('/content/drive', force_remount=True)

def classify_defect(contours, image_shape):
    defect_types = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        bounding_box = cv2.boundingRect(contour)
        aspect_ratio = bounding_box[2] / bounding_box[3] if bounding_box[3] > 0 else 0
        
        if area < 500:
            defect_types.append("Porosity")
        elif aspect_ratio > 2:
            defect_types.append("Crack")
        else:
            defect_types.append("Incomplete Fusion")
    
    return defect_types

def detect_defects(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print("Error: Image not found. Check the file path.")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Classify defects
    defect_types = classify_defect(contours, image.shape)
    
    # Draw contours on the original image
    defect_image = image.copy()
    for i, contour in enumerate(contours):
        cv2.drawContours(defect_image, [contour], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(defect_image, defect_types[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Display results
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Detection")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(defect_image, cv2.COLOR_BGR2RGB))
    plt.title("Defect Highlighted")
    plt.axis("off")
    
    plt.show()
    
    print("Detected Defects:", defect_types)
    
# Mount Google Drive
mount_drive()

# Example usage (provide an image path from Google Drive)
image_path = "/content/drive/My Drive/Colab Notebooks/images/welding_joint.jpg"  # Replace with actual image path
detect_defects(image_path)
