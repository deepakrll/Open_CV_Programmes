from google.colab import drive
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Mount Google Drive
drive.mount('/content/drive')

# Define the image path in Google Drive
image_path = "/content/drive/My Drive/Colab Notebooks/images/mechanical_part.jpg"

def detect_defects(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original image
    defect_image = image.copy()
    cv2.drawContours(defect_image, contours, -1, (0, 255, 0), 2)
    
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

# Run the function
detect_defects(image_path)
