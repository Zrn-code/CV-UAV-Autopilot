import cv2
import numpy as np
from matplotlib import pyplot as plt

#Q1-1

# Load the image
image = cv2.imread("histogram.jpg")

# Convert the image from BGR to RGB for displaying using matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Split the image into its BGR components
b, g, r = cv2.split(image)

# Apply histogram equalization to each channel
equalized_b = cv2.equalizeHist(b)
equalized_g = cv2.equalizeHist(g)
equalized_r = cv2.equalizeHist(r)

# Merge the equalized channels back
equalized_image = cv2.merge([equalized_b, equalized_g, equalized_r])

# Plot the original and equalized images
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

# Equalized Image
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB))
plt.title('Equalized Image (BGR Channels)')
plt.axis('off')

# Save the equalized image
output_path = 'histogram_equalized_image.jpg'
cv2.imwrite(output_path, equalized_image)

plt.show()

# Q1-2

# Load the image
image = cv2.imread("histogram.jpg") 

# Convert the image to HSV format
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Split the HSV image into its H, S, and V components
h, s, v = cv2.split(hsv_image)

# Apply histogram equalization only to the V (Value) channel
equalized_v = cv2.equalizeHist(v)

# Merge the equalized V channel back with H and S
equalized_hsv_image = cv2.merge([h, s, equalized_v])

# Convert the equalized HSV image back to BGR format for display
equalized_bgr_image = cv2.cvtColor(equalized_hsv_image, cv2.COLOR_HSV2BGR)

# Plot the original and equalized images
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Equalized Image (V Channel Histogram Equalization)
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(equalized_bgr_image, cv2.COLOR_BGR2RGB))
plt.title('Equalized Image (V Channel)')
plt.axis('off')

# Save the equalized image
output_path = "hsv_v_equalized_image.jpg"  # Define where to save the output image
cv2.imwrite(output_path, equalized_bgr_image)

plt.show()
