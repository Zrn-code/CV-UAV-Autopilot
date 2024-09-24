import cv2
import numpy as np

def adjust_contrast_brightness(image, contrast, brightness):
    # Convert the image to int32 to avoid overflow issues
    new_image = np.int32(image)
    
    # Apply the contrast and brightness adjustments
    new_image = (new_image - 127) * (contrast / 127 + 1) + 127 + brightness
    
    # Clip the values to be in the valid range [0, 255]
    new_image = np.clip(new_image, 0, 255)
    
    # Convert back to uint8
    return np.uint8(new_image)

# Load your image 
image = cv2.imread('test.jpg')

# Extract the blue, green, and red channels
B, G, R = image[:, :, 0], image[:, :, 1], image[:, :, 2]

# Create a mask based on the condition: (B + G) * 0.3 > R
mask = (B + G) * 0.3 > R

# Initialize the output image as a copy of the original
output_image = image.copy()

# Apply contrast and brightness adjustment only to the masked pixels
output_image[mask] = adjust_contrast_brightness(image[mask], contrast=100, brightness=40)

# Display the original and adjusted images
cv2.imshow('Original Image', image)
cv2.imshow('Adjusted Image', output_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# save the adjusted image
cv2.imwrite('adjusted_image.jpg', output_image)
