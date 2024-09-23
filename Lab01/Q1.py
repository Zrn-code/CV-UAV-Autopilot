import cv2
import numpy as np

# Read the image
image = cv2.imread('test.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Copy the grayscale image into three channels to create a 3-channel grayscale image
gray_3channel_image = cv2.merge([gray, gray, gray])

# Define the condition to identify blue pixels:

blue_mask = (image[:, :, 0] > 100) & (image[:, :, 0] * 0.6 > image[:, :, 1]) & (image[:, :, 0] * 0.6 > image[:, :, 2])

# Use the blue_mask to retain blue pixels from the original image, and replace all other pixels with the grayscale version
result = np.where(blue_mask[..., None], image, gray_3channel_image)

# Show the image with blue pixels preserved, and the rest in grayscale
cv2.imshow('Blue Filter', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the resulting image
cv2.imwrite('blue_filtered_image.jpg', result)
