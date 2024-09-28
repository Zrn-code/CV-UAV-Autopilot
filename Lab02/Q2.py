import numpy as np
import cv2


def Otsu_threshold(image):
    # Compute the histogram of the image (256 bins for grayscale values)
    histogram = np.bincount(image.ravel(), minlength=256)
    
    # Initialize variables for background
    nB = 0
    sumB = 0
    muB = 0
    
    # Initialize variables for object (initially the whole image)
    nO = image.size
    sumO = np.dot(np.arange(256), histogram)
    muO = sumO / nO
    
    max_var = 0
    threshold = 0

    for t in range(256):
        nB += histogram[t]
        if nB == 0:
            continue  # Skip if background has no pixels
        
        nO -= histogram[t]
        if nO == 0:
            break  # Stop if no object pixels are left
        
        sumB += histogram[t] * t
        muB = sumB / nB
        
        sumO -= histogram[t] * t
        muO = sumO / nO

        # Calculate between-class variance for the current threshold
        var_between = nB * nO * (muB - muO) ** 2

        if var_between > max_var:
            max_var = var_between
            threshold = t

    _, output = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    
    return output, threshold


if __name__ == '__main__':

    image = cv2.imread('input.jpg', 0)
    cv2.imshow('Original Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    image_2, optimal_threshold = Otsu_threshold(image)
    print(f'Optimal Threshold: {optimal_threshold}')
    
    cv2.imshow('Otsu Threshold Image', image_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('output.png', image_2)