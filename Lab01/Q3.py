import numpy as np
import cv2


def filtering_sobel(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Define Sobel kernels for x and y directions
    grad_x = np.array([[-1, 0, 1], 
                       [-2, 0, 2], 
                       [-1, 0, 1]])
    
    grad_y = np.array([[-1, -2, -1], 
                       [ 0,  0,  0], 
                       [ 1,  2,  1]])

    sobel_x = cv2.filter2D(blurred_image, -1, grad_x)
    sobel_y = cv2.filter2D(blurred_image, -1, grad_y)
    sobel_xy = cv2.addWeighted(sobel_x, 1, sobel_y, 1, 0)
    
    return sobel_xy


if __name__ == '__main__':

    image = cv2.imread('ive.jpg')
    cv2.imshow('Original Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    image_3 = filtering_sobel(image)
    cv2.imshow('Sobel Filter Detection', image_3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()