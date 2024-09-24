import numpy as np
import cv2

def blue_filter(image):
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

    return result
    
    
def adjust_contrast_brightness(image, contrast, brightness):
    # Convert the image to int32 to avoid overflow issues
    new_image = np.int32(image)
    
    # Apply the contrast and brightness adjustments
    new_image = (new_image - 127) * (contrast / 127 + 1) + 127 + brightness
    
    # Clip the values to be in the valid range [0, 255]
    new_image = np.clip(new_image, 0, 255)
    
    # Convert back to uint8
    return np.uint8(new_image)

def yellow_blue_contrast(image, contrast, brightness):
    mask =  ((image[:, :, 0] + image[:, :, 1]) *0.3 > image[:, :, 2])

    result = image.copy()
    # Apply contrast and brightness adjustment only to the masked pixels
    result[mask] = adjust_contrast_brightness(result[mask], contrast, brightness)

    return result
    
def bilinear_interpolation(image, rate):
    # get image size
    h, w, c = image.shape
    # new image size, cast to int to avoid issues
    h_new = int(h * rate)
    w_new = int(w * rate)
    # create new image
    image_new = np.zeros((h_new, w_new, c), np.uint8)
    # calculate scale
    scale_x = w / w_new
    scale_y = h / h_new
    # calculate new image pixel value
    for i in range(h_new):
        for j in range(w_new):
            x = (j + 0.5) * scale_x - 0.5  # make sure the center of pixel is in the center of image
            y = (i + 0.5) * scale_y - 0.5
            x0 = int(np.floor(x))  # get the nearest pixel
            y0 = int(np.floor(y))
            x1 = min(x0 + 1, w - 1)  # avoid out of range
            y1 = min(y0 + 1, h - 1)
            
            # Avoid zero division
            dx = x1 - x0 if x1 != x0 else 1
            dy = y1 - y0 if y1 != y0 else 1

            # bilinear interpolation
            # refer to https://en.wikipedia.org/wiki/Bilinear_interpolation#Repeated_linear_interpolation
            image_new[i, j] = 1 / (dx * dy) * (
                (x1 - x) * (y1 - y) * image[y0, x0] +
                (x - x0) * (y1 - y) * image[y0, x1] +
                (x1 - x) * (y - y0) * image[y1, x0] +
                (x - x0) * (y - y0) * image[y1, x1]
            )
    return image_new

    
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
    print('Lab01 Demo:\nQ1: 1.Blue Filter & 2.Contrast and Brightness \nQ2: Bilinear Interpolation\nQ3: Sobel Filter Detection\n')
    while True:
        choice = input('Enter your choice (Q1/Q2/Q3): ')
        if choice == 'Q1':
            x = int(input('Enter the subproblem: '))
            if x == 1:
                image = cv2.imread('test.jpg')
                image_1 = blue_filter(image)
                cv2.imshow('Blue Filter', image_1)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            elif x == 2:
                image = cv2.imread('test.jpg')
                contrast = int(input('Enter the contrast: '))
                brightness = int(input('Enter the brightness: '))
                image_1 = yellow_blue_contrast(image, contrast, brightness)
                cv2.imshow('Yellow and Blue Contrast Adjustment', image_1)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print('Invalid subproblem!')
        elif choice == 'Q2':
            image = cv2.imread('ive.jpg')
            rate = float(input('Enter the rate of interpolation: '))
            image_2 = bilinear_interpolation(image, rate)
            cv2.imshow('Bilinear Interpolation Image', image_2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif choice == 'Q3':
            image = cv2.imread('ive.jpg')
            image_3 = filtering_sobel(image)
            cv2.imshow('Sobel Filter Detection', image_3)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print('Invalid choice!')