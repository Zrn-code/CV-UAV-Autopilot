import numpy as np
import cv2


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
            image_new[i, j] = 1 / (dx * dy) * (
                (x1 - x) * (y1 - y) * image[y0, x0] +
                (x - x0) * (y1 - y) * image[y0, x1] +
                (x1 - x) * (y - y0) * image[y1, x0] +
                (x - x0) * (y - y0) * image[y1, x1]
            )
    return image_new



if __name__ == '__main__':
    
    image = cv2.imread('ive.jpg')
    cv2.imshow('Original Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    image_2 = bilinear_interpolation(image, 2)
    cv2.imshow('Bilinear Interpolation Image', image_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
