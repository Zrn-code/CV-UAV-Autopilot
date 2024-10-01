import cv2
import numpy as np
import random

def equalize_channel(channel):
    # Compute the histogram of the channel
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    
    # Compute the CDF of the histogram
    cdf = hist.cumsum()
    
    # Use the CDF to normalize the channel values
    cdf_m = np.ma.masked_equal(cdf, 0)  # mask the zero values in the CDF
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())  # normalize the CDF
    cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')  # fill the masked values with zeros
    
    # Apply the equalization to the channel
    equalized_channel = cdf_final[channel]
    
    return equalized_channel

def Q1_1(image):
    # Convert the image from BGR to RGB for displaying using matplotlib
    #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Split the image into its BGR components
    b, g, r = cv2.split(image)

    # Apply histogram equalization to each channel
    equalized_b = equalize_channel(b)
    equalized_g = equalize_channel(g)
    equalized_r = equalize_channel(r)
    
    # Merge the equalized channels back
    equalized_image = cv2.merge([equalized_b, equalized_g, equalized_r])
    #display_compare_result(image_rgb, cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB), 'Original Image', 'Equalized Image (BGR Channels)')
    return equalized_image

def Q1_2(image):
    # Convert the image to HSV format
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split the HSV image into its H, S, and V components
    h, s, v = cv2.split(hsv_image)

    # Apply histogram equalization only to the V (Value) channel
    equalized_v = equalize_channel(v)

    # Merge the equalized V channel back with H and S
    equalized_hsv_image = cv2.merge([h, s, equalized_v])

    # Convert the equalized HSV image back to BGR format for display
    equalized_bgr_image = cv2.cvtColor(equalized_hsv_image, cv2.COLOR_HSV2BGR)

    #display_compare_result(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cv2.cvtColor(equalized_bgr_image, cv2.COLOR_BGR2RGB), 'Original Image', 'Equalized Image (V Channel)')
    return equalized_bgr_image


def Q2(image): # Otsu_threshold
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


# Helper function to find the root of the label (for union-find algorithm)
def find_root(label, parent):
    if label == 0:
        return 0  # Background label (black)
    while parent[label] != label:
        label = parent[label]
    return label

# Helper function to union two labels (for union-find algorithm)
def union_labels(label1, label2, parent):
    if label1 == 0 or label2 == 0:
        return  # Background label (black)
    root1 = find_root(label1, parent)
    root2 = find_root(label2, parent)
    if root1 != root2:
        parent[root2] = root1

# Two Pass Algorithm for connected component labeling
def two_pass_labeling(binary_image):
    labels = np.zeros(binary_image.shape, dtype=int)  # initialize labels
    label = 1  # current label
    parent = {}  # parent dictionary for union-find algorithm

    # One pass to assign temporary labels
    for i in range(1, binary_image.shape[0]):  # skip the first row
        for j in range(1, binary_image.shape[1]):
            if binary_image[i, j] == 255:  # white pixel
                neighbors = []
                if binary_image[i - 1, j] == 255:  # upper neighbor
                    neighbors.append(labels[i - 1, j])
                if binary_image[i, j - 1] == 255:  # left neighbor
                    neighbors.append(labels[i, j - 1])

                if neighbors:
                    min_label = min(neighbors)
                    labels[i, j] = min_label
                    for neighbor_label in neighbors:
                        if neighbor_label != min_label:
                            union_labels(min_label, neighbor_label, parent)
                else:
                    labels[i, j] = label
                    parent[label] = label
                    label += 1

    # Two pass to resolve the final labels
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i, j] > 0:
                labels[i, j] = find_root(labels[i, j], parent)

    return labels, label

def Q3(image):
    # Use the two pass algorithm to find connected components
    labels, num_labels = two_pass_labeling(image)

    # Create an output image
    output_image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)

    # Generate random colors for the labels
    colors = []
    for i in range(1, num_labels):
        colors.append([random.randint(0, 255) for _ in range(3)])

    # Color the connected components
    for i in range(1, num_labels):
        output_image[labels == i] = colors[i - 1]

    # Display the output image
    #cv2.imshow('Two Pass Algorithm - Connected Components', output_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return output_image




if __name__ == "__main__":
    print("Lab02 Demo: \nQ1: Histogram Equalization\nQ2: Otsu Thresholding\nQ3: Connected Component Labeling\n")
    while True:
        choice = input("Enter your choice (Q1/Q2/Q3): ")
        if choice == "Q1":
            image = cv2.imread("histogram.jpg")
            equalized_image = Q1_1(image)
            equalized_image_2 = Q1_2(image)
            cv2.imshow("Equalized Image (BGR Channels)", equalized_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            cv2.imshow("Equalized Image (V Channel)", equalized_image_2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            #display_compare_result(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB), cv2.cvtColor(equalized_image_2, cv2.COLOR_BGR2RGB), 'Original Image', 'Equalized Image (BGR Channels)', 'Equalized Image (V Channel)')
                
        elif choice == "Q2":
            image = cv2.imread("input.jpg", 0)
            image_2, optimal_threshold = Q2(image)
            print(f"Optimal Threshold: {optimal_threshold}")
            cv2.imshow("Otsu Threshold Image", image_2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        elif choice == "Q3":
            image = cv2.imread("output.png", cv2.IMREAD_GRAYSCALE)
            output_image = Q3(image)
            cv2.imshow("Two Pass Algorithm - Connected Components", output_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        else:
            print("Invalid choice. Please enter Q1, Q2, or Q3.")
            continue
    
        
    
