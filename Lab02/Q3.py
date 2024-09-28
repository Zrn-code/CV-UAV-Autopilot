import cv2
import numpy as np
import random

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

# Read the image in grayscale
image = cv2.imread('output.png', cv2.IMREAD_GRAYSCALE)

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
cv2.imshow('Two Pass Algorithm - Connected Components', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

