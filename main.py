from typing import Tuple
from cv2 import threshold
import numpy as np
import torch
import torchvision
from skspatial.objects import Points, Plane
import matplotlib.pyplot as plt
from matplotlib.path import Path
from skimage.measure import label, regionprops
import cv2

# outputs grayscale on standardized values
def depth_to_grayscale(depth_image):
    max_depth = depth_image.max
    min_depth = depth_image.min
    multiplier = (max_depth - min_depth) / 256
    mean = depth_image.mean
    return (depth_image - min_depth) * multiplier - mean


# depth_image: 2D numpy array of LiDAR depth data
def feature_extraction(depth_image):
    grayscale = depth_to_grayscale(depth_image)
    width, height = depth_image.shape
    

def testing():
    vgg16 = torchvision.models.vgg16(pretrained=True)
    for child in vgg16.children():
        print(child)


###########################################################################

def standardize_input(depth_image):
    return (depth_image - depth_image.mean()) / (depth_image.std())


def depth_to_variance_grid(depth_image, kernel_inc):
    grid = np.zeros(depth_image.shape)
    for i in range(depth_image.shape[0]):
        for j in range(depth_image.shape[1]):
            if not (i < kernel_inc or i > depth_image.shape[0] - kernel_inc - 1 or j < kernel_inc or j > depth_image.shape[1] - kernel_inc - 1):
                kernel = np.zeros((kernel_inc * 2 + 1, kernel_inc * 2 + 1))
                for i_ in range(-kernel_inc, kernel_inc + 1):
                    for j_ in range(-kernel_inc, kernel_inc + 1):
                        kernel[i_ + kernel_inc][j_ + kernel_inc] = depth_image[i+i_][j+j_]
                grid[i][j] = np.var(kernel)
    return grid

def variance_grid_to_four_points(variance_grid):
    return





##########################################################################

def test_planar_fit():
    image = np.random.rand(100, 100)
    four_points = [(40, 40, 0.77), (60, 40, 0.1), (60, 60, 0.3), (40, 60, 0.5)]
    fixed = planar_fit(image, four_points)

    plt.imshow(fixed, cmap='gray')
    plt.show()

def examine_data(path):
    image = np.load(path)
    plt.imshow(image)
    plt.show()

def test_variance_grid(path):
    image = np.load(path)
    standardized = standardize_input(image)
    grid = depth_to_variance_grid(standardized, 10)
    plt.imshow(grid, cmap='gray')
    plt.show()

def get_readable_estimation(depth_image, x, y, dirx, diry):
    value = depth_image[x, y]
    while(value == 0):
        x += dirx
        y += diry
    return value


def fix_depth_with_known_corners(image_path, corner_indices):
    depth_image = np.load(image_path)
    x1, y1, x2, y2, x3, y3, x4, y4 = corner_indices
    value1 = get_readable_estimation(depth_image, x1, y1, 1, 1)
    value2 = get_readable_estimation(depth_image, x2, y2, -1, 1)
    value3 = get_readable_estimation(depth_image, x3, y3, -1, -1)
    value4 = get_readable_estimation(depth_image, x4, y4, 1, -1)
    four_points = [(x1, y1, value1), (x2, y2, value2), (x3, y3, value3), (x4, y4, value4)]
    fixed = planar_fit(depth_image, four_points)
    np.save('./depth_two_window_fixed.npy', fixed)
    plt.imshow(fixed)
    plt.show()


###########################################################################

# assume upper left, upper right, lower right, lower left for now
def sort_four_points(four_points):
    l1, l2, l3, l4 = four_points[0], four_points[1], four_points[2], four_points[3]
    return [(l1[0], l1[1]), (l2[0], l2[1]), (l3[0], l3[1]), (l4[0], l4[1])]

def get_fixed_depth(x, y, normal, d):
    a, b, c = normal
    return (d - a * x - b * y) / c

def find_nearest_alt(depth_image, x, y, r):
    threshold = 900
    if (depth_image[x, y] > threshold):
        return x, y
    if (r > 0):
        for i in range(x - r, x + r + 1):
            if (i < 0 or i >= depth_image.shape[0]):
                continue
            for j in range(y - r, y + r + 1):
                if (j < 0 or j >= depth_image.shape[1]):
                    continue
                value = depth_image[i, j]
                if (value > threshold):
                    return i, j
        return find_nearest_alt(depth_image, x, y, r + 1)
    return x, y


def planar_fit(depth_image, corner_indices):
    # x1, y1, x2, y2, x3, y3, x4, y4 = corner_indices
    '''
    x1, y1 = find_nearest_alt(depth_image, x1, y1, 0)
    x2, y2 = find_nearest_alt(depth_image, x2, y2, 0)
    x3, y3 = find_nearest_alt(depth_image, x3, y3, 0)
    x4, y4 = find_nearest_alt(depth_image, x4, y4, 0)
    '''
    # four_points = [(x1, y1, get_readable_estimation(depth_image, x1, y1, 1, 1)), (x2, y2, get_readable_estimation(depth_image, x2, y2, -1, 1)), (x3, y3, get_readable_estimation(depth_image, x3, y3, -1, -1)), (x4, y4, get_readable_estimation(depth_image, x4, y4, 1, -1))]
    # four_points = [(x1, y1, depth_image[x1, y1]), (x2, y2, depth_image[x2, y2]), (x3, y3, depth_image[x3, y3]), (x4, y4, depth_image[x4, y4])]
    four_points = corner_indices
    four_points_indice = sort_four_points(four_points)
    points = np.empty(((depth_image.shape[0]) * (depth_image.shape[1]), 2))
    for i in range(depth_image.shape[0]):
        for j in range(depth_image.shape[1]):
            points[i * depth_image.shape[1] + j] = [i, j]
    
    p = Path(four_points_indice)
    grid = p.contains_points(points)
    mask = grid.reshape(depth_image.shape)
    
    points_4 = Points(four_points)
    plane = Plane.best_fit(points_4)
    d = np.sum(plane.point * plane.normal)

    for i in range(depth_image.shape[0]):
        for j in range(depth_image.shape[1]):
            if mask[i][j]:
                point_depth_fixed = get_fixed_depth(i, j, plane.normal, d)
                depth_image[i][j] = point_depth_fixed
    
    return depth_image

############################################################################


def get_all_zeros(depth_image):
    return np.not_equal(depth_image, 0)

def get_bounding_box(depth_image, diff_threshold):
    depth_image = (np.array(depth_image))
    mask = np.array(get_all_zeros(depth_image))
    labels = label(mask)
    props = regionprops(labels)
    for prop in props:
        min_row, min_col, max_row, max_col = prop.bbox
        if (min_row <= 0 or min_col <= 0 or max_row >= depth_image.shape[0] - 1 or max_col >= depth_image.shape[1] - 1):
            continue
        if (max_row - min_row <= diff_threshold or max_col - min_col <= diff_threshold):
            continue
        depth_image = planar_fit(depth_image, [min_row, min_col, max_row, min_col, max_row, max_col, min_row, max_col])
    plt.imshow(depth_image)
    plt.show()




############################################################################

if __name__ == '__main__':
    image = np.load('./depth_two_window.npy')
    #test_planar_fit()
    #examine_data('./depth_two_window.npy')
    #test_variance_grid('./test1.npy')
    fix_depth_with_known_corners('./depth_two_window.npy', [72, 381, 593, 338, 601, 554, 77, 590])
    #get_bounding_box(image, 10)