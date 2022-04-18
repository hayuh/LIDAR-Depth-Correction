from typing import Tuple
import numpy as np
import torch
import torchvision
from skspatial.objects import Points, Plane
import matplotlib.pyplot as plt
from matplotlib.path import Path

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

# assume upper left, upper right, lower right, lower left for now
def sort_four_points(four_points):
    l1, l2, l3, l4 = four_points[0], four_points[1], four_points[2], four_points[3]
    return [(l1[0], l1[1]), (l2[0], l2[1]), (l3[0], l3[1]), (l4[0], l4[1])]

def get_fixed_depth(x, y, normal, d):
    a, b, c = normal
    return (d - a * x - b * y) / c

def planar_fit(depth_image, four_points):
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

def test_planar_fit():
    image = np.random.rand(10, 10)
    four_points = [(4, 4, 1), (6, 4, 1), (6, 6, 1), (4, 6, 1)]
    fixed = planar_fit(image, four_points)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in range(fixed.shape[0]):
        for j in range(fixed.shape[1]):
            ax.scatter(i, j, fixed[i][j])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


if __name__ == '__main__':
    test_planar_fit()