import numpy as np
import torch
import torchvision
from skspatial.objects import Points
from skspatial.objects import Plane


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


def get_all_window_points(depth_image, four_points):
    res = []
    lu, ru, rd, ld = sort_four_points(four_points)




def planar_fit(depth_image, four_points):
    points = Points(four_points)
    plane = Plane.best_fit(points)
