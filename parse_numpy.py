import numpy as np
from matplotlib import pyplot as plt



depth_img_array = np.load('depth_image_1649863390457055339.npy') #720 1280-length arrays
color_img_array = np.load('color_image_1649863390572339047.npy')

plt.imshow(depth_img_array, cmap='gray')
plt.show()

plt.imshow(color_img_array, cmap='gray')
plt.show()