import numpy as np
from matplotlib import pyplot as plt
import pandas as pd



depth_img_array = np.load('depth_one_window.npy') #720 1280-length arrays
color_img_array = np.load('color_one_window.npy')
# convert array into dataframe
DF_depth = pd.DataFrame(depth_img_array)
#DF_color = pd.DataFrame(color_img_array)
  
# save the dataframe as a csv file
DF_depth.to_csv("depth_img.csv")
#DF_color.to_csv("color_img.csv")

plt.imshow(depth_img_array, cmap='gray')
plt.show()

plt.imshow(color_img_array, cmap='gray')
plt.show()