import numpy as np
from matplotlib import pyplot as plt
import pandas as pd



depth_img_array = np.load('2022-05-09-08-15-33_86.75\depth_image_1652109334245600312.npy') #720 1280-length arrays
color_img_array = np.load('2022-05-09-08-15-33_86.75\color_image_1652109334259405316.npy')
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