import numpy as np
from matplotlib import pyplot as plt
import pandas as pd



depth_img_array = np.load('depth_image_1650298896740195817.npy') #720 1280-length arrays
color_img_array = np.load('color_image_1650298896750529763.npy')
# convert array into dataframe
DF = pd.DataFrame(depth_img_array)
  
# save the dataframe as a csv file
DF.to_csv("depth_img.csv")

plt.imshow(depth_img_array, cmap='gray')
plt.show()

plt.imshow(color_img_array, cmap='gray')
plt.show()