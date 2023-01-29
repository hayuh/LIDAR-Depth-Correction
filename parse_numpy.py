#Inspired by: https://www.youtube.com/watch?v=Mng57Tj18pc&t=1837s
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf

depth_img_array = np.load('86.75\depth_image_1652109334245600312.npy') #720 1280-length arrays
color_img_array = np.load('61.125\color_image_1652108090835074281.npy')
# convert array into dataframe
#DF_depth = pd.DataFrame(depth_img_array)
#DF_color = pd.DataFrame(color_img_array) #Does not work because of error: ValueError: Must pass 2-d input. shape
  
# save the dataframe as a csv file
'''
DF_depth.to_csv("depth_img.csv")
DF_color.to_csv("color_img.csv")
print(depth_img_array.shape)
#print(color_img_array.shape)
'''

plt.imshow(depth_img_array, cmap='gray')
plt.show()

plt.imshow(color_img_array, cmap='gray')
plt.show()
'''
print(psutil.cpu_times())
#test
'''