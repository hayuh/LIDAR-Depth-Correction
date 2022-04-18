import rosbag
import rospy
import os
import sys
import cv2
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class Args:
	"""
	Things that can be arguments. For now its just rosbag
	"""
	rosbag_name = None

def parseArgs():
	"""
	Parse the arguments.
	"""
	args = Args()
	if len(sys.argv) < 2:
		print("Usage: python read_rgbd.py <rosbag>")
		sys.exit(1)
	args.rosbag_name = sys.argv[1]
	return args

def parseBag(args, file_path):
	"""
	Parses the bag from the argument.
	"""
	bag = rosbag.Bag(args.rosbag_name)
	bridge = CvBridge()

	i=0
	for topic, msg, t in bag.read_messages():

		print("Msg %d:" % i)
		print(topic, type(msg), str(t))
		print("width: %d\nheight: %d\nencoding: %s\nstep: %d" % (int(msg.width), int(msg.height), msg.encoding, int(msg.step)))

		cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
		print(cv_image)
		print(topic)
		pil_image = Image.fromarray(cv_image)
		
		print(cv_image.dtype)
		print(cv_image.shape)


		if topic == "/camera/color/image_raw":
			np.save("%s/color/color_image_%s.npy" % (file_path, str(t)), cv_image)
			pil_image.save("%s/color/images/color_image_%s.jpg" % (file_path, str(t)))
		else:
			np.save("%s/depth/depth_image_%s.npy" % (file_path, str(t)), cv_image)
			pil_image.mode = "I"
			pil_image.point(lambda i:i*(1./256)).convert('L').save("%s/depth/images/depth_image_%s.jpg" % (file_path, str(t)))
			
		# plt.hist(cv_image.ravel(), bins=100)
		# if topic == "/camera/color/image_raw":
		# 	plt.imshow(cv_image)
		# else:
		# plt.imshow(cv_image, cmap=plt.cm.gray)
		# plt.show()

		i+=1
		print(" ")

		# break
		
		

if __name__ == "__main__":
	args = parseArgs()
	file_path = args.rosbag_name[:-4]
	try:
		os.mkdir("%s" % file_path)
	except Exception as e:
		print("Warn: %s" % e)

	try:
		os.mkdir("%s/color" % file_path)
	except Exception as e:
		print("Warn: %s" % e)

	try:
		os.mkdir("%s/color/images" % file_path)
	except Exception as e:
		print("Warn: %s" % e)

	try:
		os.mkdir("%s/depth" % file_path)
	except Exception as e:
		print("Warn: %s" % e)

	try:
		os.mkdir("%s/depth" % file_path)
	except Exception as e:
		print("Warn: %s" % e)

	try:
		os.mkdir("%s/depth/images" % file_path)
	except Exception as e:
		print("Warn: %s" % e)

	
	parseBag(args, file_path)
