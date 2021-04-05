#!/usr/bin/env python

'''

TEAM ID : VD_2537
TEAM NAME : JAGG

'''
'''
This is a boiler plate script that contains an example on how to subscribe a rostopic containing camera frames
and store it into an OpenCV image to use it further for image processing tasks.
Use this code snippet in your code or you can also continue adding your code in the same file
'''


from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import rospy
from vitarana_drone.msg import SetPoints

import os
from std_msgs.msg import *

'''
This program run parallel to position and attitde controller script , scans the QR code on package and publishes to the position controller
when invoked using a message published from the position controller.
The targets are sent to position controller as a String
Required IP thresholding is performed to generalise the QR scanning in most conditions.
'''
class image_proc():

	# Initialise everything
	def __init__(self):
		rospy.init_node('barcode_test') #Initialise rosnode
		self.image_sub = rospy.Subscriber("/edrone/camera/image_raw", Image, self.image_callback) #Subscribing to the camera topic
		rospy.Subscriber("/qr_initiate",Float32, self.data_find)
		self.qr_send = rospy.Publisher("/qr_scan_data",String,queue_size = 1)
		self.img = np.empty([]) # This will contain your image frame from camera
		self.bridge = CvBridge()
		self.check = 0
		self.check1 = 0
		self.check_shade = 0

		self.data_obj = String()
		self.data_obj.data = ""
		self.data_obj_list = []

	def data_find(self,data):
		if(data.data == 2.0):
			if self.data_obj_list != []:
				self.qr_send.publish(self.data_obj)
				print("DATA SENT\n")
				rospy.sleep(20)
				os.system("rosnode kill barcode_test")



	'''the functions shadow_removal_1 and shadow_removal_2 were developed to remove the shadow that may be obstructing the qr code.
	in this script we are using the shadow_removal_2 function.Both the functions are developed with the help of functions present in opencv(cv2) module.
	for qr code detection we are using pyzbar module'''


	def shadow_removal_1(self, image):

		# shot 1

		img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# ret, org = cv2.threshold(img, 75, 255, cv2.THRESH_BINARY)
		# org = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

		ret, mask = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
		ret, mask_org = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

		img2 = cv2.bitwise_and(img, img, mask=mask)

		img_thresh = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
		img_thresh = cv2.bitwise_and(img_thresh, img_thresh, mask=mask)

		img_or = cv2.bitwise_or(img_thresh, mask_org)
		# kernel = np.ones((2, 2), np.uint8)
		# erosion = cv2.erode(img_or, kernel, iterations=2)


		kernel = np.ones((2, 2), np.uint8)
		erosion = cv2.erode(img_or, kernel, iterations=2)

		data = decode(erosion)
		if data != []:
			self.check_shade += 1
			for obj in data:
				# print('Type : ', obj.type)

				coord = 'Data : ', obj.data

				print("shade_boi", coord, self.check_shade)



	def shadow_removal_2(self,img):

		# shot 2.0


		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		kernel = np.ones((7, 7), np.uint8)
		dilated_img = cv2.dilate(gray, kernel, iterations=1)
		bg_img = cv2.medianBlur(dilated_img, 21)
		erosion = cv2.erode(bg_img, np.ones((5, 5), np.uint8), iterations=2)

		diff_img = 255 - cv2.absdiff(gray, erosion)
		norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

		ador = cv2.bitwise_or(norm_img, gray)


		data = decode(ador)
		self.temp_var=['1','2','3']
		coord = ()
		if data != []:
			self.check +=1
			for obj in data:
				# print('Type : ', obj.type)

				coord = ('Data : ', obj.data)
				print("ador", coord,self.check)
		else:
			adaptive_shadow = cv2.adaptiveThreshold(ador, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
			data_ad = decode(adaptive_shadow)

			if data_ad != []:
				self.check1 +=1
				for obj in data_ad:
					# print('Type : ', obj.type)

					coord = ('Data : ', obj.data)

					print("adaptive_shadow", coord,self.check1)

		if coord != ():
			# self.data_obj =	coord[1]
			#self.temp_var = coord[1].split(',')

			# self.data_obj.latitude = float(self.temp_var[0])
			# self.data_obj.longitude = float(self.temp_var[1])
			# self.data_obj.altitude = float(self.temp_var[2])
			self.data_obj.data = coord[1]
			self.data_obj_list = coord[1].split(',')


	def image_callback(self, data):
		try:
			self.img = self.bridge.imgmsg_to_cv2(data, "bgr8") # Converting the image to OpenCV standard image
			self.shadow_removal_2(self.img)


		except CvBridgeError as e:
			print(e)
			return

if __name__ == '__main__':
	image_proc_obj = image_proc()
	while (True):
		pass
		#cv2.imshow('vid_Cap',image_proc_obj.img)


	#cv2.destroyAllWindows()
	#rospy.spin()

# END OF QR SCAN SCRIPT
