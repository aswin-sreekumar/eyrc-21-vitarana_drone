#! /usr/bin/env python

'''
# Team ID:          VD_2537
# Theme:            Vitarana Drone
# Author List:      Jai Kesav, Aswin Sreekumar, Girish K, Greeshwar R S
# Filename:         Task_6_VD_2537_position_controller

# Functions:		__init__(), handle_destination_coords(), build_alt_update(), range_bottom(), im_breakin(), marker_detect_start(), edrone_position()
					imu_callback(), get_drone_vel(), set_qr_setpoint(), set_setpoints(), find_marker(), assign(), grid_coordinates_compute(), file_read_func()
					confine(), thresh_derivative(), stability(), travel(), position_control()

# Global variables: img_loc, manifest_csv_loc, package_drop_coordinates, total_package_number, sequenced_package_index,
					prev_sequenced_package_index, built_alt_update_flag, drone_orientation_euler, Kp, Ki, Kd, Kp2, Kd2, initial_error
					initial_error_distance, error, prev_error, error_derivative, sample_time, accumulator, error_distance, error_threshold
					velocity_threshold, meter_conv, cosines, proceed, state, target, rcval, eq_rcval, thresh, command, drone_position
					gripper_check_flag, find_marker_reply, find_marker_reply.data, task_stage_flag, drone_inital_position, changeme, final_cond
					init_once_flag, enable_setpoint_mapping.data, marker_escape_flag, land_init, set_setpoints_flag, fm_c1_chk, median_list, stage_dec
					pos_record, converging_point, position_setpoint, vel, building_alt, drop_coordinate_list, pickup_coordinate_list, A1_coordinates
					X1_coordinates, delivery_grid_coordinates, return_grid_coordinates, package_cell_size, rp_range, destination_coords

'''

'''
What this script does:
This is the drone position controller script.
This script subscribes the setpoint coordinates from height_mapping script.
The position controller computes YPR values required based on PID and is further published to attitude controller.
The position controller script also houses a smaller segment of Marker detection to avoid clash between published setpoints.
'''
#####################################################################################################################

# required libraries
import rospy
from vitarana_drone.msg import *
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
import math
from pid_tune.msg import PidTune
import cv2
import numpy as np
from vitarana_drone.srv import *
import tf
import copy
import csv

############################################################################################################################

# drone class definiton
class drone_control():

	# drone constructor
	def __init__(self):

		# node definition and name
		rospy.init_node('position_control', anonymous=True)

		# all variables used with meaning, sorted based on usage

		# GREE
		# self.img_loc = '/home/greesh/PycharmProjects/chumma/map_imp.jpg'
		# self.manifest_csv_loc = '/home/greesh/catkin_ws/src/vitarana_drone/scripts/sequenced_manifest.csv'

		# ASK
		# self.img_loc = '/home/aswinsreekumar/drone_ws/src/vitarana_drone/scripts/map_img.jpg'
		# self.manifest_csv_loc = '/home/aswinsreekumar/drone_ws/src/vitarana_drone/scripts/sequenced_manifest_original.csv'

		# JK
		self.img_loc = '/home/jk56/PycharmProjects/pythonProject/map_img.jpg'
		self.manifest_csv_loc = '/home/jk56/catkin_ws/src/vitarana_drone/scripts/final_scripts/sequenced_manifest_original.csv'

		# GK

		self.package_drop_coordinates = []									# List of package drop coordinates
		self.total_package_number = 0										# Total number of packages in csv file

		self.sequenced_package_index = 0									# Package index of current package
		self.prev_sequenced_package_index = -1
		self.built_alt_update_flag = True

		# variables for basic manuevering of drone
		self.drone_orientation_euler = [0.0, 0.0, 0.0] 						# drone's current orientation feedback variable
		self.Kp = [1638*0.04, 1638*0.04, 1.0, 1.0, 136*0.6/1.024]  			# idx 0- roll, 1-pitch, 2-yaw, 3-throttle, 4-eq_throttle
		self.Ki = [.3, .3, 0.0, 0.0, 192*0.08/1.024]  						# Equilibrium control as of now
		self.Kd = [749*0.3, 749*0.3, 0.0, 0.0, 385*0.3/1.024]
		self.Kp2 = [1638*0.04, 1638*0.04, 0, 1550*0.06/1.024]  				# Short range control (Used in this task)
		self.Kd2 = [749*0.3, 749*0.3, 0, 502*0.3/1.024]						# Short range control (Used in this task)
		self.initial_error = [0.0, 0.0, 0.0]  								# idx 0-x, 1-y, 2-z
		self.initial_error_distance = 0										# Error in position initial value
		self.error = [0.0, 0.0, 0.0] 										#Position error in metres
		self.prev_error = [0.0, 0.0, 0.0] 									#Position error for derivative
		self.error_derivative = [0.0, 0.0, 0.0]
		self.sample_time = 0.2
		self.accumulator = [0, 0, 0] 										# Position error for integrating
		self.error_distance = 0												# Current error distance
		self.error_threshold = [0.05, 0.05, 0.05]  							# used in def stability
		self.velocity_threshold = 0.01    									# used in def stability
		self.meter_conv = [110692.0702932625, 105292.0089353767, 1] 		# Factor for degrees to meters conversion(lat,long,alt to x,y,z)
		self.cosines = [0, 0, 0]											# Velocity control using direction cosines
		self.proceed = True 												# Flag variable for checking state
		self.state = [False, False, False] 									# Variable for axes state checking
		self.target = [19, 72, 0.45] 										# Dynamic setpoint variable (not used for this task)
		self.rcval = [1500, 1500, 1500, 1500] 								# Roll,pitch,yaw,throttle
		self.eq_rcval = [1500, 1500, 1500, 1496.98058] 						# Equilibrium base value for different loads
		self.thresh = 150 													# Threshold value for Kd to avoid spikes
		self.command = edrone_cmd()
		self.drone_position = [19, 72, 10.0] 								# Current drone position in GPS coordinates
		self.position_setpoint = [19, 72, 9.44]								# Setpoint for drone
		self.vel = [0, 0, 0]

		self.gripper_check_flag = "False"			# Gripper check callback value

		# Marker detection
		self.find_marker_reply = Float32()			# Object for Marker detection 1-start detection
		self.find_marker_reply.data = 0				# Variable for Marker detection 1-start detection
		self.task_stage_flag = 1					# Task stage control counter
		self.drone_inital_position = [19, 72, 8.44] # Initial position of drone
		self.changeme = 0
		self.final_cond = 0                         # TO INVOKE LANDING
		self.init_once_flag = 0
		self.enable_setpoint_mapping = Float32()
		self.enable_setpoint_mapping.data = 1  		# To disable publishing of setpoints from height_mapping when marker detector is active
		self.marker_escape_flag = False
		self.land_init = 0
		self.set_setpoints_flag = True
		self.fm_c1_chk = True
		self.median_list = [[], []]
		self.stage_dec = 1
		self.pos_record = [[0, 0], [0, 0], [0, 0]]
		self.converging_point = [19, 72]

		self.assign()

		self.building_alt = 0													   # Altitude of marker building
		self.A1_coordinates = [18.9998102845, 72.000142461, 16.757981]             # A1 coordinates
		self.X1_coordinates = [18.9999367615, 72.000142461, 16.757981]             # X1 coordinates
		self.delivery_grid_coordinates = list()                                    # Coordinates of the delivery grid pad
		self.return_grid_coordinates = list()									   # Coordinates of return grid pad
		self.package_cell_size = [1.5/110692.0702932625, 1.5/105292.0089353767]    # Cell size in radians

		self.mrk_localized = False
		self.attain_height = False

		# for obtaining and storing the building coordinates from csv file
		self.grid_coordinates_compute()
		self.file_read_func()
		self.rp_range = 2

		self.destination_coords = [0, 0, 0]		# Destination setpoint subscribed from Mapping script

		# self.file_read_func()
		self.SP_pub_data = Vector3()


		rospy.Subscriber('/edrone/gps', NavSatFix, self.edrone_position)						# Drone GPS subscriber
		rospy.Subscriber('/edrone/imu/data', Imu, self.imu_callback)							# IMU sensor data
		# rospy.Subscriber('/qr_scan_data', String, self.set_qr_setpoint)							# QR coordinates
		rospy.Subscriber('/edrone/range_finder_bottom',LaserScan, self.range_bottom)			# Sensor data bottom
		rospy.Subscriber('/find_marker', String, self.find_marker)								# Getting coordinates of marker and info
		rospy.Subscriber('/set_setpoint', Vector3, self.set_setpoints)							#  Setpoint published by height_mapping
		rospy.Subscriber('/edrone/gps_velocity', Vector3Stamped, self.get_drone_vel)			# Current drone velocity
		rospy.Subscriber('/marker_detection_start', Float32, self.marker_detect_start)          # marker_detection initiate
		rospy.Subscriber('/start_breaking',Float32,self.im_breakin)								# Braking enable/disable
		rospy.Subscriber('/current_package_index', String, self.build_alt_update)				# INdex of current package handled
		rospy.Subscriber('/destination_coordinates', Vector3, self.handle_destination_coords)   # Destination from control node


		self.qr_initiate = rospy.Publisher('/qr_initiate', Float32, queue_size=1)                 # QR scanning initiate publisher
		self.drone_command = rospy.Publisher('/drone_command', edrone_cmd, queue_size = 1)		  # Drone PWM publisher
		self.find_marker_reply_pub = rospy.Publisher('/find_marker_reply', Float32, queue_size=1) # Initiate Marker detection 1-start detection
		self.enable_setpoint_pub_mapping = rospy.Publisher('/enable_setpoint_pub_mapping_script', Float32, queue_size=1) # enabling setpoint pub initially true in mapping script used during marker detection
		self.SP_pub = rospy.Publisher('/SP', Vector3, queue_size=1)

###########################################################################################################################################

# Callback functions

	def handle_destination_coords(self, coords):
		'''
		Purpose:
		---
		Subscriber callback function of topic /destination_coordinates. Assigns x, y, z of the Vector3 message to destination_coords variable

		Input Arguments:
		---
		`coords` :  [ Vector3 ]
			Vector3 type message, holds the coordinates subscribed

		Returns:
		---
		None

		Example call:
		---
		Called automatically by Subscriber /destination_coordinates
		'''

		self.destination_coords[0] = coords.x
		self.destination_coords[1] = coords.y
		self.destination_coords[2] = coords.z

	def build_alt_update(self, index):
		'''
		Purpose:
		---
		< Short-text describing the purpose of this function >

		Input Arguments:
		---
		index :  String
		    Contains id / index of current package handled. Used for ensuring and debugging data publishing and subscribing

		Returns:
		---
		None

		Example call:
		---
		Called automatically by Subscriber /current_package_index
		'''

		if str(index.data[0]).isalpha():
			list_a = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
			self.sequenced_package_index = 10 + list_a.index(str(index.data[0]))
		else:
			self.sequenced_package_index = int(index.data[0])

		if index.data[1] == "D" and self.prev_sequenced_package_index != self.sequenced_package_index:
			if self.built_alt_update_flag:
				self.building_alt = self.package_drop_coordinates[self.sequenced_package_index][2]

			self.prev_sequenced_package_index = self.sequenced_package_index

	# Bottom sensor reading callback
	def range_bottom(self, range_bottom):
		'''
		Purpose:
		---
		Subscribes to the bottom sensor readings and stores in range_bottom_data

		Input Arguments:
		---
		`range_bottom` :  ranges [ Float32 ]
		    Sensor reading of bottom sensor

		Returns:
		---
		None

		Example call:
		---
		Called automatically by Subscriber /range_finder_bottom
		'''

		self.range_bottom_data = range_bottom.ranges[0]
		self.range_bottom_dist = range_bottom.ranges[0]

	def im_breakin(self,start):
		'''
		Purpose:
		---
		< Short-text describing the purpose of this function >

		Input Arguments:
		---
		`index` :  [ String ]
		    Contains id / index of current package handled. Used for ensuring and debugging data publishing and subscribing

		Returns:
		---
		None

		Example call:
		---
		Called automatically by Subscriber /current_package_index
		'''
		if '.7' not in str(start.data):
			self.rp_range = start.data/10
			self.Kp2 = [1638 * 0.04/self.rp_range, 1638 * 0.04/self.rp_range, 0, 1550 * 0.06 / 1.024]  # Short range control (Used in this task)
			self.Kd2 = [749 * 0.3/self.rp_range, 749 * 0.3/self.rp_range, 0, 502 * 0.3 / 1.024]  # Short range control (Used in this task)
		else:
			self.rp_range = start.data / 10
			self.Kp2 = [1638 * 0.04 / self.rp_range, 1638 * 0.04 / self.rp_range, 0, 1550 * 0.06 / 1.024]
			k = 1.1
			self.Kd2 = [k*749 * 0.3 / self.rp_range, k*749 * 0.3 / self.rp_range, 0, 502 * 0.3 / 1.024]

	def marker_detect_start(self,start):
		'''
		Purpose:
		---
		Inititate marker detection

		Input Arguments:
		---
		`start` :  [ Float32 ]
		    Contains whether to enable marker detection or not. The navigation control of drone is given to position controller (this) script
			when marker detection is invoked. This function and corresponding variables ensure the same.

		Returns:
		---
		None

		Example call:
		---
		Called automatically by Subscriber /marker_detection_start
		'''

		if start.data == 1 and self.init_once_flag == 0:                  # for both hovering and landing on the marker
			self.find_marker_reply.data = 1
			self.final_cond = 0
			self.init_once_flag = 1
			self.enable_setpoint_mapping.data = 0             # disabling setpoint pub  in mapping script
		elif start.data == 2 and self.init_once_flag == 0:                         # only for hovering and not landing on the marker
			self.find_marker_reply.data = 1
			self.final_cond = 0
			self.init_once_flag = 1
			self.enable_setpoint_mapping.data = 0  # disabling setpoint pub initially true in mapping script
		elif start.data == 0:
			self.find_marker_reply.data = 0
			self.init_once_flag = 0
			self.enable_setpoint_mapping.data = 1             # enabling setpoint pub  in mapping script


	#callback function for drone position
	def edrone_position(self, gps):
		'''
		Purpose:
		---
		Subscribes the current drone GPS position.

		Input Arguments:
		---
		`gps` :  [ NavSatFix ]
		    Stores the current drone GPS coordinates in required variables drone_position from NavSatFix message subscribed from /edrone/gps.

		Returns:
		---
		None

		Example call:
		---
		Called automatically by Subscriber /edrone/gps
		'''

		self.drone_position[0] = gps.latitude
		self.drone_position[1] = gps.longitude
		self.drone_position[2] = gps.altitude

	# IMU sensor reading callback function
	def imu_callback(self, msg):
		'''
		Purpose:
		---
		Subscribes the current drone IMU data in quaternion format from Imu message subscribed from /edrone/imu/data

		Input Arguments:
		---
		`msg` :  [ Imu ]
			Subscribes to the current quaternion coordinates of the drone from Imu sensors of Imu message

		Returns:
		---
		None

		Example call:
		---
		Called automatically by Subscriber /edrone/imu/data
		'''

		drone_orientation_quaternion = [0, 0, 0, 0]
		drone_orientation_quaternion[0] = msg.orientation.x
		drone_orientation_quaternion[1] = msg.orientation.y
		drone_orientation_quaternion[2] = msg.orientation.z
		drone_orientation_quaternion[3] = msg.orientation.w
		(self.drone_orientation_euler[0], self.drone_orientation_euler[1], self.drone_orientation_euler[2]) = tf.transformations.euler_from_quaternion([drone_orientation_quaternion[0], drone_orientation_quaternion[1], drone_orientation_quaternion[2], drone_orientation_quaternion[3]])
		self.drone_orientation_euler[0] *= 180 / math.pi
		self.drone_orientation_euler[1] *= 180 / math.pi
		self.drone_orientation_euler[2] *= 180 / math.pi

	# Callback function for getting current velocity(not used here)
	def get_drone_vel(self, v):
		'''
		Purpose:
		---
		Subscribes to the current drone velocity through Vector3Stamped message subscribed from /edrone/gps_velocity

		Input Arguments:
		---
		`v` :  [ Vector3Stamped ]
			Holds drone velocity in all directions

		Returns:
		---
		None

		Example call:
		---
		Called automatically by Subscriber /edrone/gps_velocity
		'''

		self.vel[0] = v.vector.x
		self.vel[1] = v.vector.y
		self.vel[2] = v.vector.z


	#Callback function for publishing setpoints (Used during coding the script to check for different positions) (not used here)
	def set_setpoints(self,setpoint):
		'''
		Purpose:
		---
		Subscribes to the setpoint through /set_setpoint in Vector3 message. The heght_mapping script publishes setpoint through this topic.
		Assigns this data to position_setpoint and PID for the same is initiated for drone movement. Makes sure the drone moves to the
		subscribed coordinates and stabilises.

		Input Arguments:
		---
		`setpoint` :  [ Vector3 ]
			Holds the setpoint of drone

		Returns:
		---
		None

		Example call:
		---
		Called automatically by Subscriber of /set_setpoint
		'''

		if self.set_setpoints_flag:
			setpoints = [setpoint.x, setpoint.y, setpoint.z]

			if setpoints[0:2] == [0,0]:
				val = range(2,3)
			else:
				val = range(3)

			if self.position_setpoint != setpoints:
				self.proceed = False
				self.initial_error_distance = 0

				for i in val:

					self.position_setpoint[i] = setpoints[i]
					self.state[i] = False
					self.initial_error[i] = (setpoints[i] - self.drone_position[i])*self.meter_conv[i]
					self.initial_error_distance += math.pow(self.initial_error[i], 2)

					if i == 2:
						self.target[i] = self.drone_position[i] + 0.05
					else:
						self.target[i] = self.drone_position[i]

				self.initial_error_distance = math.pow(self.initial_error_distance, 0.5)

			else:
				pass

		else:
			pass

	# Marker detection
	def find_marker(self, found):
		'''
		Purpose:
		---


		Input Arguments:
		---
		`found` :  [ String ]
			Contains data regarding marker detection and navigation towards marker.

		Returns:
		---
		None

		Example call:
		---
		Called automatically by Subscriber /marker_detection
		'''


		print("changeme : ", self.changeme)
		print("mrk reply : ", self.find_marker_reply.data)

		self.set_setpoints_flag = False

		# if self.append_det_val and found.data != "":
		# 	if str(found.data[0]).isdigit():
		# 		x = float(found.data.split(" ")[0]) / self.meter_conv[0]
		# 		y = float(found.data.split(" ")[1]) / self.meter_conv[1]
		# 		self.median_list[0].append(x)
		# 		self.median_list[1].append(y)


		if (found.data == "upward_marker_check" and self.changeme == 0):
			h = self.destination_coords[2] + 5
			#print("towards destination")
			self.position_setpoint[0] = self.destination_coords[0]
			self.position_setpoint[1] = self.destination_coords[1]

			if self.attain_height == False:
				self.position_setpoint[2] = self.destination_coords[2] + 5
				self.attain_height = True

			elif (abs(self.drone_position[2] - self.position_setpoint[2])<=0.5):
				self.position_setpoint[2] += 5



			self.mrk_localized = False


		elif found.data == "upward_marker_check_done" and self.changeme == 0:
			#print("towards detecting point")
			self.position_setpoint = list(self.drone_position)
			# self.position_setpoint[2] += 0.5  # for moving some x meters above the found coords
			self.changeme = 1
			self.set_setpoints_flag = True

			self.mrk_localized = False



		elif found.data == "upward_marker_check_done" and self.changeme == 1:

			self.attain_height = False
			state_check = [False, False, False]
			for idx in range(3):
				if abs(self.position_setpoint[idx] - self.drone_position[idx]) * self.meter_conv[idx] < 2.5:
					state_check[idx] = True

			if state_check == [True, True, True]:
				self.find_marker_reply.data = 2

			self.mrk_localized = False


		elif found.data != "" and found.data != "upward_marker_check_done" and found.data != "upward_marker_check" and found.data != "land_on_marker" and self.changeme == 1:
			# check 1
			self.attain_height = False
			drone_pos = list(self.drone_position)


			# self.position_setpoint[0] = float(found.data.split(" ")[0]) / self.meter_conv[0]
			# self.position_setpoint[1] = float(found.data.split(" ")[1]) / self.meter_conv[1]
			# self.position_setpoint[2] = self.building_alt + 5
			## bottom_dist = self.drone_position[2] - self.building_alt
			## #self.position_setpoint[2] -= bottom_dist - 5

			# try:
			# 	x = statistics.mean(self.median_list[0])
			# 	y = statistics.mean(self.median_list[1])
			# 	# err_x = self.drone_position[0] - x
			# 	# err_y = self.drone_position[1] - y
			# 	#
			# 	# if err_x < 0:
			# 	# 	x = min(self.median_list[0])
			# 	# else:
			# 	# 	x = max(self.median_list[0])
			# 	#
			# 	# if err_y < 0:
			# 	# 	y = min(self.median_list[1])
			# 	# else:
			# 	# 	y = max(self.median_list[1])
			#
			# 	self.position_setpoint[0] = x
			# 	self.position_setpoint[1] = y
			#
			# except statistics.StatisticsError:
			if self.drone_position[2] - self.building_alt >= 10:
				self.stage_dec = 2
			else:
				self.stage_dec = 1

			x = float(found.data.split(" ")[0])
			y = float(found.data.split(" ")[1])
			self.pos_record[0:2] = self.pos_record[1:]
			self.pos_record[2] = [x, y]
			x_mean, y_mean = [0, 0]
			for i in range(3):
				x_mean += self.pos_record[i][0]
				y_mean += self.pos_record[i][1]
			x_mean /= 3
			y_mean /= 3
			prox_chk = [True, True]
			for i in range(3):
				if abs(self.pos_record[i][0] - x_mean) > 0.1:
					prox_chk[0] = False
					break
				if abs(self.pos_record[i][1] - y_mean) > 0.1:
					prox_chk[1] = False
					break

			self.position_setpoint[0] = float(x)
			self.position_setpoint[1] = float(y)


			if prox_chk == [True, True]:
				self.changeme = 2
				x_cord = x_mean/self.meter_conv[0]
				y_cord = y_mean/self.meter_conv[1]

				self.converging_point = [x_mean/self.meter_conv[0], y_mean/self.meter_conv[1]]

				x_diff = x_cord - self.drone_position[0]
				y_diff = y_cord - self.drone_position[1]

				# if x_diff > 0:
				# 	x_cord += 0.5*x_diff
				# else:
				# 	x_cord -= 0.5*x_diff
				#
				# if y_diff > 0:
				# 	y_cord += 0.5*y_diff
				# else:
				# 	y_cord -= 0.5*y_diff

				self.position_setpoint[0] = x_cord
				self.position_setpoint[1] = y_cord

				if self.stage_dec == 1:
					# self.position_setpoint[2] = self.building_alt + 2
					pass
				else:
					self.position_setpoint[2] = self.building_alt + 5

				self.mrk_localized = True

				self.pos_record = [[0, 0], [0, 0], [0, 0]]

			self.fm_c1_chk = True

		elif found.data != "" and found.data != "upward_marker_check_done" and found.data != "upward_marker_check" and found.data != "land_on_marker" and self.changeme == 2:

			state_check = [False, False, True]
			for idx in range(2):
				if abs(self.converging_point[idx] - self.drone_position[idx]) * self.meter_conv[idx] < 1.5:  # 0.2
					state_check[idx] = True

			if state_check == [True, True, True]:
				self.changeme = 3
				self.find_marker_reply.data = 3

			# if self.fm_c1_chk:
			# 	state_check = [False, False, True]
			# 	for idx in range(2):
			# 		if abs(self.position_setpoint[idx] - self.drone_position[idx]) * self.meter_conv[idx] < 0.5:  # 0.2
			# 			state_check[idx] = True
			#
			# 	if state_check == [True, True, True]:
			# 		self.fm_c1_chk = False
			#
			# else:
			#
			# 	if str(found.data[0]).isdigit():
			#
			# 		x = float(found.data.split(" ")[0]) / self.meter_conv[0]
			# 		y = float(found.data.split(" ")[1]) / self.meter_conv[1]
			#
			# 		state_check = [False, False, True]
			# 		for idx in range(2):
			# 			if abs(self.position_setpoint[idx] - self.drone_position[idx]) * self.meter_conv[idx] < 0.1:  # 0.2
			# 				state_check[idx] = True
			#
			# 		if ((x - self.drone_position[0]) * (self.vel[0]) >= 0 and (y - self.drone_position[1]) * self.vel[1] <= 0 and abs(x - self.position_setpoint[0]) > 0.15 and abs(
			# 				y - self.position_setpoint[1]) > 0.15) or state_check == [True, True, True]:
			#
			# 			self.position_setpoint[0] = float(found.data.split(" ")[0]) / self.meter_conv[0]
			# 			self.position_setpoint[1] = float(found.data.split(" ")[1]) / self.meter_conv[1]
			# 			self.position_setpoint[2] = self.building_alt + 3
			# 			state_check = [False, False, True]
			# 			for idx in range(2):
			# 				if abs(self.position_setpoint[idx] - self.drone_position[idx]) * self.meter_conv[idx] < 0.1:  # 0.2
			# 					state_check[idx] = True
			# 			if state_check == [True, True, True]:
			# 				self.changeme = 3
			# 				self.find_marker_reply.data = 3
			#
			# 	else:
			# 		for idx in range(2):
			# 			if abs(self.position_setpoint[idx] - self.drone_position[idx]) * self.meter_conv[idx] < 0.1:  # 0.2
			# 				err = self.position_setpoint[idx] - self.drone_position[idx]
			# 				if err > 0:
			# 					self.position_setpoint[idx] -= 0.2/self.meter_conv[idx]
			# 				else:
			# 					self.position_setpoint[idx] += 0.2/ self.meter_conv[idx]


			# if state_check == [True, True, True]:
			# 		self.find_marker_reply.data = 3
			# 		self.changeme = 3


		elif found.data == "" and self.changeme == 1 and self.mrk_localized == False:
			if(abs(self.position_setpoint[2]-self.drone_position[2])<1.0):
				self.position_setpoint[2] += 5

		# elif found.data == "" and self.changeme == 3 and self.mrk_localized == False:
		# 	if (abs(self.position_setpoint[2] - self.drone_position[2]) < 1.0):
		# 		self.position_setpoint[2] += 10

		elif found.data != "" and ("LOL" in found.data) and found.data != "upward_marker_check_done" and found.data != "upward_marker_check" and found.data != "land_on_marker" and self.changeme == 3:
			# check 2
			drone_pos = list(self.drone_position)
			# straight out of detector

			# x = statistics.mean(self.median_list[0])
			# y = statistics.mean(self.median_list[1])
			# # err_x = self.drone_position[0] - x
			# # err_y = self.drone_position[1] - y
			# #
			# # if err_x < 0:
			# # 	x = min(self.median_list[0])
			# # else:
			# # 	x = max(self.median_list[0])
			# #
			# # if err_y < 0:
			# # 	y = min(self.median_list[1])
			# # else:
			# # 	y = max(self.median_list[1])
			#
			# self.position_setpoint[0] = x
			# self.position_setpoint[1] = y
			self.changeme = 4
			self.position_setpoint[0] = float(found.data.split(" ")[0]) / self.meter_conv[0]
			self.position_setpoint[1] = float(found.data.split(" ")[1]) / self.meter_conv[1]
			self.find_marker_reply.data = 4

		elif found.data != "" and found.data != "upward_marker_check_done" and found.data != "upward_marker_check" and found.data != "land_on_marker" and self.changeme == 4:

			self.find_marker_reply.data = 4        # 4
			self.median_list = [[], []]
			self.fm_c1_chk = False

		elif found.data == "land_on_marker" and self.changeme == 4:
			self.fm_c1_chk = False
			if self.final_cond == 0:
				self.changeme = 0
				self.find_marker_reply.data = 5

			elif self.final_cond == 1:
				if self.land_init == 0:
					bottom_dist = self.drone_position[2] - self.building_alt
					self.position_setpoint[2] = self.building_alt
					self.land_init = 1

				else:
					state_check = [False, False, False]
					for idx in range(3):
						if abs(self.position_setpoint[idx] - self.drone_position[idx]) * self.meter_conv[idx] < 2:
							state_check[idx] = True

					if state_check == [True, True, True]:
						self.land_init = 0
						self.changeme = 0
						self.find_marker_reply.data = 5
		else:
			self.set_setpoints_flag = True


##############################################################################################################

# Publisher functions

	# Computed YPRT published to attitude controller script
	def assign(self):
		'''
		Purpose:
		---
		Publishing the Rc values (YPR) to the attitude controller script.

		Input Arguments:
		---
		None

		Returns:
		---
		None

		Example call:
		---
		Called by assign() in travel()
		'''

		self.command.rcRoll = self.rcval[1]
		self.command.rcPitch = self.rcval[0]
		self.command.rcYaw = self.rcval[2]
		self.command.rcThrottle = self.rcval[3]

###############################################################################################################


# Utility functions



##############################################################################################################################33

# MISC functions
	def grid_coordinates_compute(self):
		'''
		Purpose:
		---
		Calculate the coordinates of grid cells of delivery and return using A1 X1 and cell size.
		Stores the calculated value in required variables

		Input Arguments:
		---
		None

		Returns:
		---
		None

		Example call:
		---
		Called by grid_coordinates_compute() in __init__()
		'''

		row_cell_coordinates_list = list()
		for i in range(0,3):
			for j in range(0,3):
				row_cell_coordinates_list.append([self.A1_coordinates[0]+(i*self.package_cell_size[0]),self.A1_coordinates[1]+(j*self.package_cell_size[1]),self.A1_coordinates[2]])
			self.delivery_grid_coordinates.append(copy.copy(row_cell_coordinates_list))
			del row_cell_coordinates_list[:]
			for j in range(0,3):
				row_cell_coordinates_list.append([self.X1_coordinates[0]+(i*self.package_cell_size[0]),self.X1_coordinates[1]+(j*self.package_cell_size[1]),self.X1_coordinates[2]])
			self.return_grid_coordinates.append(copy.copy(row_cell_coordinates_list))
			del row_cell_coordinates_list[:]

	# Extract data from csv file
	def file_read_func(self):

		'''
		Purpose:
		---
		Reads the sequenced_manifest file and processes it to read the coordinates of deliveries and returns.
		Stores those coordinates in package_drop_coordinates.

		Input Arguments:
		---
		None

		Returns:
		---
		None

		Example call:
		---
		Called by file_read_func() in __init__()
		'''

		with open(self.manifest_csv_loc) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			for row in csv_reader:
				if row[0] == "DELIVERY":
					package_list = row[2].split(';')
					self.package_drop_coordinates.append([float(package_list[0]), float(package_list[1]), float(package_list[2])])
					self.total_package_number += 1
				else:
					self.package_drop_coordinates.append('pongada')



	#Function for confining rcValues between limits(1000-2000)
	def confine(self, val):
		'''
		Purpose:
		---
		Limits the PID caluclated values of Rc (YPR) within 1000 to 2000 range (eqbm is 1500)

		Input Arguments:
		---
		`Val` : [ Float32 ]
			Contains the final computed value of Rc, which is made within limits set.

		Returns:
		---
		None

		Example call:
		---
		self.confine(2100)
		Called in equilibrium()
		'''

		if val > 2000:
			return 2000
		elif val < 1000:
			return 1000
		else:
			return val

	#Function to check whether it has reached the setpoint stable.
	def stability(self, idx):
		'''
		Purpose:
		---
		Checks for stability of drone position wrt the setpoint published. The threshold can be set for the same, but not controlled in the final task.


		Input Arguments:
		---
		`idx` :  [ int ]
			Holds the index of coordinates list (0,1,2). This corresponds to lat, long or alt to be checked for stability.
			state[idx] is made if the error of drone position from setpoint is lower than threshold set.

		Returns:
		---
		None

		Example call:
		---
		self.stability(2)
		'''

		# if self.proceed:
		if abs(self.position_setpoint[idx] - self.drone_position[idx])*self.meter_conv[idx] < self.error_threshold[idx] and abs(self.vel[idx]) < self.velocity_threshold:
			self.state[idx] = True
		else:
			self.state[idx] = False


################################################################################################################

# Algorithm functions

	# Function for PID calculation of rcValues
	def travel(self):
		'''
		Purpose:
		---
		Main algorithm in this script.
		Computes PID Rc values for desired setpoint using errors and publishes to attitude controller

		Input Arguments:
		---
		None

		Returns:
		---
		None

		Example call:
		---
		travel()
		Called in position_control()
		'''

		for i in range(3):
			self.error[i] = (self.position_setpoint[i] - self.drone_position[i]) * self.meter_conv[i]#converting error in GPS to x,y,z
			self.error_derivative[i] = (self.error[i] - self.prev_error[i]) / self.sample_time
			self.error_distance += self.error[i]**2  #velocity control
			self.prev_error[i] = self.error[i]

		self.error_distance = math.pow(self.error_distance, 0.5)#velocity control

		#velocity control
		for i in range(3):
			pass
			#self.cosines[i] = self.error[i]/self.error_distance

		#This block is always executed in this task as absolute distance is less.


		for i in range(2):
			self.rcval[i] = self.confine(self.eq_rcval[i] + self.Kp2[i] * self.error[i] + self.Kd2[i] * self.error_derivative[i])

		self.rcval[3] = self.confine(self.eq_rcval[3] + self.Kp2[3] * self.error[2] + self.Kd2[3] * self.error_derivative[2])

		self.assign()
		self.drone_command.publish(self.command)
		self.error_distance = 0


	# Function for controlling position
	def position_control(self):
		'''
		Purpose:
		---
		This function is repeatedly invoked in __main__
		Invokes travel() for drone to navigate to setpoint obtained and checks error between setpoint and drone position using stability()
		Publishes the setpoint data coordinates to marker detector script for navigation during marker detection stage.

		Input Arguments:
		---
		None

		Returns:
		---
		None

		Example call:
		---
		position_control()
		Called in __main__
		'''

		for i in range(3):
			self.stability(i)

		self.travel()

		for i in range(3):
			self.stability(i)

		#thedetector
		self.find_marker_reply_pub.publish(self.find_marker_reply)
		self.enable_setpoint_pub_mapping.publish(self.enable_setpoint_mapping)

		self.SP_pub_data.x = float(self.position_setpoint[0])
		self.SP_pub_data.y = float(self.position_setpoint[1])
		self.SP_pub_data.z = float(self.position_setpoint[2])
		self.SP_pub.publish(self.SP_pub_data)
															# for finding building coords


###############################################################################################################

#main function
if __name__ == "__main__":
	'''
	Purpose:
	---
	Main part of script. Includes node Frequency control.

	Input Arguments:
	---
	None

	Returns:
	---
	None

	Example call:
	---
	Called automatically by ROS
	'''

	drone_boi = drone_control()
	r = rospy.Rate(5) # Frequency of
	while not rospy.is_shutdown():
		drone_boi.position_control()
		# print(drone_boi.building_alt, drone_boi.sequenced_package_index)
		# print(drone_boi.changeme)
		r.sleep()

################################################################################################################

# END OF POSITION CONTROLLER PROGRAM
