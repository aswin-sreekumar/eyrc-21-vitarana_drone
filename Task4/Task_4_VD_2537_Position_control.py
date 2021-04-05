#! /usr/bin/env python


'''
TEAM ID: 2537
TEAM Name: JAGG
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

'''

Algorithm of task 4 alone

'''

############################################################################################################################

# drone class definiton
class drone_control():

	# drone constructor
	def __init__(self):

		# node definition and name
		rospy.init_node('position_control', anonymous=True)

		# all variables used with meaning, sorted based on usage

		self.img_loc = '/home/jk56/PycharmProjects/pythonProject/map_img.jpg'
		self.manifest_csv_loc = '/home/jk56/catkin_ws/src/vitarana_drone/scripts/manifest.csv'

		# variables for basic manuevering of drone
		self.drone_orientation_euler = [0.0, 0.0, 0.0] 						# drone's current orientation feedback variable
		self.Kp = [1638*0.04, 1638*0.04, 1.0, 1.0, 136*0.6/1.024]  			# idx 0- roll, 1-pitch, 2-yaw, 3-throttle, 4-eq_throttle
		self.Ki = [.3, .3, 0.0, 0.0, 192*0.08/1.024]  						# Equilibrium control as of now
		self.Kd = [749*0.3, 749*0.3, 0.0, 0.0, 385*0.3/1.024]
		self.Kp2 = [1638*0.04, 1638*0.04, 0, 1550*0.06/1.024]  				# Short range control (Used in this task)
		self.Kd2 = [749*0.3, 749*0.3, 0, 502*0.3/1.024]						# Short range control (Used in this task)
		self.initial_error = [0.0, 0.0, 0.0]  								# idx 0-x, 1-y, 2-z
		self.initial_error_distance = 0
		self.error = [0.0, 0.0, 0.0] 										#Position error in metres
		self.prev_error = [0.0, 0.0, 0.0] 									#Position error for derivative
		self.error_derivative = [0.0, 0.0, 0.0]
		self.sample_time = 0.2
		self.accumulator = [0, 0, 0] 										#Position error for integrating
		self.error_distance = 0
		self.error_threshold = [0.05, 0.05, 0.05]  							# used in def stability
		self.velocity_threshold = 0.01    									# used in def stability
		self.meter_conv = [110692.0702932625, 105292.0089353767, 1] 						#Factor for degrees to meters conversion(lat,long,alt to x,y,z)
		self.cosines = [0, 0, 0]											#Velocity control using direction cosines
		self.proceed = True 												#Flag variable for checking state
		self.state = [False, False, False] 									#Variable for axes state checking
		self.eq_flags = [False, False, False]
		self.target = [19, 72, 0.45] 										#Dynamic setpoint variable (not used for this task)
		self.rcval = [1500, 1500, 1500, 1500] 								#Roll,pitch,yaw,throttle
		self.eq_rcval = [1500, 1500, 1500, 1496.98058] 						#Equilibrium base value for different loads
		self.thresh = 150 													#Threshold value for Kd to avoid spikes
		self.command = edrone_cmd()
		self.drone_position = [19, 72, 10.0] 								#Current drone position in GPS coordinates
		self.position_setpoint = [19, 72, 9.44]
		self.vel = [0, 0, 0]

		# Height control variables
		self.height_flag = False 					# Height control enable flag
		self.calculated_height = 0.0				# Computed altitude of drone
		self.range_bottom_data = 0.0				# Bottom sensor reading
		self.height_once_flag = False				# Flag for avoiding looping in height_control fn
		self.destination_coords = [0.0,0.0,0.0] 	# Destination coordinates in a stage
		self.height_once_flag_prev = False			# To avoid height increase looping in height_control
		self.initial_height_flag = False			# Drone altitude control proceeding to destination
		self.next_destination = [0.0,0.0,0.0,0.0]	# Next destination coordinates after stage
		self.temp_destination = [0.0,0.0,0.0]		# Temporary destination variable

		# QR detection
		self.qr_position = [0, 0, 0]				# Decoded QR coordinates

		self.qr_start = Float32()					# Object for QR detection enable/disable
		self.qr_start.data = 0						# 0- disabled, 1-enabled
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
		self.enable_setpoint_mapping.data = 1  # initially true
		self.marker_escape_flag = False
		self.land_init = 0
		self.set_setpoints_flag = True

		self.assign()

		self.building_alt = 0

		self.package_drop_coordinates = []
		self.total_package_number = 0
		# for obtaining and storing the building coordinates from csv file
		self.file_read_func()



		rospy.Subscriber('/edrone/gps', NavSatFix, self.edrone_position)
		rospy.Subscriber('/edrone/imu/data', Imu, self.imu_callback)							# IMU sensor data
		rospy.Subscriber('/qr_scan_data', String, self.set_qr_setpoint)							# QR coordinates
		rospy.Subscriber('/edrone/range_finder_bottom',LaserScan, self.range_bottom)			# Sensor data bottom
		rospy.Subscriber('/find_marker', String, self.find_marker)								# Getting coordinates of marker and info
		rospy.Subscriber('/set_setpoint', Vector3, self.set_setpoints)
		rospy.Subscriber('/edrone/gps_velocity', Vector3Stamped, self.get_drone_vel)
		rospy.Subscriber('/marker_detection_start', Float32, self.marker_detect_start)           # marker_detection initiate publisher

		
		self.qr_initiate = rospy.Publisher('/qr_initiate', Float32, queue_size=1)                 # QR scanning initiate publisher
		self.drone_command = rospy.Publisher('/drone_command', edrone_cmd, queue_size = 1)		  # Drone PWM publisher
		self.find_marker_reply_pub = rospy.Publisher('/find_marker_reply', Float32, queue_size=1) # Initiate Marker detection 1-start detection
		self.enable_setpoint_pub_mapping = rospy.Publisher('/enable_setpoint_pub_mapping_script', Float32, queue_size=1) # enabling setpoint pub initially true in mapping script used during marker detection

###########################################################################################################################################

# Callback functions

	def marker_detect_start(self,start):
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
		self.drone_position[0] = gps.latitude
		self.drone_position[1] = gps.longitude
		self.drone_position[2] = gps.altitude

	# IMU sensor reading callback function
	def imu_callback(self, msg):
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
		self.vel[0] = v.vector.x
		self.vel[1] = v.vector.y
		self.vel[2] = v.vector.z

	# Bottom sensor reading callback
	def range_bottom(self,range_bottom):
		self.range_bottom_data = range_bottom.ranges[0]
		self.range_bottom_dist = range_bottom.ranges[0]
		self.height_control()

	# Gets the setpoint from the qr code detecting node and gives it to the required variable here
	def set_qr_setpoint(self,set_point):
		qr_list = []
		for i in set_point.data.split(','):
			qr_list.append(float(i))
		self.qr_position = list(qr_list)
		self.position_setpoint_dest = list(qr_list)

	#Callback function for publishing setpoints (Used during coding the script to check for different positions) (not used here)
	def set_setpoints(self,setpoint):
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
				self.eq_flags = [False, False, False]

			else:
				pass

		else:
			pass

	# Marker detection
	def find_marker(self, found):
		self.set_setpoints_flag = False

		if (found.data == "upward_marker_check" and self.changeme == 0):
			if (self.drone_position[2] >= (self.position_setpoint[2] - 0.4)) and (self.drone_position[2] < (self.position_setpoint[2] + 0.4)):
				self.position_setpoint[2] += 5

		elif found.data == "upward_marker_check_done" and self.changeme == 0:
			self.position_setpoint = list(self.drone_position)
			self.position_setpoint[2] += 0  # for moving some x meters above the found coords
			self.changeme = 1
			self.set_setpoints_flag = True

		elif found.data == "upward_marker_check_done" and self.changeme == 1:
			state_check = [False, False, False]
			for idx in range(3):
				if abs(self.position_setpoint[idx] - self.drone_position[idx]) * self.meter_conv[idx] < 2:
					state_check[idx] = True

			if state_check == [True, True, True]:
				self.find_marker_reply.data = 2

		elif found.data != "" and found.data != "upward_marker_check_done" and found.data != "upward_marker_check" and found.data != "land_on_marker" and self.changeme == 1:
			# check 1
			drone_pos = list(self.drone_position)
			self.changeme = 2
			self.position_setpoint[0] = float(found.data.split(" ")[0]) / self.meter_conv[0] + drone_pos[0]
			self.position_setpoint[1] = float(found.data.split(" ")[1]) / self.meter_conv[1] + drone_pos[1]
			bottom_dist = self.drone_position[2] - self.building_alt

		elif found.data != "" and found.data != "upward_marker_check_done" and found.data != "upward_marker_check" and found.data != "land_on_marker" and self.changeme == 2:
			state_check = [False, False, False]
			for idx in range(3):
				if abs(self.position_setpoint[idx] - self.drone_position[idx]) * self.meter_conv[idx] < 1:  # 0.2
					state_check[idx] = True
			if state_check == [True, True, True]:
				if self.marker_escape_flag == True:
					self.marker_escape_flag = False
					self.changeme = 0
					self.find_marker_reply.data = 5
				else:
					self.find_marker_reply.data = 3
					self.changeme = 3

		elif found.data != "" and ("LOL" in found.data) and found.data != "upward_marker_check_done" and found.data != "upward_marker_check" and found.data != "land_on_marker" and self.changeme == 3:
			# check 2
			drone_pos = list(self.drone_position)
			# straight out of detector
			self.changeme = 4
			self.position_setpoint[0] = float(found.data.split(" ")[0]) / self.meter_conv[0] + drone_pos[0]
			self.position_setpoint[1] = float(found.data.split(" ")[1]) / self.meter_conv[1] + drone_pos[1]

		elif found.data != "" and found.data != "upward_marker_check_done" and found.data != "upward_marker_check" and found.data != "land_on_marker" and self.changeme == 4:

			self.find_marker_reply.data = 4        # 4

		elif found.data == "land_on_marker" and self.changeme == 4:
			if self.final_cond == 0:
				self.changeme = 0
				self.find_marker_reply.data = 5

			elif self.final_cond == 1 :
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
		self.command.rcRoll = self.rcval[1]
		self.command.rcPitch = self.rcval[0]
		self.command.rcYaw = self.rcval[2]
		self.command.rcThrottle = self.rcval[3]

###############################################################################################################


# Utility functions


##############################################################################################################################33

# MISC functions

	# Extract data from csv file
	def file_read_func(self):

		with open(self.manifest_csv_loc) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			for row in csv_reader:
				self.package_drop_coordinates.append([float(row[1]), float(row[2]), float(row[3])])
				self.total_package_number += 1

	def build_alt_update(self):
		o = 1
		for i in range(self.total_package_number):
			o += 1
			if abs(self.drone_position[0] * self.meter_conv[0] - self.package_drop_coordinates[i][0] * self.meter_conv[0]) < 8 and abs(self.drone_position[1] * self.meter_conv[1] - self.package_drop_coordinates[i][1] * self.meter_conv[1]) < 8:
				self.building_alt = self.package_drop_coordinates[i][2]


	#Function for confining rcValues between limits(1000-2000)
	def confine(self, val):
		if val > 2000:
			return 2000
		elif val < 1000:
			return 1000
		else:
			return val

	#Function for restricting value of derivative term
	def thresh_derivative(self, derivative):
		if abs(derivative) > self.thresh:
			if derivative > 0:
				return self.thresh
			else:
				return -1*self.thresh
		else:
			return derivative

	#Function to check whether it has reached the setpoint stable.
	def stability(self, idx):
		# if self.proceed:
		if abs(self.position_setpoint[idx] - self.drone_position[idx])*self.meter_conv[idx] < self.error_threshold[idx] and abs(self.vel[idx]) < self.velocity_threshold:
			self.state[idx] = True
		else:
			self.state[idx] = False


################################################################################################################

# Algorithm functions

	# This function will be used when the package has mass. It hasn't been used now
	def equilibrium(self):

		for i in range(3):
			self.error[i] = (self.target[i] - self.drone_position[i]) * self.meter_conv[i]
			self.error_derivative[i] = (self.error[i] - self.prev_error[i]) / self.sample_time
			self.prev_error[i] = self.error[i]
			self.accumulator[i] += self.error[i] * self.sample_time

		for i in range(2):
			if not self.state[i]:
				self.rcval[i] = self.confine(self.eq_rcval[i] + self.Kp2[i] * self.error[i] + self.Kd2[i] * self.error_derivative[i] + self.accumulator[i]*self.Ki[i])
			else:
				self.rcval[i] = self.confine(self.eq_rcval[i] + self.Kp2[i] * self.error[i] + self.Kd2[i] * self.error_derivative[i])

		if not self.state[2]:
			self.rcval[3] = self.confine(self.eq_rcval[3] + self.Kp2[3] * self.error[2] + self.Kd2[3] * self.error_derivative[2] + self.accumulator[2]*self.Ki[3])
		else:
			self.rcval[3] = self.confine(self.eq_rcval[3] + self.Kp2[3] * self.error[2] + self.Kd2[3] * self.error_derivative[2])

		if self.state[0] and not self.eq_flags[0]:
			self.eq_rcval[0] = self.command.rcRoll
			self.eq_flags[0] = True
			self.accumulator[0] = 0
		if self.state[1] and not self.eq_flags[1]:
			self.eq_rcval[1] = self.command.rcPitch
			self.eq_flags[1] = True
			self.accumulator[1] = 0
		if self.state[2] and not self.eq_flags[2]:
			self.eq_rcval[3] = self.command.rcThrottle
			self.eq_flags[2] = True
			self.accumulator[2] = 0

		if self.state[0] and self.state[1] and self.state[2]:
			self.proceed = True

		self.assign()
		self.drone_command.publish(self.command)
		rospy.loginfo("In equilib.")

	# Function for PID calculation of rcValues
	def travel(self):

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
		for i in range(3):
			self.stability(i)

		self.travel()

		for i in range(3):
			self.stability(i)

		#thedetector
		self.find_marker_reply_pub.publish(self.find_marker_reply)
		self.enable_setpoint_pub_mapping.publish(self.enable_setpoint_mapping)
		self.build_alt_update() 													# for finding building coords

	# Computing altitude of drone when enabled
	def height_control(self):
		if(self.height_flag == True):
			if((self.next_destination[2] - self.drone_inital_position[2] > 5) and self.initial_height_flag == False ):
				self.temp_destination = self.destination_coords
				self.destination_coords[0] = self.drone_inital_position[0]
				self.destination_coords[1] = self.drone_inital_position[1]
				self.destination_coords[2] = self.drone_inital_position[2] + 11
				self.initial_height_flag = True

			if(self.initial_height_flag == True):
				if (self.drone_position[2]>(self.destination_coords[2]-1) and self.drone_position[2]<(self.destination_coords[2]+1)):
					self.destination_coords = self.temp_destination
					self.initial_height_flag = False

			if(self.range_bottom_data<4):
				if(self.destination_coords[2]<self.drone_position[2]):
					if(self.height_once_flag == False):
						self.calculated_height = self.drone_position[2] + 10
						self.height_once_flag = True

					if(self.height_once_flag == True and self.height_once_flag_prev == False):
						self.calculated_height = self.calculated_height - 7
				else:
					self.calculated_height = self.destination_coords[2]

			elif(self.range_bottom_data > 20):
				self.calculated_height = self.destination_coords[2]
				self.height_once_flag = False

			elif(abs(self.destination_coords[2]-self.drone_position[2]) < 4):
				self.calculated_height = self.destination_coords[2]

		else:
			self.calculated_height = self.destination_coords[2]

			self.height_once_flag = False

		self.height_once_flag_prev = self.height_once_flag


###############################################################################################################

#main function
if __name__ == "__main__":
	drone_boi = drone_control()
	r = rospy.Rate(5) # Frequency of
	while not rospy.is_shutdown():
		drone_boi.position_control()
		r.sleep()

################################################################################################################

# END OF POSITION CONTROLLER PROGRAM
