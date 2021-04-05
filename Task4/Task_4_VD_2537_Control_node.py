#! /usr/bin/env python


'''
TEAM ID: 2537
TEAM Name: JAGG
'''

'''
Node name : CONTROLLER NODE SCRIPT
Node desc : Controls the flow of the task and works with each stage of the drone navigation and funtions for TASK4
'''

'''
Algorithm for Task 4:
The required coordinates of package pickup and drop coordinates are computed. 
Binary grid mapping is implemented with region of interest around the package drop building coordinates. 
The drone's sensors are used to map the surroundings above a specific height. 
Using the map data, Modified A* algorithm is implemented along with smoothing functions to improvise set points 
and make it a realistic navigation giving an efficient algorithm of path planning. 
All packages undergo the same stages of delivery. 
Braking algorithm is integrated into the control flow algorithm in order to prevent the drone from overshooting. 
Upon reaching the building, the drone searches for marker using Marker Detection and lands at the required place with accuracy. 
Upon successful delivery of a package, the drone proceeds to the warehouse grid to deliver the next package. 
The drone goes back to the initial position after completion of deliveries, before indicating the end of task.
'''
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

###################################################################################################################

class control_node():

    # drone constructor
    def __init__(self):

    	# node definition and name
    	rospy.init_node('control_node', anonymous=True)    # Node name and declaration

        self.img_loc = '/home/jk56/PycharmProjects/pythonProject/map_img.jpg'
        self.manifest_csv_loc = '/home/jk56/catkin_ws/src/vitarana_drone/scripts/manifest.csv'

        # Main control flow node VARIABLES
        self.destination_coords_object = Vector3()        # Object for destination publisher
        self.task_stage_flag = 1                            # Current stage
        self.drone_inital_position = [19.0, 72.0, 8.44]     # Drone initial position
        self.destination_coords = [0.0, 0.0, 0.0]           # Published destination coordinates
        self.meter_conv = [110693.227, 105078.267, 1] 		# Factor for degrees to meters conversion(lat,long,alt to x,y,z)
        self.map_altitude = 0.0                             # Altitude of 2D map (threshold)
        self.drone_position = [19, 72, 10.0]                # Current drone position in GPS coordinates
        self.controller_info_obj = String()                 # Object for publishing package info
        self.path_planner_enable_flag = False

        # Gripper Flag
        self.gripper_flag = False                           # Enable or disable gripper
        self.gripper_check_flag = False                     # State of gripper service

        # TASK 4 specific variables
        self.first_package_coordinates = [18.9999864489, 71.9999430161, 8.44099749139]  # A1 coordinates
        self.package_cell_size = [0.000013552, 0.000014245]                             # Cell size in radians
        self.package_pickup_coordinates = list()                                        # Package pick up coordinates
        self.package_drop_coordinates = list()                                          # Package drop coordinates
        self.total_package_number = 0                                                   # Total packages to be delivered
        self.package_altitude = self.drone_inital_position[2]		                    # Altitude of packages
        self.current_package_index = 0      						                    # Index of current package
        self.next_destination = [0.0, 0.0, 0.0]                                         # Next destination of task

        # Marker detection
        self.marker_detection_start = Float32()  # Object for Marker detection 1-start detection
        self.marker_detection_start.data = 0     # Marker detection control flag
        self.find_marker_reply = 0               # Marker detection confirmation

        self.braking_coordinates = [0.0, 0.0, 0.0]  # Braking algorithm setpoint
        self.braking_difference = [0.0, 0.0, 0.0]   # Difference in braking algorithm

#################################################################################################################################################

# Subscriber and publishers

        rospy.Subscriber('/edrone/gps', NavSatFix, self.edrone_position)            # Position of drone
        rospy.Subscriber('/edrone/gripper_check',String,self.gripper_function)      # Gripper state check
        rospy.Subscriber('/find_marker_reply', Float32, self.find_marker_reply_fun)

        self.destination_coords_pub = rospy.Publisher('/destination_coordinates', Vector3, queue_size=1)	# Publishes destination to mapper
        self.controller_info_pub = rospy.Publisher('/controller_info', String, queue_size=1)                # Publishes ackage information
        self.marker_detection_start_pub = rospy.Publisher('/marker_detection_start', Float32,queue_size=1)  # marker_detection initiate publisher

##################################################################################################################################################

# Call back functions and publishers

    # capturing the reply sent to the detector from the position control
    def find_marker_reply_fun(self, reply):
        self.find_marker_reply = reply.data

    #callback function for drone position
    def edrone_position(self, gps):
        self.drone_position[0] = gps.latitude
        self.drone_position[1] = gps.longitude
        self.drone_position[2] = gps.altitude

    # Destination publisher to position controller
    def destination_publisher_func(self):
        self.destination_coords_object.x = self.destination_coords[0]
        self.destination_coords_object.y = self.destination_coords[1]
        self.destination_coords_object.z = self.destination_coords[2]
        self.destination_coords_pub.publish(self.destination_coords_object)

    # Publish package information
    def package_info_pub_func(self):
        self.controller_info_obj.data = str(self.current_package_index)
        self.controller_info_obj.data += str(self.task_stage_flag)
        if(self.path_planner_enable_flag == True):
            self.controller_info_obj.data += 'Y'
        else:
            self.controller_info_obj.data += 'N'
        self.controller_info_pub.publish(self.controller_info_obj)

##################################################################################################################################################

# Utility functions

    # Extract data from csv file
    def file_read_func(self):
        cell_data = [0, 0]
        #'/home/greesh/catkin_ws/src/vitarana_drone/scripts/manifest.csv'
        with open(self.manifest_csv_loc) as csv_file:
        	csv_reader = csv.reader(csv_file, delimiter=',')
        	for row in csv_reader:
        		self.package_drop_coordinates.append([float(row[1]), float(row[2]), float(row[3])])
        		self.total_package_number += 1

    # Decision to pickup or drop the package will be sent here
    def gripper_function(self,data):
        self.gripper_check_flag = data.data
        if(self.gripper_flag == True):
                  if(data.data == "True"):
        	                   self.gripper_service_client(True)
        else:
            self.gripper_service_client(False)

	# Broadcasts the activation and deactivation commands to the service
    def gripper_service_client(self,activate_gripper):
        rospy.wait_for_service('/edrone/activate_gripper') #waiting for the service to be advertised
        try:
        	self.gripper = rospy.ServiceProxy('edrone/activate_gripper', Gripper) #setting up a local proxy for using the service. Arguments are the name of the service and the service type
        	self.req = self.gripper(activate_gripper)
        except rospy.ServiceException as e:
        	pass

####################################################################################################################################################

# MISC functions

    # Checking if drone is inside the window for picking packages wrt coords and error window
    def package_window_check(self, coords, error):
        if(abs(self.drone_position[0]-coords[0])<(error[0]/self.meter_conv[0])):
            if(abs(self.drone_position[1]-coords[1])<(error[1]/self.meter_conv[1])):
                if(abs(self.drone_position[2]-coords[2])<(error[2]/self.meter_conv[2])):
                    return 1
        else:
            return 0

####################################################################################################################################################

# Algorithm

    # Inital data feed into the script (file read and package pickup compute) and compute map altitude
    def initial_setup_func(self):
    	self.package_pickup_coordinates.append(self.first_package_coordinates)
    	self.package_pickup_coordinates.append([self.first_package_coordinates[0]+(2*self.package_cell_size[0]),self.first_package_coordinates[1],self.first_package_coordinates[2]])
    	self.package_pickup_coordinates.append([self.first_package_coordinates[0]+self.package_cell_size[0],self.first_package_coordinates[1]+self.package_cell_size[1],self.first_package_coordinates[2]])
    	self.file_read_func()
        for i in self.package_drop_coordinates:
            if(self.map_altitude < i[2]):
                self.map_altitude = i[2]
        self.next_destination = self.package_pickup_coordinates[1]

	# Control flow of the task
    def control_flow(self):

        stg_9_error_window = [3.0,3.0, float('inf')]

        if self.current_package_index == self.total_package_number - 1:
            stg_9_error_window = [10.0, 10.0, float('inf')]

        # All packages delivered, returning to initial drone position
        if(self.current_package_index == self.total_package_number):
            self.destination_coords[0] = self.drone_inital_position[0]
            self.destination_coords[1] = self.drone_inital_position[1]
            self.destination_coords[2] = self.drone_inital_position[2]
            self.gripper_flag = False
            self.path_planner_enable_flag = False
            self.destination_publisher_func()
            if(self.package_window_check(self.drone_inital_position,[0.2,0.2,0.2])):
                print("TASK COMPLETED .... ")
            return

        # Drone goes to package pickup coord above
        elif(self.task_stage_flag == 1):
            self.destination_coords[0] = self.package_pickup_coordinates[self.current_package_index][0]
            self.destination_coords[1] = self.package_pickup_coordinates[self.current_package_index][1]
            self.destination_coords[2] = self.package_altitude + 8
            self.destination_publisher_func()
            self.gripper_flag = False
            self.path_planner_enable_flag = False
            if(self.package_window_check([self.package_pickup_coordinates[self.current_package_index][0],self.package_pickup_coordinates[self.current_package_index][1],self.package_altitude+8], [0.45,0.45,float('inf')])):
                self.task_stage_flag = 2

        # Descend and pick up package
        elif(self.task_stage_flag == 2):
            self.destination_coords[0] = self.package_pickup_coordinates[self.current_package_index][0]
            self.destination_coords[1] = self.package_pickup_coordinates[self.current_package_index][1]
            self.destination_coords[2] = self.package_altitude - 1
            self.gripper_flag = True
            self.path_planner_enable_flag = False
            self.destination_publisher_func()
            if(self.gripper_check_flag == "True"):
                for i in range(2):
                    self.braking_difference[i] = self.package_drop_coordinates[self.current_package_index][i] -  self.package_pickup_coordinates[self.current_package_index][i]
                    if (self.braking_difference[i] >= 0):
                        self.braking_difference[i] -= 2/self.meter_conv[i]
                    else:
                        self.braking_difference[i] += 2/self.meter_conv[i]
                    self.braking_coordinates[i] = self.package_pickup_coordinates[self.current_package_index][i] + self.braking_difference[i]

                self.task_stage_flag = 4
                self.path_planner_enable_flag = True

        # Go to destination position 1m before destination
        elif (self.task_stage_flag == 4):
            self.destination_coords[0] = self.braking_coordinates[0]
            self.destination_coords[1] = self.braking_coordinates[1]
            if (self.package_window_check(self.braking_coordinates, [10, 10, float('inf')])):
                self.destination_coords[2] = self.package_drop_coordinates[self.current_package_index][2] + 7
            else:
                self.destination_coords[2] = self.map_altitude + 2
            self.destination_publisher_func()
            self.path_planner_enable_flag = True
            self.gripper_flag = True
            if (self.package_window_check(self.braking_coordinates, [6, 6, float('inf')])):  
                self.task_stage_flag = 4.1
        
        # Marker improvised algorithm
        elif self.task_stage_flag == 4.1:
            if (self.package_window_check(self.braking_coordinates, [1, 1, float('inf')])): 
                self.task_stage_flag = 4.5

            self.marker_detection_start.data = 1
            if self.find_marker_reply == 5:
                self.marker_detection_start.data = 0
                self.task_stage_flag = 7


        # Go to destination position
        elif (self.task_stage_flag == 4.5):
            self.destination_coords[0] = self.package_drop_coordinates[self.current_package_index][0]
            self.destination_coords[1] = self.package_drop_coordinates[self.current_package_index][1]
            self.destination_coords[2] = self.package_drop_coordinates[self.current_package_index][2] + 7

            self.destination_publisher_func()
            self.marker_detection_start.data = 1
            self.path_planner_enable_flag = False
            self.gripper_flag = True
            if (self.package_window_check(self.package_drop_coordinates[self.current_package_index],
                                          [2.2, 2.2, float('inf')])):
                self.task_stage_flag = 5
            if self.find_marker_reply == 5:
                self.marker_detection_start.data = 0
                self.task_stage_flag = 7

        # Drone marker detection initialization
        elif (self.task_stage_flag == 5):
            self.marker_detection_start.data = 1  # starts the marker detection
            self.path_planner_enable_flag = False
            self.task_stage_flag = 6

        # Drone searches for marker and hovers above
        elif (self.task_stage_flag == 6):
            self.path_planner_enable_flag = False
            if self.find_marker_reply == 5:
                self.marker_detection_start.data = 0
                self.task_stage_flag = 7
         
        # Drone descends and drops the package
        elif (self.task_stage_flag == 7): 
            self.destination_coords[0] = 0
            self.destination_coords[1] = 0
            self.destination_coords[2] = self.package_drop_coordinates[self.current_package_index][2]
            self.destination_publisher_func()
            self.path_planner_enable_flag = False
            if(self.package_window_check(self.package_drop_coordinates[self.current_package_index],[float('inf'),float('inf'),0.45])):
                #self.marker_detection_start.data = 0
                self.gripper_flag = False
                self.path_planner_enable_flag = True
                self.task_stage_flag = 8

                for i in range(2):
                    self.braking_difference[i] = self.next_destination[i] - self.package_drop_coordinates[self.current_package_index][i]
                    if (self.braking_difference[i] >= 0):
                        self.braking_difference[i] -= 2/self.meter_conv[i]
                    else:
                        self.braking_difference[i] += 2/self.meter_conv[i]
                    self.braking_coordinates[i] = self.package_drop_coordinates[self.current_package_index][i] + self.braking_difference[i]


        # Go to next box position 2 metre before
        elif(self.task_stage_flag == 8):
            self.destination_coords[0] = self.braking_coordinates[0]
            self.destination_coords[1] = self.braking_coordinates[1]
            self.path_planner_enable_flag = True
            if(self.package_drop_coordinates[self.current_package_index][2]-self.next_destination[2] < 5.0):
                self.destination_coords[2] = self.package_altitude + 8.0
            else:
                self.destination_coords[2] = self.map_altitude + 2
            self.destination_publisher_func()
            self.gripper_flag = False
            if(self.package_window_check(self.braking_coordinates, [0.2,0.2, float('inf')])):
                self.task_stage_flag = 9

        # Go to warehouse coordinates specific
        elif(self.task_stage_flag == 9):
            self.destination_coords[0] = self.next_destination[0]
            self.destination_coords[1] = self.next_destination[1]
            self.path_planner_enable_flag =False
            self.destination_coords[2] = self.package_altitude + 8.0
            self.gripper_flag = False
            self.destination_publisher_func()
            if(self.package_window_check(self.next_destination, stg_9_error_window)):
                self.task_stage_flag = 1
                self.current_package_index += 1
                try:
                    self.next_destination = self.package_pickup_coordinates[self.current_package_index+1]
                except:
                    self.next_destination = self.drone_inital_position

        print('PACKAGE', self.current_package_index+1,'STAGE:', self.task_stage_flag)
        self.package_info_pub_func()
        # the detector
        self.marker_detection_start_pub.publish(self.marker_detection_start)

#####################################################################################################################################################

# main function
if __name__ == "__main__":

    drone_boi = control_node()
    r = rospy.Rate(20) # Frequency of 20 Hz
    drone_boi.initial_setup_func()
    while not rospy.is_shutdown():
        drone_boi.control_flow()
        r.sleep()

#########################################################################################################################

# END OF CONTROLLER NODE SCRIPT
