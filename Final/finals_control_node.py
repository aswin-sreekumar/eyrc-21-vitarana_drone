#! /usr/bin/env python

'''
# Team ID:          VD_2537
# Theme:            Vitarana Drone
# Author List:      Jai Kesav, Aswin Sreekumar, Girish K, Greeshwar R S
# Filename:         Task_6_VD_2537_Control_node

# Functions:        __init__(), range_finder_bottom(), find_marker_reply_fun(), edrone_position(), gps_velocity(), destination_publisher_func()
                    package_info_pub_func(), file_read_func(), gripper_function(), max_roll_pitch_change(), find_breaking_coords(),
                    gripper_service_client(), package_window_check(), grid_coordinates_compute(), initial_setup_func(), control_flow()
                    im_breakin(), range_finder_bottom(), braking_pub_fun()

# Global variables: img_loc, manifest_csv_loc, task_stage_flag, drone_inital_position, destination_coords, prev_destination_altitude, meter_conv
                    drone_position,  path_planner_enable_flag, gripper_flag, gripper_check_flag, grip_status, A1_coordinates, X1_coordinates
                    delivery_grid_coordinates, return_grid_coordinates, pickup_coordinate_list, drop_coordinate_list, current_package_index
                    package_cell_size, action_req, total_package_number, marker_detection_start.data, find_marker_reply, braking_coordinate
                    braking_difference, yolo, brake_flag, brake_pub_flag, drone_velocity_res, prev_drone_velocity_res
                    bug_brake_check,
'''

'''
What this script does:
Control node is the brain of the entire task and computes required coordinates on a broader level and publishes to mapping script
for further computation based on map obtained. Scheduler algorithm is performed separately to obtain sequenced manifest file and required coordinates
of pickup and drop are computed and stored to appropriate variables in required order from the sequenced_manifest
Based on error windows and controller node algorithm, the coordinates are published to mapping script.
One entire delivery or return is broken into separate STAGES, which continously runs and gets updated based on drone position and
various other CHECK parameters. The same entire frame of stages are run for each delivery or return and is executed till all the packages
in the csv file are completed handling.
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

        # Local path of JAGG
        # JK
        self.img_loc = '/home/jk56/PycharmProjects/pythonProject/map_img.jpg'
        self.manifest_csv_loc = '/home/jk56/catkin_ws/src/vitarana_drone/scripts/manifest.csv'
        # ASK
        # self.img_loc = '/home/aswinsreekumar/drone_ws/src/vitarana_drone/scripts/map_img.jpg'
        # self.manifest_csv_loc = '/home/aswinsreekumar/drone_ws/src/vitarana_drone/scripts/sequenced_manifest_original.csv'
        # Gree
        # self.img_loc = '/home/greesh/PycharmProjects/chumma/map_imp.jpg'
        # self.manifest_csv_loc = '/home/greesh/catkin_ws/src/vitarana_drone/scripts/finals/sequenced_manifest_original.csv'
        # GK
        # self.img_loc = '/home/aswinsreekumar/drone_ws/src/vitarana_drone/scripts/map_img.jpg'
        # self.manifest_csv_loc = '/home/aswinsreekumar/drone_ws/src/vitarana_drone/scripts/manifest.csv'

        # Main control flow node VARIABLES
        self.destination_coords_object = Vector3()                                 # Object for destination publisher
        self.task_stage_flag = 0                                                   # Current stage
        self.drone_inital_position = [18.9998887906, 72.0002184402, 16.7579714806] # Inital position of drone
        self.destination_coords = [0.0, 0.0, 0.0]                                  # Published destination coordinates
        self.prev_destination_altitude = 16.7579714806                             # Altitude of previous destination
        self.meter_conv = [110693.227, 105078.267, 1] 		                       # Factor for degrees to meters conversion(lat,long,alt to x,y,z)
        self.drone_position = [18.9998887906, 72.0002184402, 16.7579714806]        # Current drone position in GPS coordinates
        self.controller_info_obj = String()                                        # Object for publishing package info
        self.path_planner_enable_flag = False                                      # Enable/ disable path planner algorithm

        # Gripper Flag
        self.gripper_flag = False                  # Enable or disable gripper
        self.gripper_check_flag = False            # State of gripper service
        self.grip_status = False                   # Status of gripper returned by rosservice

        # TASK 6 specific variables
        self.A1_coordinates = [18.9998102845, 72.000142461, 16.757981]             # A1 coordinates
        self.X1_coordinates = [18.9999367615, 72.000142461, 16.757981]             # X1 coordinates
        self.delivery_grid_coordinates = list()                                    # Coordinates of the delivery grid pad
        self.return_grid_coordinates = list()                                      # Coordinates of the return grid pad
        self.pickup_coordinate_list = list()                                       # Coordinates of the package pickup locations
        self.drop_coordinate_list = list()                                         # Coordinates of the package drop locations
        self.current_package_index = 0                                             # Current package handled
        self.package_cell_size = [1.5/110692.0702932625, 1.5/105292.0089353767]    # Cell size in radians (1.5 m sq)
        self.action_req = list()                                                   # Delivery or return in sequence
        self.total_package_number = 0                                              # Total number of packages in csv file

        # Marker detection
        self.marker_detection_start = Float32()  # Object for Marker detection 1-start detection
        self.marker_detection_start.data = 0     # Marker detection control flag
        self.find_marker_reply = 0               # Marker detection confirmation
        self.current_package_index_marker = String()

        self.braking_coordinates = [0.0, 0.0, 0.0]  # Braking algorithm setpoint
        self.braking_difference = [0.0, 0.0, 0.0]   # Difference in braking algorithm

        self.im_breaking_init = Float32()
        self.yolo = 0
        self.brake_flag = False
        self.brake_pub_flag = True
        self.drone_velocity_res = 0
        self.prev_drone_velocity_res = -1

        self.task_timing = []

        self.bug_brake_check = 0

        self.raise_alt = 10
        self.raise_cal_flag = True

        self.brake_updated_once = False


        self.yaw = Float32()
        self.drone_orientation_euler = [0, 0, 0]
        self.yaw_update_flag = True
        self.dynamic_transformation_var = Float32()
        self.dynamic_transformation_var.data = 0

        self.Dest_mapped = True
        self.Dest_mapped_var = Vector3()


#################################################################################################################################################

# Subscriber and publishers

        rospy.Subscriber('/edrone/gps', NavSatFix, self.edrone_position)            # Position of drone
        rospy.Subscriber('/edrone/gps_velocity',Vector3Stamped,self.gps_velocity)   # Current velocity of drone
        rospy.Subscriber('/edrone/gripper_check',String,self.gripper_function)      # Gripper state check
        rospy.Subscriber('/find_marker_reply', Float32, self.find_marker_reply_fun) # Acknowledgement and status of marker detector
        rospy.Subscriber('/start_breaking', Float32, self.im_breakin)
        rospy.Subscriber('/edrone/imu/data', Imu, self.imu_callback)
        rospy.Subscriber('/Dest_mapped_reply', Float32, self.Dest_mapped_reply_callback)


        self.destination_coords_pub = rospy.Publisher('/destination_coordinates', Vector3, queue_size=1)	# Publishes destination to mapper
        self.controller_info_pub = rospy.Publisher('/controller_info', String, queue_size=1)                # Publishes ackage information
        self.marker_detection_start_pub = rospy.Publisher('/marker_detection_start', Float32,queue_size=1)  # marker_detection initiate publisher
        self.im_breaking_init_pub = rospy.Publisher('/start_breaking', Float32, queue_size=1)               # Braking algorithm publisher
        self.current_package_index_pub = rospy.Publisher('/current_package_index',String,queue_size = 1)    # Current package index handled
        self.Yaw_pub = rospy.Publisher('/Decided_Yaw', Float32, queue_size=1)
        self.dynamic_transformation_pub = rospy.Publisher('/Dynamic_transform', Float32, queue_size = 1)
        self.Dest_mapped_pub = rospy.Publisher('/Dest_mapped', Vector3, queue_size = 1)


##################################################################################################################################################

# Call back functions and publishers

    def Dest_mapped_reply_callback(self, data):
        if data.data == 1:
            self.Dest_mapped = True
        elif data.data == 0:
            self.Dest_mapped = False

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
        (self.drone_orientation_euler[0], self.drone_orientation_euler[1],
         self.drone_orientation_euler[2]) = tf.transformations.euler_from_quaternion(
            [drone_orientation_quaternion[0], drone_orientation_quaternion[1], drone_orientation_quaternion[2],
             drone_orientation_quaternion[3]])
        self.drone_orientation_euler[0] *= 180 / math.pi
        self.drone_orientation_euler[1] *= 180 / math.pi
        self.drone_orientation_euler[2] *= 180 / math.pi

    def im_breakin(self,start):
        self.bug_brake_check = start.data

    def range_finder_bottom(self,bottom):

        val = bottom.ranges[0]

    # capturing the reply sent to the detector from the position control
    def find_marker_reply_fun(self, reply):
        '''
		Purpose:
		---
		Subscribes to the value from marker detection, indicating whether its running, on hold or it ended.

		Input Arguments:
		---
		`reply` :  ranges [ Float32 ]
		    Value of status of marker detection

		Returns:
		---
		None

		Example call:
		---
		Called automatically by Subscriber /find_marker_reply
		'''
        self.find_marker_reply = reply.data

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

    def gps_velocity(self,gps):
        '''
		Purpose:
		---
		Subscribes to the current drone velocity through Vector3Stamped message subscribed from /edrone/gps_velocity

		Input Arguments:
		---
		`gps` :  [ Vector3Stamped ]
			Holds drone velocity in all directions

		Returns:
		---
		None

		Example call:
		---
		Called automatically by Subscriber /edrone/gps_velocity
		'''
        self.prev_drone_velocity_res = self.drone_velocity_res
        self.drone_velocity_res = math.sqrt((gps.vector.x)**2 + (gps.vector.y)**2 ) #+ (gps.vector.z)**2)

    # Destination publisher to position controller
    def destination_publisher_func(self):
        '''
		Purpose:
		---
		Publishes the setpoint to height_mapping script. The setpoint is fixed by control node based on task stage and other paramters like braking algorithm.

		Input Arguments:
		---
		None

		Returns:
		---
		None

		Example call:
		---
		destination_publisher_func()
		'''

        self.destination_coords_object.x = self.destination_coords[0]
        self.destination_coords_object.y = self.destination_coords[1]
        self.destination_coords_object.z = self.destination_coords[2]
        self.destination_coords_pub.publish(self.destination_coords_object)

    # Publish package information
    def package_info_pub_func(self):
        '''
		Purpose:
		---
		Publishes the information comprising of current package index, stage and path planner flag to height_mapping script for navigation purposes.

		Input Arguments:
		---
		None

		Returns:
		---
		None

		Example call:
		---
		package_info_pub_func()
		'''

        list_a = ['A','B','C',"D",'E','F','G','H']
        if self.current_package_index < 10:
            self.controller_info_obj.data = str(self.current_package_index)
        elif self.current_package_index >= 10 and self.current_package_index < 18:

            self.controller_info_obj.data = list_a[self.current_package_index-10]
        else:
            self.controller_info_obj.data = 'I'

        self.controller_info_obj.data += str(self.task_stage_flag)
        if(self.path_planner_enable_flag == True):
            self.controller_info_obj.data += 'Y'
        else:
            self.controller_info_obj.data += 'N'
        self.controller_info_obj.data += self.action_req[self.current_package_index]
        self.controller_info_pub.publish(self.controller_info_obj)

##################################################################################################################################################

# Utility functions

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

        grid_index = [0,0]
        with open(self.manifest_csv_loc) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[0] == 'DELIVERY':
                    grid_index = [int(row[1][1])-1,int(ord(row[1][0])-65)]
                    #print(grid_index)
                    coords = row[2].split(';')
                    self.pickup_coordinate_list.append(self.delivery_grid_coordinates[grid_index[1]][grid_index[0]])
                    self.drop_coordinate_list.append([float(coords[0]),float(coords[1]),float(coords[2])])
                    self.action_req.append('D')
                    self.total_package_number += 1
                else:
                    grid_index = [int(row[2][1])-1,int(ord(row[2][0])-88)]
                    coords = row[1].split(';')
                    self.drop_coordinate_list.append(self.return_grid_coordinates[grid_index[1]][grid_index[0]])
                    self.pickup_coordinate_list.append([float(coords[0]),float(coords[1]),float(coords[2])])
                    self.action_req.append('R')
                    self.total_package_number += 1

    # Decision to pickup or drop the package will be sent here
    def gripper_function(self,data):
        '''
		Purpose:
		---
		Acitvates or deactivates gripper based on requirements, this is a subscriber from topic /edrone/gripper_check

		Input Arguments:
		---
		`data`: [ String ]
            Consists of whether package lies within the gripper and can be picked up.

		Returns:
		---
		None

		Example call:
		---
		Called automatically by subscriber through topic /edrone/gripper_check
		'''

        self.gripper_check_flag = data.data
        if(self.gripper_flag == True):
            if(data.data == "True"):
                self.gripper_service_client(True)
        else:
            self.gripper_service_client(False)

    def max_roll_pitch_change(self,brake_flag,**kargs):
        '''
		Purpose:
		---
		Publishes the maximum roll or pitch that is allowed for the drone to the position controller/attitude controller

		Input Arguments:
		---
		`brake_flag`: [ bool ]
            Enable or disable braking algorithm

        `kargs`: [ bool ]


		Returns:
		---
		None

		Example call:
		---
		self.max_roll_pitch_change(brake_flag = True, old_alt = self.prev_destination_altitude, thresh = 1.5)
		'''

        self.brake_flag = brake_flag
        # self.prev_im_braking_init = self.im_breaking_init.data

        if self.brake_flag == True and (self.package_window_check(self.braking_coordinates, [6, 6, float('inf')])): # and self.prev_drone_velocity_res > self.drone_velocity_res: # and self.brake_pub_flag : #(self.package_window_check(self.braking_coordinates, [6, 6, float('inf')])):
            self.im_breaking_init.data = 30 # 30
            self.brake_updated_once = True

        elif self.brake_updated_once == False:
                self.im_breaking_init.data = 20

        elif self.brake_flag == False:
            self.brake_updated_once = False
            self.im_breaking_init.data = 20


        # self.im_breaking_init.data = 15


            # for breaking
        self.braking_pub_fun()

    def braking_pub_fun(self):
        if '.5' not in str(self.bug_brake_check):
            self.im_breaking_init_pub.publish(self.im_breaking_init)
        else:
            pass

    def find_breaking_coords(self,from_coords,to_coords):
        '''
		Purpose:
		---
		Computes braking coordinates to be 4 m before the destination_coords in order to avoid overshoot

		Input Arguments:
		---
		`from_coords`: list [ Float32 ]
            Starting position of drone

        `to_coords`: list [ Float32 ]
            Ending destination position of drone

		Returns:
		---
		None

		Example call:
		---
		self.find_breaking_coords(from_coords=self.drone_position, to_coords=self.pickup_coordinate_list[self.current_package_index])
		'''

        lat_dist = abs(from_coords[0] - to_coords[0])*self.meter_conv[0]
        long_dist = abs(from_coords[1] - to_coords[1])*self.meter_conv[1]
        dist = math.sqrt(lat_dist**2 + long_dist**2)
        ratio = [lat_dist/dist, long_dist/dist]

        for i in range(2):
            if to_coords[i] > from_coords[i]:
                self.braking_coordinates[i] = to_coords[i] # - (4 / self.meter_conv[i])
            else:
                self.braking_coordinates[i] = to_coords[i] # + (4 / self.meter_conv[i])



    # Broadcasts the activation and deactivation commands to the service
    def gripper_service_client(self,activate_gripper):
        '''
		Purpose:
		---
		Enable or disable gripper of drone using ROSservice

		Input Arguments:
		---
		`activate_gripper`: [ bool ]
            Enable ot disable the gripper ROSservice

		Returns:
		---
		None

		Example call:
		---
        self.gripper_service_client(True)
		'''

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
        '''
		Purpose:
		---
		Check whether the drone is within the threshold window with respect to the passed coordinates.
        Returns TRUE if drone is inside the window, else returns False

		Input Arguments:
		---
		`coords`: list [ Float32 ]
            Destination or reference coordinates [lat, long, alt]

        `error`: list [ Float32 ]
            Error threshold in metres [lat, long, alt]

		Returns:
		---
		'1 or 0' : [ int ]

		Example call:
		---
        self.package_window_check([self.pickup_coordinate_list[self.current_package_index][0],self.pickup_coordinate_list[self.current_package_index][1],0], [0.45,0.45,float('inf')])
		'''

        if(abs(self.drone_position[0]-coords[0])<(error[0]/self.meter_conv[0])):
            if(abs(self.drone_position[1]-coords[1])<(error[1]/self.meter_conv[1])):
                if(abs(self.drone_position[2]-coords[2])<(error[2]/self.meter_conv[2])):
                    return 1
        else:
            return 0

    # Compute coordinates of delivery and return grid pad
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
		Called by grid_coordinates_compute() in initial_setup_func()
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

####################################################################################################################################################

# Algorithm

    # Inital data feed into the script (file read and package pickup compute) and compute map altitude
    def initial_setup_func(self):
        '''
		Purpose:
		---
		Initial setup of the node including computing all required pickup drop coordinates and reading the sequenced_manifest file

		Input Arguments:
		---
		None

		Returns:
		---
		None

		Example call:
		---
		Called using initial_setup_func() in __main__
		'''
        self.grid_coordinates_compute()
        self.file_read_func()

    def get_Yaw(self, to_coords):
        x_err = (to_coords[0] - self.drone_position[0])*self.meter_conv[0]
        y_err = - (to_coords[1] - self.drone_position[1])*self.meter_conv[1]
        dist = math.sqrt(x_err**2 + y_err**2)

        sin = y_err/dist
        cos = x_err/dist

        if sin > 0 and cos > 0:
            quad = 1
        elif sin > 0 and cos < 0:
            quad = 2
        elif sin < 0 and cos > 0:
            quad = 4
        elif sin < 0 and cos < 0:
            quad = 3
        elif sin == 0 or cos == 0:
            self.yaw.data = 0
            return

        print("quadrant : ", quad)

        mod_cos_angle = math.acos(abs(x_err)/dist)*180/math.pi
        print("mod_cos_angle : ", mod_cos_angle)

        if quad in [1, 3]:
            alpha = mod_cos_angle
        else:
            alpha = 90 - mod_cos_angle

        current_yaw = self.drone_orientation_euler[2]
        # if current_yaw > 90:
        #     current_yaw = current_yaw - 90
        # elif current_yaw < -90:
        #     current_yaw = current_yaw + 90

        yaw_possibilities = [alpha, 90 + alpha, -90 + alpha, -180 + alpha]
        min = float('inf')
        idx = -1
        for i in range(4):
            if abs(current_yaw - yaw_possibilities[i]) < min:
                idx = i
                min = abs(current_yaw - yaw_possibilities[i])

        self.yaw.data = yaw_possibilities[idx]



        # if quad in [1, 3]:
        #     if abs(alpha - ):
        #         self.yaw.data = mod_cos_angle
        #     else:
        #         self.yaw.data = -90 + mod_cos_angle

        # else:
            # if mod_cos_angle <= 45:
            #     self.yaw.data = - mod_cos_angle
            # else:
            # self.yaw.data = 90 - mod_cos_angle

        # angle = mod_cos_angle
        # if quad == 1:
        #     angle = mod_cos_angle
        # elif quad == 2:
        #     angle = 180 - mod_cos_angle
        # elif quad == 3:
        #     angle = -180 + mod_cos_angle
        # elif quad == 4:
        #     angle = - mod_cos_angle
        #
        # yaw_diff = abs(mod_cos_angle - self.drone_orientation_euler[2])




    # Control flow of the task
    def control_flow(self):
        '''
		Purpose:
		---
		Main algorithm of this node. Each delivery/return is split into different stages and is properly invoked and stage shifted based on conditions
        and parameters. Comprises of if elif ladder for the same purpose.

		Input Arguments:
		---
		None

		Returns:
		---
		None

		Example call:
		---
		Called by control_node() in __main__
		'''
        # All deliveries/returns completed

        print("timings : ", self.task_timing)
        print("Dest_mapped_flag : ", self.Dest_mapped)

        if (self.current_package_index == self.total_package_number):
            print("DELIVERIES AND RETURNS COMPLETED")
            return

        elif self.task_stage_flag == 0:
            self.find_breaking_coords(from_coords=self.drone_position, to_coords=self.pickup_coordinate_list[self.current_package_index])
            self.task_stage_flag = 1


            # if self.current_package_index == 0:
            #     self.task_timing.append([rospy.get_time(), 0])
            # now = rospy.get_time()
            # self.task_timing.append([now, now - self.task_timing[-1][0]])
            #
            self.destination_coords[0] = self.pickup_coordinate_list[self.current_package_index][0]
            self.destination_coords[1] = self.pickup_coordinate_list[self.current_package_index][1]
            self.destination_coords[2] = self.pickup_coordinate_list[self.current_package_index][2] + 0.5
            self.path_planner_enable_flag = True
            self.Yaw_pub.publish(self.yaw)



        # Drone goes to above package
        elif(self.task_stage_flag == 1):

            self.destination_coords[0] = self.pickup_coordinate_list[self.current_package_index][0]
            self.destination_coords[1] = self.pickup_coordinate_list[self.current_package_index][1]
            self.destination_coords[2] = self.pickup_coordinate_list[self.current_package_index][2] + 0.5
            self.destination_publisher_func()
            self.path_planner_enable_flag = True

            self.max_roll_pitch_change(brake_flag = True, old_alt = self.prev_destination_altitude, thresh = 1.5)   #invoking braking


            self.gripper_flag = False
            # try:
            #     if (self.package_window_check([self.drop_coordinate_list[self.current_package_index-1][0],
            #                                    self.drop_coordinate_list[self.current_package_index-1][1], 0],
            #                                   [7, 7, float('inf')])):
            #         self.im_breaking_init.data = 10  #20
            #         self.braking_pub_fun()
            #
            #
            # except IndexError:
            #     pass


            if(self.package_window_check([self.pickup_coordinate_list[self.current_package_index][0],self.pickup_coordinate_list[self.current_package_index][1],0], [0.45,0.45,float('inf')])):
                # self.max_roll_pitch_change(brake_flag = False)  # to stop braking
                self.path_planner_enable_flag = False
                self.task_stage_flag = 2

        # Descend and pick up package
        elif(self.task_stage_flag == 2):
            self.max_roll_pitch_change(brake_flag=False)  # to stop braking from stage flag 1 +++++++++++++++++++++++++++++++++++=
            self.destination_coords[0] = self.pickup_coordinate_list[self.current_package_index][0]
            self.destination_coords[1] = self.pickup_coordinate_list[self.current_package_index][1]
            self.destination_coords[2] = self.pickup_coordinate_list[self.current_package_index][2]

            self.im_breaking_init.data = 20 #25
            self.braking_pub_fun()

            self.Dest_mapped_var.x = self.drop_coordinate_list[self.current_package_index][0]
            self.Dest_mapped_var.y = self.drop_coordinate_list[self.current_package_index][1]
            self.Dest_mapped_var.z = self.drop_coordinate_list[self.current_package_index][2]
            self.Dest_mapped_pub.publish(self.Dest_mapped_var)



            self.raise_cal_flag = True
            self.gripper_flag = True
            self.path_planner_enable_flag = False
            self.destination_publisher_func()

            if(self.gripper_check_flag == "True"):
                # to find breaking coordinates
                self.find_breaking_coords(from_coords=self.pickup_coordinate_list[self.current_package_index], to_coords=self.drop_coordinate_list[self.current_package_index])
                self.task_stage_flag = 'z'   # changed from 3 to 2.5
                self.raise_cal_flag = True
                # self.destination_coords[0] = self.braking_coordinates[0]  # braking needs to be here
                # self.destination_coords[1] = self.braking_coordinates[1]
                self.destination_publisher_func()
                self.path_planner_enable_flag = False # True new
                self.prev_destination_altitude = self.destination_coords[2]

                if not self.Dest_mapped:
                    self.get_Yaw(self.drop_coordinate_list[self.current_package_index])


        elif (self.task_stage_flag == 'z'):
            # self.destination_coords[0] = self.braking_coordinates[0]  # braking needs to be here
            # self.destination_coords[1] = self.braking_coordinates[1]
            self.destination_coords[0] = float(self.drone_position[0])
            self.destination_coords[1] = float(self.drone_position[1])

            if self.raise_cal_flag == True:
                self.destination_coords[0] = float(self.drone_position[0])
                self.destination_coords[1] = float(self.drone_position[1])
                self.raise_cal_flag = False
                dest = list(self.drone_position)
                self.raise_alt = 0.15*(math.sqrt(((dest[0] - self.drop_coordinate_list[self.current_package_index][0]))**2 + (dest[1] - self.drop_coordinate_list[self.current_package_index][0])**2 ))
                if self.raise_alt < 3:
                    self.raise_alt = 3


            self.destination_coords[2] = self.pickup_coordinate_list[self.current_package_index][2] + self.raise_alt
            if abs(self.drone_position[2] - self.destination_coords[2]) < 4:
                self.Yaw_pub.publish(self.yaw)

            self.destination_publisher_func()
            self.gripper_flag = True
            self.path_planner_enable_flag = False # True new
            self.dynamic_transformation_var.data = 0
            self.dynamic_transformation_pub.publish(self.dynamic_transformation_var)

            if abs(self.drone_position[2] - self.destination_coords[2]) < 0.5 and abs(self.yaw.data - self.drone_orientation_euler[2]) < 3:
                self.destination_coords[0] = self.braking_coordinates[0]  # braking needs to be here
                self.destination_coords[1] = self.braking_coordinates[1]
                self.destination_publisher_func()
                self.path_planner_enable_flag = True  # changed from true to false
                self.gripper_flag = True
                self.raise_cal_flag = True   ######## new
                self.task_stage_flag = 3
                self.raise_cal_flag = True

        # Go to destination position 1m before destination
        elif (self.task_stage_flag == 3):
            self.destination_coords[0] = self.braking_coordinates[0]  #braking needs to be here
            self.destination_coords[1] = self.braking_coordinates[1]
            self.destination_coords[2] = self.drop_coordinate_list[self.current_package_index][2]

            if (self.package_window_check(self.braking_coordinates, [6, 6, float('inf')]) and self.action_req[self.current_package_index] == 'D') :
                self.destination_coords[2] = self.drop_coordinate_list[self.current_package_index][2] + 5.5
            else:
                try:
                    self.destination_coords[2] = max(self.drop_coordinate_list[self.current_package_index][2] + 5, self.drop_coordinate_list[self.current_package_index - 1][2] + 5)
                except IndexError:
                    self.destination_coords[2] = self.drop_coordinate_list[self.current_package_index][2] + 5

            self.max_roll_pitch_change(brake_flag = True, old_alt = self.prev_destination_altitude,thresh = 1.5)  #invoking braking

            self.destination_publisher_func()
            self.path_planner_enable_flag = True
            self.gripper_flag = True

            # if (self.package_window_check(self.braking_coordinates, [25, 25, float('inf')]) and self.action_req[self.current_package_index] == "D"):
            #     self.yaw.data = 0
            #     self.Yaw_pub.publish(self.yaw)
            #     self.dynamic_transformation_var.data = 1
            #     self.dynamic_transformation_pub.publish(self.dynamic_transformation_var)

            if (self.package_window_check(self.braking_coordinates, [8, 8, float('inf')])) and self.action_req[self.current_package_index] == "D":
                self.marker_detection_start.data = 1
                self.task_stage_flag = 4

            if self.action_req[self.current_package_index] == "D":
                if (self.package_window_check(self.braking_coordinates,[1, 1, float('inf')]) and abs(self.yaw.data - self.drone_orientation_euler[2]) < 3) and self.drone_velocity_res < 1:
                    # self.max_roll_pitch_change(brake_flag=False)  # to stop braking

                    if(self.action_req[self.current_package_index] == 'D'):
                        self.task_stage_flag = 4
                        self.destination_coords[0] = self.drop_coordinate_list[self.current_package_index][0]# braking needs to be here
                        self.destination_coords[1] = self.drop_coordinate_list[self.current_package_index][1]
                        self.destination_coords[2] = self.drop_coordinate_list[self.current_package_index][2]
                        self.destination_publisher_func()
                    else:
                        self.task_stage_flag = 7  # changed from 7 to 8

                    self.dynamic_transformation_var.data = 0
                    self.dynamic_transformation_pub.publish(self.dynamic_transformation_var)
            else:
                if (self.package_window_check(self.braking_coordinates, [8, 8, float('inf')])):

                    # self.max_roll_pitch_change(brake_flag=False)  # to stop braking
                    if (self.action_req[self.current_package_index] == 'D'):
                        self.task_stage_flag = 4
                    else:
                        self.task_stage_flag = 7  # changed from 7 to 8

                    self.dynamic_transformation_var.data = 0
                    self.dynamic_transformation_pub.publish(self.dynamic_transformation_var)


        # Marker detection stage + slow move to destination coordinates
        # Drone marker detection initialization
        elif (self.task_stage_flag == 4):
            # self.yaw.data = 0
            # self.Yaw_pub.publish(self.yaw)

            # self.dynamic_transformation_var.data = 1
            # self.dynamic_transformation_pub.publish(self.dynamic_transformation_var)

            if abs(self.drone_orientation_euler[2] - self.yaw.data) < 2:
                self.max_roll_pitch_change(brake_flag=False)  # to stop braking
                self.im_breaking_init.data =20 #25
                self.braking_pub_fun()
                self.marker_detection_start.data = 1  # starts the marker detection
                self.path_planner_enable_flag = False
                self.destination_coords[0] = self.drop_coordinate_list[self.current_package_index][0]  # braking needs to be here
                self.destination_coords[1] = self.drop_coordinate_list[self.current_package_index][1]
                self.destination_coords[2] = self.drop_coordinate_list[self.current_package_index][2]
                self.destination_publisher_func()
                self.task_stage_flag = 5

                self.dynamic_transformation_var.data = 0
                self.dynamic_transformation_pub.publish(self.dynamic_transformation_var)


        # Drone searches for marker and hovers above
        elif (self.task_stage_flag == 5):
            self.im_breaking_init.data = 20 #25
            self.braking_pub_fun()
            self.path_planner_enable_flag = False
            self.destination_coords[0] = self.drop_coordinate_list[self.current_package_index][0]  # braking needs to be here
            self.destination_coords[1] = self.drop_coordinate_list[self.current_package_index][1]
            self.destination_coords[2] = self.drop_coordinate_list[self.current_package_index][2]
            self.destination_publisher_func()

            self.dynamic_transformation_var.data = 0
            self.dynamic_transformation_pub.publish(self.dynamic_transformation_var)
            self.Dest_mapped = True

            if self.find_marker_reply == 5:
                self.marker_detection_start.data = 3
                self.task_stage_flag = 6


        # Drone descends and drops the package
        elif (self.task_stage_flag == 6):
            self.im_breaking_init.data = 20 #25
            self.braking_pub_fun()
            self.destination_coords[0] = 0
            self.destination_coords[1] = 0
            self.destination_coords[2] = self.drop_coordinate_list[self.current_package_index][2]
            self.destination_publisher_func()

            self.Dest_mapped_var.x = self.pickup_coordinate_list[self.current_package_index + 1][0]
            self.Dest_mapped_var.y = self.pickup_coordinate_list[self.current_package_index + 1][1]
            self.Dest_mapped_var.z = self.pickup_coordinate_list[self.current_package_index + 1][2]
            self.Dest_mapped_pub.publish(self.Dest_mapped_var)

            if not self.Dest_mapped:
                self.get_Yaw(self.pickup_coordinate_list[self.current_package_index + 1])

            self.path_planner_enable_flag = False
            if (self.package_window_check(self.drop_coordinate_list[self.current_package_index],[float('inf'), float('inf'), 0.5])):
                # self.marker_detection_start.data = 0
                self.gripper_flag = False
                self.path_planner_enable_flag = False # changed to false from true
                if (self.gripper_check_flag == "False"):
                    self.marker_detection_start.data = 0
                    #to find the next breaking coords
                    self.find_breaking_coords(from_coords=self.destination_coords,to_coords=self.pickup_coordinate_list[self.current_package_index+1])
                    self.task_stage_flag = 'y' #changed from 9 to 6.5
                    self.raise_cal_flag = True
                    self.prev_destination_altitude = self.destination_coords[2]



            # Publish the next destination (self.pickup_coordinate_list[self.current_package_index + 1])



        elif (self.task_stage_flag == 'y'):
            self.destination_coords[0] = 0
            self.destination_coords[1] = 0

            if self.raise_cal_flag == True:
                self.raise_cal_flag = False
                dest = list(self.drone_position)
                self.raise_alt = 8000*(math.sqrt(((dest[0] - self.pickup_coordinate_list[self.current_package_index+1][0]))**2 + (dest[1] -  self.pickup_coordinate_list[self.current_package_index+1][1])**2 ))
                if self.action_req[self.current_package_index] == "R":
                    self.raise_alt = 4
                elif self.raise_alt < 8:
                    self.raise_alt = 8

            self.Yaw_pub.publish(self.yaw)

            height = 0
            self.destination_publisher_func()
            self.gripper_flag = False
            self.path_planner_enable_flag = False
            if self.action_req[self.current_package_index] == "D":
                height = self.destination_coords[2] = self.drop_coordinate_list[self.current_package_index][2] + self.raise_alt  + 2 #10
                h_up = 2

            else:
                height = self.destination_coords[2] = self.drop_coordinate_list[self.current_package_index][2] + self.raise_alt  + 2 #3
                h_up = 2

            if (self.package_window_check([0, 0, height],[float('inf'), float('inf'), h_up])) and abs(self.yaw.data - self.drone_orientation_euler[2]) < 3:
                self.raise_cal_flag = True  ######## new
                self.path_planner_enable_flag = True
                self.task_stage_flag = 9


        # Go above required return grid
        elif (self.task_stage_flag == 7):
            self.max_roll_pitch_change(brake_flag=False)  # to stop braking
            self.destination_coords[0] = self.drop_coordinate_list[self.current_package_index][0]
            self.destination_coords[1] = self.drop_coordinate_list[self.current_package_index][1]
            self.destination_coords[2] = self.drop_coordinate_list[self.current_package_index][2] + 0.4
            self.destination_publisher_func()
            self.gripper_flag = True
            self.path_planner_enable_flag = False
            if(self.package_window_check([self.drop_coordinate_list[self.current_package_index][0],self.drop_coordinate_list[self.current_package_index][1],0], [0.5,0.5,float('inf')])):
                self.task_stage_flag = 8

        # Descend and drop package
        elif(self.task_stage_flag == 8):
            self.destination_coords[0] = self.drop_coordinate_list[self.current_package_index][0]
            self.destination_coords[1] = self.drop_coordinate_list[self.current_package_index][1]
            self.destination_coords[2] = self.drop_coordinate_list[self.current_package_index][2]
            self.gripper_flag = True
            self.path_planner_enable_flag = False
            self.destination_publisher_func()
            if(self.package_window_check([self.drop_coordinate_list[self.current_package_index][0],self.drop_coordinate_list[self.current_package_index][1],self.drop_coordinate_list[self.current_package_index][2]], [0.4,0.4,0.5])):
                self.gripper_flag = False
                if self.gripper_check_flag == "False":
                    #to find the next breaking coords
                    self.find_breaking_coords(from_coords=self.drop_coordinate_list[self.current_package_index],to_coords=self.pickup_coordinate_list[self.current_package_index+1])
                    self.task_stage_flag = 'y' #changed from 9 to 6.5

        # Reseting all variables for next delivery / return
        elif(self.task_stage_flag == 9):

            self.task_stage_flag = 0
            self.current_package_index += 1
            self.prev_destination_altitude = self.destination_coords[2]  ##
            self.path_planner_enable_flag = False

        print('PACKAGE', self.current_package_index+1,'STAGE:', self.task_stage_flag)
        #print('Timing: ', self.task_timing)
        self.package_info_pub_func()
        # the detector
        self.marker_detection_start_pub.publish(self.marker_detection_start)

        #pub cur package index to marker
        if self.current_package_index < 10:
            self.current_package_index_marker.data = str(self.current_package_index)+self.action_req[self.current_package_index]
        elif self.current_package_index < 17:
            list_a = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            self.current_package_index_marker.data = str(list_a[self.current_package_index-10]) + self.action_req[self.current_package_index]
        else:
            self.current_package_index_marker.data = 'I' + self.action_req[self.current_package_index]

        self.current_package_index_pub.publish(self.current_package_index_marker)


#####################################################################################################################################################

# main function
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

    drone_boi = control_node()
    r = rospy.Rate(20) # Frequency of 20 Hz
    drone_boi.initial_setup_func()
    while not rospy.is_shutdown():
        drone_boi.control_flow()
        print("raise_height :",drone_boi.raise_alt)
        r.sleep()

#########################################################################################################################

# END OF CONTROLLER NODE SCRIPT
