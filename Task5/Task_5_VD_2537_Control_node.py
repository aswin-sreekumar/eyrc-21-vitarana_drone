#! /usr/bin/env python


'''
TEAM ID: 2537
TEAM Name: JAGG
'''

'''
Node name : CONTROLLER NODE SCRIPT
Node desc : Controls the flow of the task and works with each stage of the drone navigation and funtions for TASK5
'''
'''
Algorithm:
Control node is the brain of the entire task and computess required coordinates on a broader level and publishes to mapping script
for further computation based on map obtained. Scheduler algorithm is performed to obtain sequenced manifest file and required coordinates
of pickup and drop are computed and stored to appropriate variables in required order.
Based on error windows and controller node algorithm, the coordinates are published to mapping script.

The mapping script subscribes to these coordinates and performs integrated A* based on conditions and mapping is enabled based on thresholds.
It uses an hybrid-continous scale mapping technique, compared to binary mapping we used in TASK4.
The entire arena is a grid divided into cells and each cell stores a floating point number based on key/
The sensor readings of drone are used to compute the safe flying distance of each cell and is updated continously.
During each navigation condition, the altitudes values in path cells are processed and a suitable flying height is assigned to the drone.
Integrated A* is applied during navigation segments and basic bug algorithm has been enabled in case of initial unmapped arena condition.
The bug algorithm gives a shot in raised altitude control before performing the required bug algorithm itself.
The position setpoint computed is published to position controller script

The position controller computed YPR values required based on PID and is further published to attitude controller.
The position controller script also houses a smaller segment of Marker detection to avoid clash between published setpoints.

The Marker detection kicks in at required conditions and ensures great accuracy maintaining speed efficiency.

The attitude controller script performs PID to maintain the required YPR values
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
        self.manifest_csv_loc = '/home/jk56/catkin_ws/src/vitarana_drone/scripts/sequenced_manifest.csv'
        # ASK
        # self.img_loc = '/home/aswinsreekumar/drone_ws/src/vitarana_drone/scripts/map_img.jpg'
        # self.manifest_csv_loc = '/home/aswinsreekumar/drone_ws/src/vitarana_drone/scripts/sequenced_manifest.csv'
        # Gree
        # self.img_loc = '/home/greesh/PycharmProjects/chumma/map_imp.jpg'
        # self.manifest_csv_loc = '/home/greesh/catkin_ws/src/vitarana_drone/scripts/sequenced_manifest.csv'
        # GK
        # self.img_loc = '/home/aswinsreekumar/drone_ws/src/vitarana_drone/scripts/map_img.jpg'
        # self.manifest_csv_loc = '/home/aswinsreekumar/drone_ws/src/vitarana_drone/scripts/manifest.csv'

        # Main control flow node VARIABLES
        self.destination_coords_object = Vector3()        # Object for destination publisher
        self.task_stage_flag = 0                            # Current stage
        self.drone_inital_position = [18.9998887906, 72.0002184402, 16.7579714806] # Inital position of drone
        self.destination_coords = [0.0, 0.0, 0.0]           # Published destination coordinates
        self.prev_destination_altitude = 16.7579714806
        self.meter_conv = [110693.227, 105078.267, 1] 		# Factor for degrees to meters conversion(lat,long,alt to x,y,z)
        #self.map_altitude = 0.0                             # Altitude of 2D map (threshold)
        self.drone_position = [18.9998887906, 72.0002184402, 16.7579714806]                # Current drone position in GPS coordinates
        self.controller_info_obj = String()                 # Object for publishing package info
        self.path_planner_enable_flag = False

        # Gripper Flag
        self.gripper_flag = False                           # Enable or disable gripper
        self.gripper_check_flag = False                     # State of gripper service
        self.grip_status = False

        # TASK 5 specific variables
        self.A1_coordinates = [18.9998102845, 72.000142461, 16.757981]             # A1 coordinates
        self.X1_coordinates = [18.9999367615, 72.000142461, 16.757981]             # X1 coordinates
        self.delivery_grid_coordinates = list()                                    # Coordinates of the delivery grid pad
        self.return_grid_coordinates = list()                                       # Coordinates of the return grid pad
        self.pickup_coordinate_list = list()
        self.drop_coordinate_list = list()                                     # Coordinates of the return grid pad
        self.current_package_index = 0
        self.package_cell_size = [1.5/110692.0702932625, 1.5/105292.0089353767]    # Cell size in radians
        self.action_req = list()                                        # Delivery or return in sequence
        self.total_package_number = 0

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

        self.task_timing = [[rospy.get_time(), 0]]



#################################################################################################################################################

# Subscriber and publishers

        rospy.Subscriber('/edrone/gps', NavSatFix, self.edrone_position)            # Position of drone
        rospy.Subscriber('/edrone/gps_velocity',Vector3Stamped,self.gps_velocity)
        rospy.Subscriber('/edrone/gripper_check',String,self.gripper_function)      # Gripper state check
        rospy.Subscriber('/find_marker_reply', Float32, self.find_marker_reply_fun)


        self.destination_coords_pub = rospy.Publisher('/destination_coordinates', Vector3, queue_size=1)	# Publishes destination to mapper
        self.controller_info_pub = rospy.Publisher('/controller_info', String, queue_size=1)                # Publishes ackage information
        self.marker_detection_start_pub = rospy.Publisher('/marker_detection_start', Float32,queue_size=1)  # marker_detection initiate publisher
        self.im_breaking_init_pub = rospy.Publisher('/start_breaking', Float32, queue_size=1)
        self.current_package_index_pub = rospy.Publisher('/current_package_index',String,queue_size = 1)


##################################################################################################################################################

# Call back functions and publishers

    def range_finder_bottom(self,bottom):

        val = bottom.ranges[0]

        # if val >= 0.4:
        #     if val <= 4:
        #         self.brake_pub_flag = False
        #         self.im_breaking_init.data = 10
        #     elif not self.brake_flag:
        #         self.im_breaking_init.data = 20
        #     else:
        #         self.brake_pub_flag = True
        #
        #     if not self.brake_pub_flag:
        #         self.im_breaking_init_pub.publish(self.im_breaking_init)

    # capturing the reply sent to the detector from the position control
    def find_marker_reply_fun(self, reply):
        self.find_marker_reply = reply.data

    #callback function for drone position
    def edrone_position(self, gps):
        self.drone_position[0] = gps.latitude
        self.drone_position[1] = gps.longitude
        self.drone_position[2] = gps.altitude

    def gps_velocity(self,gps):
        self.prev_drone_velocity_res = self.drone_velocity_res

        self.drone_velocity_res = math.sqrt((gps.vector.x)**2 + (gps.vector.y)**2 + (gps.vector.z)**2)


    # Destination publisher to position controller
    def destination_publisher_func(self):
        self.destination_coords_object.x = self.destination_coords[0]
        self.destination_coords_object.y = self.destination_coords[1]
        self.destination_coords_object.z = self.destination_coords[2]
        self.destination_coords_pub.publish(self.destination_coords_object)

    # Publish package information
    def package_info_pub_func(self):
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
        self.gripper_check_flag = data.data
        if(self.gripper_flag == True):
            if(data.data == "True"):
                self.gripper_service_client(True)
        else:
            self.gripper_service_client(False)

     # def braking_check(self):


    def max_roll_pitch_change(self,brake_flag,**kargs):

        self.brake_flag = brake_flag
        # self.prev_im_braking_init = self.im_breaking_init.data

        if self.brake_flag == True and self.prev_drone_velocity_res > self.drone_velocity_res: # and self.brake_pub_flag : #(self.package_window_check(self.braking_coordinates, [6, 6, float('inf')])):
            self.im_breaking_init.data = 40 # 30
        else:
            self.im_breaking_init.data = 25


        # if self.brake_pub_flag : #and self.prev_im_braking_init != self.im_breaking_init.data:
            # for breaking
        self.im_breaking_init_pub.publish(self.im_breaking_init)

    def find_breaking_coords(self,from_coords,to_coords):


        # to - self.drop_coordinate_list[self.current_package_index]
        # from - self.pickup_coordinate_list[self.current_package_index]
        # for i in range(2):
        #     self.braking_difference[i] = to_coords[i] - from_coords[i]
        #     if (self.braking_difference[i] >= 0):
        #         self.braking_difference[i] += (2 / self.meter_conv[i])
        #     else:
        #         self.braking_difference[i] += (2 / self.meter_conv[i])
        #
        #     self.braking_coordinates[i] = from_coords[i] + self.braking_difference[i]

        for i in range(2):
            if to_coords[i] > from_coords[i]:
                self.braking_coordinates[i] = to_coords[i] - (4 / self.meter_conv[i])
            else:
                self.braking_coordinates[i] = to_coords[i] + (4 / self.meter_conv[i])

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

    # Compute coordinates of delivery and return grid pad
    def grid_coordinates_compute(self):
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
        self.grid_coordinates_compute()
        self.file_read_func()


    # Control flow of the task
    def control_flow(self):

        # for i in range(18):
        #     print(i, self.pickup_coordinate_list[i], self.drop_coordinate_list[i], "\n")

        #to find to initial breaking coords


        # All deliveries/returns completed
        if (self.current_package_index == self.total_package_number):
            print("DELIVERIES AND RETURNS COMPLETED")
            return

        elif self.task_stage_flag == 0:
            self.find_breaking_coords(from_coords=self.drone_position, to_coords=self.pickup_coordinate_list[self.current_package_index])
            self.task_stage_flag = 1

            now = rospy.get_time()
            self.task_timing.append([now, now - self.task_timing[-1][0]])
            #print(self.task_timing)
            self.path_planner_enable_flag = False

        # Drone goes to above package
        elif(self.task_stage_flag == 1):

            self.destination_coords[0] = self.pickup_coordinate_list[self.current_package_index][0]
            self.destination_coords[1] = self.pickup_coordinate_list[self.current_package_index][1]
            self.destination_coords[2] = self.pickup_coordinate_list[self.current_package_index][2] + 0.5

            self.path_planner_enable_flag = True

            ####change
            # if self.current_package_index == 0:
            #     self.destination_coords[2] = self.pickup_coordinate_list[self.current_package_index][2] + 0.5
            #     # self.path_planner_enable_flag = False
            #
            # elif self.action_req[self.current_package_index] == "D" and self.action_req[self.current_package_index - 1] == "R":
            #     self.destination_coords[2] = self.pickup_coordinate_list[self.current_package_index][2] + 0.5
            #     # self.path_planner_enable_flag = False
            #
            # else:
            #     try:
            #         self.destination_coords[2] = max(self.pickup_coordinate_list[self.current_package_index][2] + 5, self.drop_coordinate_list[self.current_package_index-1][2] + 5)
            #     except IndexError:
            #         self.destination_coords[2] = self.pickup_coordinate_list[self.current_package_index][2] + 5
            #     self.path_planner_enable_flag = True

            ####----

            self.max_roll_pitch_change(brake_flag = True, old_alt = self.prev_destination_altitude, thresh = 1.5)   #invoking braking

            self.destination_publisher_func()
            self.gripper_flag = False

            if(self.package_window_check([self.pickup_coordinate_list[self.current_package_index][0],self.pickup_coordinate_list[self.current_package_index][1],0], [0.45,0.45,float('inf')])):
                self.max_roll_pitch_change(brake_flag = False)  # to stop braking
                self.path_planner_enable_flag = False
                self.task_stage_flag = 2


        # Descend and pick up package
        elif(self.task_stage_flag == 2):
            self.destination_coords[0] = self.pickup_coordinate_list[self.current_package_index][0]
            self.destination_coords[1] = self.pickup_coordinate_list[self.current_package_index][1]
            self.destination_coords[2] = self.pickup_coordinate_list[self.current_package_index][2]

            self.im_breaking_init.data = 25
            self.im_breaking_init_pub.publish(self.im_breaking_init)

            #print("YAY")

            # if self.package_window_check(self.destination_coords, [0.08, 0.08, float('inf')]):
            #     self.destination_coords[2] = self.pickup_coordinate_list[self.current_package_index][2]
            #     print("satisfying")

            self.gripper_flag = True
            self.path_planner_enable_flag = False
            self.destination_publisher_func()

            if(self.gripper_check_flag == "True"):
            # if True:
                # to find breaking coordinates
                self.find_breaking_coords(from_coords=self.pickup_coordinate_list[self.current_package_index], to_coords=self.drop_coordinate_list[self.current_package_index])
                self.task_stage_flag = 'z'   # changed from 3 to 2.5
                self.destination_coords[0] = self.braking_coordinates[0]  # braking needs to be here
                self.destination_coords[1] = self.braking_coordinates[1]
                self.destination_publisher_func()
                self.path_planner_enable_flag = True #changed from true to false
                self.prev_destination_altitude = self.destination_coords[2]


        elif (self.task_stage_flag == 'z'):
            self.destination_coords[0] = self.braking_coordinates[0]  # braking needs to be here
            self.destination_coords[1] = self.braking_coordinates[1]
            self.destination_coords[2] = self.pickup_coordinate_list[self.current_package_index][2] + 10

            self.destination_publisher_func()
            self.gripper_flag = True
            self.path_planner_enable_flag = True
            if not (self.package_window_check([self.pickup_coordinate_list[self.current_package_index][0], self.pickup_coordinate_list[self.current_package_index][1], self.destination_coords[2]], [1, 1, float('inf')])):
                self.path_planner_enable_flag = False
                self.gripper_flag = True
                # self.destination_coords[0] = self.pickup_coordinate_list[self.current_package_index][0]
                # self.destination_coords[1] = self.pickup_coordinate_list[self.current_package_index][1]
                # self.destination_publisher_func()
                self.task_stage_flag = 3

        # Go to destination position 1m before destination
        elif (self.task_stage_flag == 3):
            self.destination_coords[0] = self.braking_coordinates[0]  #braking needs to be here
            self.destination_coords[1] = self.braking_coordinates[1]
            self.destination_coords[2] = self.drop_coordinate_list[self.current_package_index][2]

            #self.destination_coords[0] = self.drop_coordinate_list[self.current_package_index][0]
            #self.destination_coords[1] = self.drop_coordinate_list[self.current_package_index][1]

            if (self.package_window_check(self.braking_coordinates, [6, 6, float('inf')]) and self.action_req[self.current_package_index] == 'D') :
                self.destination_coords[2] = self.drop_coordinate_list[self.current_package_index][2] + 5.5
            else:
                try:
                    self.destination_coords[2] = max(self.drop_coordinate_list[self.current_package_index][2] + 5, self.drop_coordinate_list[self.current_package_index - 1][2] + 5)
                except IndexError:
                    self.destination_coords[2] = self.drop_coordinate_list[self.current_package_index][2] + 5

            self.max_roll_pitch_change(brake_flag=True, old_alt = self.prev_destination_altitude,thresh = 1.5)  #invoking braking

            self.destination_publisher_func()
            self.path_planner_enable_flag = False
            self.gripper_flag = True

            #print(self.destination_coords)

            if self.action_req[self.current_package_index] == "D":
                if (self.package_window_check(self.braking_coordinates,[3, 3, float('inf')])):

                    self.max_roll_pitch_change(brake_flag=False)  #to stop braking
                    if(self.action_req[self.current_package_index] == 'D'):
                        self.task_stage_flag = 4
                    else:
                        self.task_stage_flag = 7  # changed from 7 to 8
            else:
                #print(" R window increased")
                if (self.package_window_check(self.braking_coordinates, [8, 8, float('inf')])):

                    self.max_roll_pitch_change(brake_flag=False)  # to stop braking
                    if (self.action_req[self.current_package_index] == 'D'):
                        self.task_stage_flag = 4
                    else:
                        self.task_stage_flag = 7  # changed from 7 to 8


        # Marker detection stage + slow move to destination coordinates
        # Drone marker detection initialization
        elif (self.task_stage_flag == 4):
            self.im_breaking_init.data = 15
            self.im_breaking_init_pub.publish(self.im_breaking_init)
            self.marker_detection_start.data = 1  # starts the marker detection
            self.path_planner_enable_flag = False
            self.task_stage_flag = 5

        # Drone searches for marker and hovers above
        elif (self.task_stage_flag == 5):
            self.im_breaking_init.data = 15
            self.im_breaking_init_pub.publish(self.im_breaking_init)
            self.path_planner_enable_flag = False
            if self.find_marker_reply == 5:
                self.marker_detection_start.data = 0
                self.task_stage_flag = 6

        # Drone descends and drops the package
        elif (self.task_stage_flag == 6):
            self.im_breaking_init.data = 15
            self.im_breaking_init_pub.publish(self.im_breaking_init)
            self.destination_coords[0] = 0
            self.destination_coords[1] = 0
            self.destination_coords[2] = self.drop_coordinate_list[self.current_package_index][2]
            self.destination_publisher_func()
            self.path_planner_enable_flag = False
            if (self.package_window_check(self.drop_coordinate_list[self.current_package_index],[float('inf'), float('inf'), 0.5])):
                # self.marker_detection_start.data = 0
                self.gripper_flag = False
                self.path_planner_enable_flag = False # changed to false from true
                if (self.gripper_check_flag == "False"):
                    #to find the next breaking coords
                    self.find_breaking_coords(from_coords=self.destination_coords,to_coords=self.pickup_coordinate_list[self.current_package_index+1])
                    self.task_stage_flag = 'y' #changed from 9 to 6.5
                    self.prev_destination_altitude = self.destination_coords[2]

        elif (self.task_stage_flag == 'y'):
            self.destination_coords[0] = 0
            self.destination_coords[1] = 0
            self.destination_coords[2] = self.drop_coordinate_list[self.current_package_index][2] + 5
            self.destination_publisher_func()
            self.gripper_flag = False
            self.path_planner_enable_flag = False
            if (self.package_window_check([0, 0, self.drop_coordinate_list[self.current_package_index][2] + 5],[float('inf'), float('inf'), 4])):
                self.path_planner_enable_flag = True
                self.task_stage_flag = 9


        # Go above required return grid
        elif (self.task_stage_flag == 7):
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
            if(self.package_window_check([self.drop_coordinate_list[self.current_package_index][0],self.drop_coordinate_list[self.current_package_index][1],self.drop_coordinate_list[self.current_package_index][2]], [0.5,0.5,0.5])):
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
            self.path_planner_enable_flag = True

        print('PACKAGE', self.current_package_index+1,'STAGE:', self.task_stage_flag)
        # print(self.destination_coords)
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
        # print(self.prev_destination_altitude, self.destination_coords[2])
        #print("tt: ", self.task_timing)

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
