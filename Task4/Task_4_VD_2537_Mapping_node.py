#! /usr/bin/env python

#TEAM NAME- JAGG
#TEAM ID- 2537

#####################################################################################################################
'''
Algorithm for Task 4:
The required coordinates of package pickup and drop coordinates are computed. Binary grid
mapping is implemented with region of interest around the package drop building coordinates.
The drone's sensors are used to map the surroundings above a specific height.
Using the map data, Modified A* algorithm is implemented along with smoothing functions to
improvise set points and make it a realistic navigation giving an efficient algorithm of path planning.
All packages undergo the same stages of delivery. Braking algorithm is integrated into the control
flow algorithm in order to prevent the drone from overshooting. Upon reaching the building, the
drone searches for marker using Marker Detection and lands at the required place with accuracy.
Upon successful delivery of a package, the drone proceeds to the warehouse grid to deliver the
next package. The drone goes back to the initial position after completion of deliveries, before
indicating the end of task.
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
import numpy as np


###################################################################################################################

# Class of mapping node
class mapping():

    def __init__(self):

        # node definition and name
    	rospy.init_node('mapping_pathplanner', anonymous=True)

        # variables used for mapping part

        #self.img_loc = '/home/jk56/PycharmProjects/pythonProject/map_img.jpg'
        self.manifest_csv_loc = '/home/jk56/catkin_ws/src/vitarana_drone/scripts/manifest.csv'

        # Basic variables for initial setup
        self.first_package_coordinates = [18.9999864489, 71.9999430161, 8.44099749139]  # A1 coordinates
        self.drone_inital_position = [19, 72, 8.44]                 # Inital position of drone
        self.position_setpoint = [0.0,0.0,0.0]                      # Setpoint of drone
        self.setpoint_pub_object = Vector3()                      # Object for setpoints publisher
        self.package_cell_size = [1.5/110692.0702932625, 1.5/105292.0089353767]         # Cell size in radians
        self.package_pickup_coordinates = list()                    # Package pick up coordinates
        self.package_drop_coordinates = list()                      # Package drop coordinates
        self.range_bottom_data = 0.0
        self.map_altitude = 0.0
        self.sensor_validity = [True, True, True, True]

        # Grid map generation
        self.grid_length = 0.0                                      # No of cells in length
        self.grid_width = 0.0                                       # No of cells in width
        self.grid_cell_length = 1.50                                # Cell side length
        self.boundaries_computed = [[0.0,0.0],[0.0,0.0]]            # Boundaries obtained [[LEFT-TOP],[RIGHT-BOTTOM]]
        self.boundaries_final = [[0.0,0.0],[0.0,0.0]]               # Boundaries approx [[LEFT-TOP],[RIGHT-BOTTOM]]

        # Grid map cell data
        self.map_list = list()                                      # Contains data of map, 'x'-unknown; '0'-empty; '1'-filled
        self.current_cell_index = [0, 0]                            # Cell lat,long index
        self.drone_cell_index = [0, 0]                              # Cell of drone position GPS
        self.package_drop_cells = list()                            # Cell of package drop coordinates
        # Map image and related variables
        self.map_img_len = 1
        self.map_img_width = 1
        self.map_img = np.zeros((self.map_img_len*3, self.map_img_width*3, 3), np.uint8)         # Image of the map
        self.map_img_1 = np.zeros((self.map_img_len * 3, self.map_img_width * 3, 3), np.uint8)  # Image of the map
        self.img_scale = 15                                         # Scaling the map

        # Mapping drone DATA
        self.mapping_enable_flag = True
        self.range_top_dist = [float('inf'), float('inf'), float('inf'), float('inf'),float('inf')]  # [front,right,back,left,top]
        self.drone_position = [0.0, 0.0, 0.0]
        self.prev_drone_position = [0.0, 0.0, 0.0]
        self.meter_conv = [110692.0702932625, 105292.0089353767, 1] 				#Factor for degrees to meters conversion(lat,long,alt to x,y,z)
        self.map_sensor_data = [0.0, 0.0, 0.0, 0.0]                 # Sensor data [Front, Right, Back, Left]
        self.prev_map_sensor_data = [float('inf'), float('inf'), float('inf'), float('inf'),float('inf')]            # Previous map sensor data [Front, Right, Back, Left]
        self.obstacle_cell_constant = [4,4]                           # Minimum number of cells in building

        # Path planning
        self.destination_coords = [0.0, 0.0, 0.0]              # Destination coordinates of drone
        self.path_planner_flag = False
        self.path_planning_complete = False
        self.controller_info = '000'
        self.prev_controller_info = '000'

        # Marker detection
        self.enable_setpoint_pub_flag = 1  # enabling setpoint pub initially true

        #Variables used for path planning part
        self.start_node = []
        self.target_node = []
        self.grid = self.map_list
        self.grid_size = [self.grid_width - 1, self.grid_length - 1]
        self.open_list = []     # Nodes that are yet to be evaluated
        self.closed_list = []   # Nodes that have already been evaluated

        self.g_cost_list = []   # For storing g-cost of each node
        self.h_cost_list = []   # For storing h-cost of each node
        self.f_cost_list = []   # For storing f-cost of each node
        self.parent_list = {}   # For storing the parent of each node
        self.current = []  # For node in open with lowest f-cost

        self.found_path = []  # Path found from A*
        self.smoothened_path = []  # Final path obtaining after smoothening

        self.a_star_index = 0
        self.a_star_flag = False

        self.a_star_nav_completed_flag = True           # Indicates whether A* path destination reached
        self.a_star_comp_completed_flag = True
        self.a_star_path_index = 0                      # Current index of A* path list
        self.a_star_coords = [0.0, 0.0, 0.0]            # A* coordinates published to position_controller

        ###################################################################################################################

# Subscribers and Publishers

        self.setpoint_publisher = rospy.Publisher('/set_setpoint', Vector3, queue_size = 1)   # Setpoints publisher

        rospy.Subscriber('/edrone/range_finder_top', LaserScan, self.range_top)                 # Sensor data
        rospy.Subscriber('/edrone/gps', NavSatFix, self.edrone_position)                        # Current GPS of drone
        rospy.Subscriber('/destination_coordinates', Vector3, self.handle_destination_coords) # Destination from control node
        rospy.Subscriber('/controller_info', String, self.controller_info_func)  # Package info callback
        rospy.Subscriber('/enable_setpoint_pub_mapping_script', Float32, self.enable_setpoint_pub)

###################################################################################################################

# Callback and publisher functions

    def enable_setpoint_pub(self,enable):
        self.enable_setpoint_pub_flag = enable.data

    # Callback for package information
    def controller_info_func(self, data):

        self.controller_info = str(data.data)

        if self.controller_info[2] == "Y":
            self.a_star_flag = True
            if self.controller_info[0:2] != self.prev_controller_info[0:2]:
                self.a_star_comp_completed_flag = False
                self.a_star_nav_completed_flag = False
                #print(self.controller_info)
        else:
            self.a_star_flag = False
            self.a_star_nav_completed_flag = True

        self.prev_controller_info = data.data

    # Publisher for setpoints to position controller
    def setpoint_pub_func(self):
        self.setpoint_pub_object.x = self.position_setpoint[0]
        self.setpoint_pub_object.y = self.position_setpoint[1]
        self.setpoint_pub_object.z = self.position_setpoint[2]
        self.setpoint_publisher.publish(self.setpoint_pub_object)
        #

    # Sensor reading callback
    def range_top(self, range_top_data):

        self.prev_map_sensor_data = copy.copy(self.map_sensor_data)
        self.range_top_dist = range_top_data.ranges

        for i in range(4):
            if(self.range_top_dist[i] == float('inf')):
                self.map_sensor_data[i] = 25.0
            elif (self.range_top_dist[i]<0.5):
                self.map_sensor_data[i] = self.prev_map_sensor_data[i]
            else:
                self.map_sensor_data[i] = self.range_top_dist[i]

    # Drone GPS callback function
    def edrone_position(self, gps):
        self.prev_drone_position = copy.copy(self.drone_position)
        self.drone_position[0] = gps.latitude
        self.drone_position[1] = gps.longitude
        self.drone_position[2] = gps.altitude
        self.drone_cell_index = self.find_cell_index([self.drone_position[0], self.drone_position[1]])

    # Store incoming destination data
    def handle_destination_coords(self, coords):
        self.destination_coords[0] = coords.x
        self.destination_coords[1] = coords.y
        self.destination_coords[2] = coords.z


######################################################################################################################

# MISC functions

    # Extract data from csv file
    def file_read_func(self):
        cell_data = [0, 0]
        with open(self.manifest_csv_loc) as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                self.package_drop_coordinates.append([float(row[1]), float(row[2]), float(row[3])])

    # Checking if drone is inside the window for picking packages wrt coords and error window
    def package_window_check(self, coords, error):
        if(abs(self.drone_position[0]-coords[0])<(error[0]/self.meter_conv[0])):
            if(abs(self.drone_position[1]-coords[1])<(error[1]/self.meter_conv[1])):
                if(abs(self.drone_position[2]-coords[2])<(error[2]/self.meter_conv[2])):
                    return 1
        else:
            return 0


    # Updating the map image's individual cells
    def map_img_cell_update(self, cell, sign):

        if sign == '1':
            color = (255, 255, 255)
        elif sign == '0':
            color = (0, 0, 0)
        elif sign == 'x':
            color = (0, 0, 255)
        elif sign == 'b':
            color = (255, 0, 0)
        elif sign == 'g':
            color = (0, 255, 0)
        elif sign == 't':
            color = (125,125,125)
        elif sign == 'Spath':
            color = (255,255,0)
        elif sign == 'Fpath':
            color = (203,192,255)

        x, y = cell
        self.map_img[self.img_scale * x: self.img_scale * x + self.img_scale,self.img_scale * y: self.img_scale * y + self.img_scale] = color

    # Updating the map using map_list
    def map_img_list_update(self):
        for i in range(int(self.grid_width)):
            for j in range(int(self.grid_length)):
                self.map_img_cell_update([i, j], str(self.map_list[i][j]))

        cv2.imwrite(self.img_loc, self.map_img)


    # Drawing a grid on the map and other stuff
    def map_img_initialize(self):

        self.map_img_len = int(self.grid_length) * self.img_scale
        self.map_img_width = int(self.grid_width) * self.img_scale
        self.map_img = np.zeros((self.map_img_width, self.map_img_len, 3), np.uint8)

        for i in range(0, int(self.grid_length)):
            cv2.line(self.map_img, (self.img_scale * i, 0), (self.img_scale * i, self.map_img_width), (255, 255, 255), 1)
        for j in range(0, int(self.grid_width)):
            cv2.line(self.map_img, (0, self.img_scale * j), (self.map_img_len, self.img_scale * j), (255, 255, 255), 1)

        self.map_img_list_update()
        cv2.imwrite(self.img_loc, self.map_img)

    def obs_validity(self):
        # sensor data is [front, right, back, left]

        lat_diff = self.drone_position[0] - self.prev_drone_position[0]
        long_diff = self.drone_position[1] - self.prev_drone_position[1]
        sensor_val_change = [0,0,0,0]
        for i in range(0,4):
            sensor_val_change[i] = self.map_sensor_data[i] - self.prev_map_sensor_data[i]
            if self.map_sensor_data[i] == self.prev_map_sensor_data[i]:
                sensor_val_change[i] = 25.1

        self.sensor_validity = [0, 0, 0, 0]
        ratio = 0.8

        if (abs(lat_diff*self.meter_conv[0]/sensor_val_change[1]) > ratio and sensor_val_change[1]*lat_diff < 0) or self.map_sensor_data[1] == 25:     # right sensor
            self.sensor_validity[1] = True
        else:
            self.sensor_validity[1] = False

        if (abs(lat_diff*self.meter_conv[0]/sensor_val_change[3]) > ratio and sensor_val_change[3]*lat_diff > 0) or self.map_sensor_data[3] == 25:     # left sensor
            self.sensor_validity[3] = True
        else:
            self.sensor_validity[3] = False

        if (abs(long_diff*self.meter_conv[1]/sensor_val_change[2]) > ratio and sensor_val_change[2]*long_diff < 0) or self.map_sensor_data[2] == 25:     # back sensor
            self.sensor_validity[2] = True
        else:
            self.sensor_validity[2] = False

        if (abs(long_diff*self.meter_conv[1]/sensor_val_change[0]) > ratio and sensor_val_change[0]*long_diff > 0) or self.map_sensor_data[0] == 25 :     # back sensor
            self.sensor_validity[0] = True
        else:
            self.sensor_validity[0] = False



##################################################################################################################

# Algorithm

    # To compute the boundaries of grid map using drone pos, package pick and drop positions
    def boundaries_compute_func(self):
        min_latitude = 100.0
        max_latitude = 0.0
        min_longitude = 100.0
        max_longitude = 0.0

        for coordinate in self.package_pickup_coordinates:
            if(coordinate[0]>max_latitude):
                max_latitude = coordinate[0]
            if(coordinate[0]<min_latitude):
                min_latitude = coordinate[0]
            if(coordinate[1]>max_longitude):
                max_longitude = coordinate[1]
            if(coordinate[1]<min_longitude):
                min_longitude = coordinate[1]

        for coordinate in self.package_drop_coordinates:
            if(coordinate[0]>max_latitude):
                max_latitude = coordinate[0]
            if(coordinate[0]<min_latitude):
                min_latitude = coordinate[0]
            if(coordinate[1]>max_longitude):
                max_longitude = coordinate[1]
            if(coordinate[1]<min_longitude):
                min_longitude = coordinate[1]

        if(min_latitude>self.drone_inital_position[0]):
            min_latitude = self.drone_inital_position[0]
        if(max_latitude<self.drone_inital_position[0]):
            max_latitude = self.drone_inital_position[0]
        if(min_longitude>self.drone_inital_position[1]):
            min_longitude = self.drone_inital_position[1]
        if(max_longitude<self.drone_inital_position[1]):
            max_longitude = self.drone_inital_position[1]

        self.boundaries_computed = [[min_latitude, min_longitude], [max_latitude, max_longitude]]

        self.boundaries_final[0] = [self.boundaries_computed[0][0]-(4*self.package_cell_size[0]), self.boundaries_computed[0][1]-(4*self.package_cell_size[0])]
        self.boundaries_final[1] = [self.boundaries_computed[1][0]+(4*self.package_cell_size[1]), self.boundaries_computed[1][1]+(4*self.package_cell_size[1])]

        max_latitude = (int((self.boundaries_final[1][0]-self.boundaries_final[0][0])/self.package_cell_size[0])*self.package_cell_size[0])+self.boundaries_final[0][0]
        max_longitude = (int((self.boundaries_final[1][1]-self.boundaries_final[0][1])/self.package_cell_size[1])*self.package_cell_size[1])+self.boundaries_final[0][1]

        self.boundaries_final[1] = [max_latitude, max_longitude]

        self.grid_length = math.ceil((self.boundaries_final[1][0]-self.boundaries_final[0][0])/self.package_cell_size[0])
        self.grid_width = math.ceil((self.boundaries_final[1][1]-self.boundaries_final[0][1])/self.package_cell_size[1])
        print("grid_length", self.grid_length)
        print("grid_width", self.grid_width)

    # Finding index values of cell given a coordinates into self.position_coordinates
    def find_cell_index(self, position_coordinates):
        current_index = [0, 0]
        try:
            current_index[1] = int(math.floor(((position_coordinates[0] - self.boundaries_final[0][0]) / self.package_cell_size[0]) - 0))
            current_index[0] = int(math.floor(((position_coordinates[1] - self.boundaries_final[0][1]) / self.package_cell_size[1]) - 0))

            if (current_index[1] > (self.grid_length - 1)):
                current_index[1] = int(self.grid_length - 1)
            elif (current_index[1] < 0):
                current_index[1] = int(0)
            if (current_index[0] > (self.grid_width - 1)):
                current_index[0] = int(self.grid_width - 1)
            elif (current_index[0] < 0):
                current_index[0] = int(0)

        except:
            pass
        return current_index

    # Input for drone's setpoint coordinates
    def setpoint_function(self):
        self.position_setpoint[0] = input("LATITUDE: ")
        self.position_setpoint[1] = input("LONGITUDE: ")
        self.position_setpoint[2] = input("ALTITUDE : ")
        self.setpoint_pub_func()

    # Initial setup of drone in TASK4
    def initial_setup_func(self):
        map_row_temp_list = list()
        self.package_pickup_coordinates.append(self.first_package_coordinates)
        self.package_pickup_coordinates.append([self.first_package_coordinates[0]+(2*self.package_cell_size[0]),self.first_package_coordinates[1],self.first_package_coordinates[2]])
        self.package_pickup_coordinates.append([self.first_package_coordinates[0]+self.package_cell_size[0],self.first_package_coordinates[1]+self.package_cell_size[1],self.first_package_coordinates[2]])
        self.file_read_func()
        self.boundaries_compute_func()

        row = []
        for _ in range(int(self.grid_length)):
            row.append('x')
        for _ in range(int(self.grid_width)):
            self.map_list.append(list(row))
        del row

        cell_data = [0,0]
        for i in self.package_drop_coordinates:
            cell_data = self.find_cell_index([i[0], i[1]])
            self.package_drop_cells.append(cell_data)

        #print(self.package_drop_cells)

        for i in self.package_drop_cells:
            self.map_list[i[0]][i[1]] = 'g'

        for i in self.package_drop_coordinates:
            if(self.map_altitude < i[2]):
                self.map_altitude = i[2]
        self.map_altitude += 2


        #self.map_img_initialize()


    # Map list updation, mode= 1-front/back, mode=0-left/right
    def map_update_func(self, new_data, mode):

        if (self.map_list[self.drone_cell_index[0]][self.drone_cell_index[1]] == '0' or self.map_list[self.drone_cell_index[0]][self.drone_cell_index[1]] == 'x'):
            self.map_list[self.drone_cell_index[0]][self.drone_cell_index[1]] = 'b'

        if (self.mapping_enable_flag == False):
            return

        if(mode == 0):
            if(new_data[1]>=self.drone_cell_index[1]):
                for i in range(int(self.drone_cell_index[1]), int(new_data[1])+1):
                    if(self.map_list[self.drone_cell_index[0]][i] == 'x' or self.map_list[self.drone_cell_index[0]][i] == 't'):
                        self.map_list[self.drone_cell_index[0]][i] = '0'

                for i in range(int(new_data[1])-1, int(new_data[1])+self.obstacle_cell_constant[0]):
                    if(i<self.grid_length and self.map_sensor_data[1]<25.0):
                        self.map_list[self.drone_cell_index[0]][i] = '1'

                for i in range(int(new_data[1])-2,int(new_data[1])+self.obstacle_cell_constant[0]):
                    if(i<self.grid_length and self.map_sensor_data[1]<25.0):
                        for j in range(self.drone_cell_index[0]-self.obstacle_cell_constant[1],self.drone_cell_index[0]+self.obstacle_cell_constant[1]+1):
                            try:
                                if(self.map_list[j][i] == 'x'):
                                    self.map_list[j][i] = '1'
                            except:
                                pass

            else:
                for i in range(int(new_data[1]), int(self.drone_cell_index[1])+1):
                    if(self.map_list[self.drone_cell_index[0]][i] == 'x'):
                        self.map_list[self.drone_cell_index[0]][i] = '0'

                for i in range(int(new_data[1])-self.obstacle_cell_constant[0], int(new_data[1])+1):
                    if(i>=0 and self.map_sensor_data[3]<25.0):
                        self.map_list[self.drone_cell_index[0]][i] = '1'

                for i in range(int(new_data[1])-self.obstacle_cell_constant[0], int(new_data[1])+2):
                    if(i>=0 and self.map_sensor_data[3]<25.0):
                        for j in range(self.drone_cell_index[0]-self.obstacle_cell_constant[1],self.drone_cell_index[0]+self.obstacle_cell_constant[1]+1):
                            try:
                                if(self.map_list[j][i] == 'x'):
                                    self.map_list[j][i] = '1'
                            except:
                                pass

        else:
            if(new_data[0]>=self.drone_cell_index[0]):
                for i in range(int(self.drone_cell_index[0]), int(new_data[0])+1):
                    if(self.map_list[i][self.drone_cell_index[1]] == 'x'):
                        self.map_list[i][self.drone_cell_index[1]] = '0'

                for i in range(int(new_data[0])-1, int(new_data[0])+self.obstacle_cell_constant[0]):
                    if(i<self.grid_width and self.map_sensor_data[2]<25.0):
                        self.map_list[i][self.drone_cell_index[1]] = '1'

                for i in range(int(new_data[0])-2, int(new_data[0])+self.obstacle_cell_constant[0]):
                    if(i<self.grid_width and self.map_sensor_data[2]<25.0):
                        for j in range(self.drone_cell_index[1]-self.obstacle_cell_constant[1], self.drone_cell_index[1]+self.obstacle_cell_constant[1]+1):
                            try:
                                if(self.map_list[i][j] == 'x'):
                                    self.map_list[i][j] = '1'
                            except:
                                pass

            else:
                for i in range(int(new_data[0]), int(self.drone_cell_index[0])+1):
                    if(self.map_list[i][self.drone_cell_index[1]] == 'x'):
                        self.map_list[i][self.drone_cell_index[1]] = '0'

                for i in range(int(new_data[0])-self.obstacle_cell_constant[0], int(new_data[0])+1):
                    if(i>=0 and self.map_sensor_data[0]<25.0):
                        self.map_list[i][self.drone_cell_index[1]] = '1'

                for i in range(int(new_data[0])-self.obstacle_cell_constant[0], int(new_data[0])+2):
                    if(i>=0 and self.map_sensor_data[0]<25.0):
                        for j in range(self.drone_cell_index[1]-self.obstacle_cell_constant[1], self.drone_cell_index[1]+self.obstacle_cell_constant[1]+1):
                            try:
                                if(self.map_list[i][j] == 'x'):
                                    self.map_list[i][j] = '1'
                            except:
                                pass


    # Main map generation functions
    def mapping_func(self):
        # sensor data is [front, right, back, left]

        sensor_deg_data = [[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]]

        # Mapping front
        sensor_deg_data[0][1] = self.drone_position[1] - (self.map_sensor_data[0]/self.meter_conv[1])
        sensor_deg_data[0][0] = self.drone_position[0]
        self.current_cell_index = self.find_cell_index(sensor_deg_data[0])

        if self.sensor_validity[0]:
            # print(self.current_cell_index)
            self.map_update_func(self.current_cell_index, 1)

        # Mapping right
        sensor_deg_data[1][1] = self.drone_position[1]
        sensor_deg_data[1][0] = self.drone_position[0] + (self.map_sensor_data[1]/self.meter_conv[0])
        self.current_cell_index = self.find_cell_index(sensor_deg_data[1])

        if self.sensor_validity[1]:
            # print(self.current_cell_index)
            self.map_update_func(self.current_cell_index, 0)

        # Mapping back
        sensor_deg_data[2][1] = self.drone_position[1] + (self.map_sensor_data[2]/self.meter_conv[1])
        sensor_deg_data[2][0] = self.drone_position[0]
        self.current_cell_index = self.find_cell_index(sensor_deg_data[2])

        if self.sensor_validity[2]:
            # print(self.current_cell_index)
            self.map_update_func(self.current_cell_index, 1)

        # Mapping left
        sensor_deg_data[3][1] = self.drone_position[1]
        sensor_deg_data[3][0] = self.drone_position[0] - (self.map_sensor_data[3]/self.meter_conv[0])
        self.current_cell_index = self.find_cell_index(sensor_deg_data[3])

        if self.sensor_validity[3]:
            # print(self.current_cell_index)
            self.map_update_func(self.current_cell_index, 0)

        #self.map_img_list_update()


    #Functions used for path planning
    def find_next_nodes(self, node):

        next_nodes = []
        gn = []

        if node[0] - 1 >= 0 and (self.grid[node[0] - 1][node[1]] == '0' or self.grid[node[0] - 1][node[1]] == 'b' or
                                 self.grid[node[0] - 1][node[1]] == 'g' or self.grid[node[0] - 1][node[1]] == 'x'):  # f
            next_nodes.append([node[0] - 1, node[1]])
            gn.append(1)

        if node[1] + 1 <= self.grid_size[1] and node[0] - 1 >= 0 and (self.grid[node[0] - 1][node[1] + 1] == '0' or self.grid[node[0] - 1][node[1] + 1] == 'b' or
                self.grid[node[0] - 1][node[1] + 1] == 'g' or self.grid[node[0] - 1][node[1] + 1] == 'x'):  # fr
            next_nodes.append([node[0] - 1, node[1] + 1])
            gn.append(1.41)
        if node[1] + 1 <= self.grid_size[1] and (
                self.grid[node[0]][node[1] + 1] == '0' or self.grid[node[0]][node[1] + 1] == 'b' or self.grid[node[0]][node[1] + 1] == 'g' or self.grid[node[0]][node[1] + 1] == 'x'):  # r
            next_nodes.append([node[0], node[1] + 1])
            gn.append(1)

        if node[0] + 1 <= self.grid_size[0] and node[1] + 1 <= self.grid_size[1] and (
                self.grid[node[0] + 1][node[1] + 1] == '0' or self.grid[node[0] + 1][node[1] + 1] == 'b' or
                self.grid[node[0] + 1][node[1] + 1] == 'g' or self.grid[node[0] + 1][node[1] + 1] == 'x'):  # br
            next_nodes.append([node[0] + 1, node[1] + 1])
            gn.append(1.41)

        if node[0] + 1 <= self.grid_size[0] and (
                self.grid[node[0] + 1][node[1]] == '0' or self.grid[node[0] + 1][node[1]] == 'b' or
                self.grid[node[0] + 1][node[1]] == 'g' or self.grid[node[0] + 1][node[1]] == 'x'):  # b

            next_nodes.append([node[0] + 1, node[1]])
            gn.append(1)

        if node[1] - 1 >= 0 and node[0] + 1 <= self.grid_size[0] and (
                self.grid[node[0] + 1][node[1] - 1] == '0' or self.grid[node[0] + 1][node[1] - 1] == 'b' or
                self.grid[node[0] + 1][node[1] - 1] == 'g' or self.grid[node[0] + 1][node[1] - 1] == 'x'):  # bl

            next_nodes.append([node[0] + 1, node[1] - 1])
            gn.append(1.41)

        if node[1] - 1 >= 0 and (
                self.grid[node[0]][node[1] - 1] == '0' or self.grid[node[0]][node[1] - 1] == 'b' or self.grid[node[0]][node[1] - 1] == 'g' or self.grid[node[0]][node[1] - 1] == 'x'):  # l

            next_nodes.append([node[0], node[1] - 1])
            gn.append(1)

        if node[0] - 1 >= 0 and node[1] - 1 >= 0 and (
                self.grid[node[0] - 1][node[1]] == '0' or self.grid[node[0] - 1][node[1]] == 'b' or
                self.grid[node[0] - 1][node[1]] == 'g' or self.grid[node[0] - 1][node[1]] == 'x'):  # fl
            next_nodes.append([node[0] - 1, node[1] - 1])
            gn.append(1.41)

        # print("next_nodes :", next_nodes)
        # print("gn :",gn)

        # return next_nodes,gn

        return next_nodes

    def get_dist(self, node1, node2):
        dist_x = abs(node1[0] - node2[0])
        dist_y = abs(node1[1] - node2[1])

        if dist_x > dist_y:
            return 1.41 * dist_y + 1 * (dist_x - dist_y)
        else:
            return 1.41 * dist_x + 1 * (dist_y - dist_x)
        # dx = abs(node1[0] - node2[0])
        # dy = abs(node1[1] - node2[1])
        # return math.sqrt(dx * dx + dy * dy)

    def retrace(self):
        path = []
        curr_node = self.target_node

        while curr_node != self.start_node:
            path.append(curr_node)
            curr_node = self.parent_list[str(curr_node)]
        path.append(self.start_node)
        self.found_path = path

        self.smoothened_path = self.smoothing(path) # returns the path indices
        self.path_update_on_map(self.found_path,"Fpath")
        self.path_update_on_map(self.smoothened_path,"Spath")

        self.a_star_flag = False

    def path_update_on_map(self,path,val):
        if val == "Spath":
            for i in path:
                self.map_list[i[0]][i[1]] = 'Spath'
        elif val == "Fpath":
            for i in path:
                self.map_list[i[0]][i[1]] = 'Fpath'


    def clean_path_on_map(self):

        for i in range(int(self.grid_width)):
            for j in range(int(self.grid_length)):
                if self.map_list[i][j] == 'Spath' or self.map_list[i][j] == 'Fpath':
                    self.map_list[i][j] = '0'


    # Main A* function
    def a_star(self,start_node,target_node):

        self.clean_path_on_map() # for cleaning the path from the map

        # Variables used for path planning part
        self.start_node = start_node
        self.target_node = target_node
        self.grid = self.map_list
        self.grid_size = [self.grid_width - 1, self.grid_length - 1]
        self.open_list = [self.start_node]  # Nodes that are yet to be evaluated
        self.closed_list = []  # Nodes that have already been evaluated

        self.g_cost_list = [0]  # For storing g-cost of each node
        self.h_cost_list = [self.get_dist(self.start_node, self.target_node)]  # For storing h-cost of each node
        self.f_cost_list = [0 + self.get_dist(self.start_node, self.target_node)]  # For storing f-cost of each node
        self.parent_list = {}  # For storing the parent of each node
        self.current = []  # For node in open with lowest f-cost

        self.found_path = []  # Path found from A*
        self.smoothened_path = []  # Final path obtaining after smoothening

        while len(self.open_list) > 0:
            # for calculating the node with min fn
            self.current = self.open_list[0]
            min_id = 0
            for i in range(1, len(self.f_cost_list)):
                if (self.f_cost_list[i] < self.f_cost_list[min_id]) or (self.f_cost_list[i] == self.f_cost_list[min_id] and self.h_cost_list[i] < self.h_cost_list[min_id]):
                    min_id = i

            self.current = self.open_list[min_id]
            # print(self.current)
            self.open_list.pop(min_id)
            curr_f_cost = self.f_cost_list.pop(min_id)
            curr_g_cost = self.g_cost_list.pop(min_id)
            curr_h_cost = self.h_cost_list.pop(min_id)
            self.closed_list.append(self.current)

            if self.current == self.target_node:
                self.retrace()
                break

            next_possible_nodes = self.find_next_nodes(self.current)

            for i in range(len(next_possible_nodes)):
                if next_possible_nodes[i] in self.closed_list:
                    continue

                neigh_node_cost = curr_g_cost + self.get_dist(self.current, next_possible_nodes[i])

                if next_possible_nodes[i] not in self.open_list:
                    self.open_list.append(next_possible_nodes[i])

                    self.g_cost_list.append(neigh_node_cost)
                    self.h_cost_list.append(self.get_dist(next_possible_nodes[i], self.target_node))
                    self.f_cost_list.append(neigh_node_cost + self.get_dist(next_possible_nodes[i], self.target_node))
                    self.parent_list[str(next_possible_nodes[i])] = self.current

                elif next_possible_nodes[i] in self.open_list:
                    if neigh_node_cost < self.g_cost_list[self.open_list.index(next_possible_nodes[i])]:
                        self.g_cost_list[self.open_list.index(next_possible_nodes[i])] = neigh_node_cost
                        self.h_cost_list[self.open_list.index(next_possible_nodes[i])] = self.get_dist(
                            next_possible_nodes[i], self.target_node)
                        self.f_cost_list[
                            self.open_list.index(next_possible_nodes[i])] = neigh_node_cost + self.get_dist(
                            next_possible_nodes[i], self.target_node)
                        self.parent_list[str(next_possible_nodes[i])] = self.current

                # print("open_list :", self.open_list)
                # print("close_list :", self.closed_list)
                # print("gn_", self.g_cost_list, len(self.g_cost_list))
                # print("hn_", self.h_cost_list, len(self.h_cost_list))
                # print("fn_", self.f_cost_list, len(self.f_cost_list))
                # print("parent :", self.parent_list)
                print("A* algorithm done")



    # Function for getting center coordinate given the node index as a list
    def get_center_coordinate(self, node):
        # x and y coordinates of center point of origin
        x_origin = self.boundaries_final[0][0] + 0.75 / self.meter_conv[0]
        y_origin = self.boundaries_final[0][1] + 0.75 / self.meter_conv[1]
        # x and y coordinates of given point
        lat_jump = 1.5 / self.meter_conv[0]
        long_jump = 1.5 / self.meter_conv[1]
        x = x_origin + lat_jump * node[1]
        y = y_origin + long_jump * node[0]
        # returning x and y in coordinate form( NOT LIST FORM!)
        return [x, y]

    # Function for smoothening the path given by A*
    def smoothing(self, computed_path):  # Input parameter as the path output from retrace() function
        path = list(computed_path)  # dummy variable for storing the path and working on it

        # PART-1 of smoothening- Removing collinear points (By comparing slopes)
        path_length = len(path)  # Number of coordinates in the initial path
        # print("Total number of points = ", path_length)
        end_node = path[path_length - 1]  # Final node in the path
        current_node = path[0]
        while current_node != end_node:
            # print("Enters the loop")
            i = path.index(current_node)
            if path[i + 1] == end_node:  # If the next coordinate is the end node
                #print("Terminating")
                break
            try:
                m_1 = (path[i + 1][0] - current_node[0]) / (path[i + 1][1] - current_node[1])  # First slope
                m_2 = (path[i + 2][0] - path[i + 1][0]) / (path[i + 2][1] - path[i + 1][1])  # second slope
            except:
                if path[i + 1][1] == current_node[1]:
                    m_1 = 0
                else:
                    m_2 = 0

            if m_1 == m_2:
                path.pop(i + 1)  # remove the node from the path
            else:
                current_node = path[i + 1]  # Don't remove it. Go to the next node in the path

            # print(path)
        # print("modified path is (part 1) ", path)

        # PART-2 of smoothening- Further smoothening to remove zig zaggedness
        checkpoint = path[0]  # initially it is the starting point in the path
        current_point = path[1]  # initializing it with the next node in the path
        end_point = path[len(path) - 1]  # End node of the path

        while (current_point != end_point):
            # print("Enters the loop")
            current_point_next = path[path.index(current_point) + 1]  # The node next to current_point
            # print("sampling between ", checkpoint, "and", current_point_next)
            if self.walkable(checkpoint, current_point_next):
                temp = current_point
                current_point = current_point_next
                path.remove(temp)  # remove from the list
                # print("node removed-", temp)
            else:
                checkpoint = current_point
                current_point = current_point_next
                # print("cannot remove this node-", checkpoint)
        # print("final path is (part 2)", path)  # convert this to cartesian and publish
        path = path[::-1]
        print("Smoothing done")
        return path

    # Walkable function- sampling points along a line from point A to B. To be used in smoothening function
    def walkable(self, node_a, node_b):
        #print(node_a, node_b)
        coor_a = self.get_center_coordinate(node_a)  # (x,y) of node a
        coor_b = self.get_center_coordinate(node_b)  # (x,y) of node b

        if coor_a[0] < coor_b[0]:
            t = coor_a
            coor_a = coor_b
            coor_b = t

        tan = ((coor_b[1] - coor_a[1]) * self.meter_conv[1]) / (
                    (coor_b[0] - coor_a[0]) * self.meter_conv[0])  # Slope i.e tan(theta)
        theta = math.atan(tan)
        cosgree = math.cos(theta)
        singree = math.sin(theta)

        distance = math.sqrt(math.pow((coor_b[1] - coor_a[1]) * self.meter_conv[1], 2) + math.pow(
            (coor_b[0] - coor_a[0]) * self.meter_conv[0], 2))
        current_node = list(coor_a)  # Initial
        flag = True  # initially
        index = [0, 0]  # initialize

        while math.sqrt(math.pow((coor_a[1] - current_node[1]) * self.meter_conv[1], 2) + math.pow(
                (coor_a[0] - current_node[0]) * self.meter_conv[0], 2)) < distance:

            current_node[0] -= 0.3 * cosgree / self.meter_conv[0]
            current_node[1] -= 0.3 * singree / self.meter_conv[1]
            index = self.find_cell_index(current_node)

            if self.map_list[index[0]][index[1]] == '1' or self.map_list[index[0]][index[1]] == 't':
                flag = False
                # print("obstable detected at", index)
                break

        return flag


    # Main function of this node
    def path_planner(self):

        # print("FLAGS __________________")
        # print("a_star_flag", self.a_star_flag)
        # print("a_start_computed flag", self.a_star_comp_completed_flag)
        # print("a_star_nav_completed", self.a_star_nav_completed_flag)

        self.obs_validity()
        if self.a_star_flag == False or self.a_star_nav_completed_flag == True:

            self.position_setpoint = self.destination_coords
            if self.enable_setpoint_pub_flag == 1:
                self.setpoint_pub_func()

            if (self.drone_position[2] >= (self.map_altitude - 3.5)):
                self.mapping_enable_flag = True
            else:
                self.mapping_enable_flag = False
            self.mapping_func()

            # print("Direct publish")

        else:
            # self.mapping_enable_flag = False
            self.mapping_func()
            if (self.a_star_comp_completed_flag == False):
                try:
                    if self.destination_coords[0] != 0.0:
                        self.a_star(self.find_cell_index(self.drone_position[0:2]), self.find_cell_index(self.destination_coords[0:2]))
                        self.a_star_comp_completed_flag = True
                        print("A Star happening ")
                        #print("destination co-ords", self.destination_coords[0:2])
                        self.a_star_path_index = 0
                except:
                    pass



            else:
                # try:
                self.a_star_coords[0:2] = self.get_center_coordinate(self.smoothened_path[self.a_star_path_index])
                # self.a_star_coords[0] /= self.meter_conv[0]
                # self.a_star_coords[1] /= self.meter_conv[1]
                self.a_star_coords[2] = self.destination_coords[2]
                self.position_setpoint = list(self.a_star_coords)
                self.setpoint_pub_func()
                if (self.package_window_check(self.position_setpoint, [1.5, 0.7, float('inf')])):
                    if self.a_star_path_index < len(self.smoothened_path)-1:
                        self.a_star_path_index += 1
                    else:
                        self.a_star_path_index = 0
                        self.a_star_nav_completed_flag = True
                        print("NAV completed")




            if (self.drone_position[2] >= (self.map_altitude - 3.5)):
                self.mapping_enable_flag = True
            else:
                self.mapping_enable_flag = False


########################################################################################################################

# main function
if __name__ == "__main__":

    drone_boi = mapping()
    r = rospy.Rate(50) # Frequency of
    drone_boi.initial_setup_func()
    kk = 1

    while not rospy.is_shutdown():
        drone_boi.path_planner()
        #cv2.imshow("img",drone_boi.map_img)

        #print(drone_boi.map_list)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            print("started_planning")
            drone_boi.a_star_flag = True



        if cv2.waitKey(1) & 0xFF == ord('k'):
            print("k updated")
            drone_boi.a_star_index += kk
            if drone_boi.a_star_index  == 0 or drone_boi.a_star_index  == 2:
                kk -= kk






        r.sleep()

#########################################################################################################################

# END OF MAPPING SCRIPT
