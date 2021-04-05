#! /usr/bin/env python

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

######################################################################################################################
# Class of mapping node

class mapping():

    def __init__(self):

        # node definition and name
        rospy.init_node('mapping_pathplanner', anonymous=True)

        # variables used for mapping part

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
        #self.img_loc = '/home/aswinsreekumar/drone_ws/src/vitarana_drone/scripts/map_img.jpg'
        #self.manifest_csv_loc = '/home/aswinsreekumar/drone_ws/src/vitarana_drone/scripts/manifest.csv'

        # Basic variables for initial setup
        self.A1_coordinates = [18.9998102845, 72.000142461, 16.757981]             # A1 coordinates
        self.X1_coordinates = [18.9999367615, 72.000142461, 16.757981]             # X1 coordinates
        self.delivery_grid_coordinates = list()                                    # Coordinates of the delivery grid pad
        self.return_grid_coordinates = list()
        self.pickup_coordinate_list = list()
        self.drop_coordinate_list = list()                                     # Coordinates of the return grid pad
        self.drone_inital_position = [18.9998887906, 72.0002184402, 16.7579714806] # Inital position of drone
        self.position_setpoint = [0.0,0.0,0.0]                                     # Setpoint of drone published
        self.setpoint_pub_object = Vector3()                                       # Object for setpoints publisher
        self.package_cell_size = [1.5/110692.0702932625, 1.5/105292.0089353767]    # Cell size in radians

        # Grid map generation
        self.grid_length = 0.0                                      # No of cells in length
        self.grid_width = 0.0                                       # No of cells in width
        self.grid_cell_length = 1.50                                # Cell side length
        self.boundaries_computed = [[0.0,0.0],[0.0,0.0]]            # Boundaries obtained [[LEFT-TOP],[RIGHT-BOTTOM]]
        self.boundaries_final = [[0.0,0.0],[0.0,0.0]]               # Boundaries approx [[LEFT-TOP],[RIGHT-BOTTOM]]

        # Grid map cell data
        self.map_list = list()                                      # Contains altitude data of map
        self.current_cell_index = [0, 0]                            # Cell lat,long index
        self.drone_cell_index = [0, 0]                              # Cell of drone position GPS
        self.drop_coordinate_list_cells = list()                            # Package drop cell
        self.pickup_coordinate_list_cells = list()                          # Package pickup cell

        # Map image and related variables
        self.map_img_len = 1
        self.map_img_width = 1
        self.map_img = np.zeros((self.map_img_len*3, self.map_img_width*3, 3), np.uint8)         # Image of the map
        self.map_img_1 = np.zeros((self.map_img_len * 3, self.map_img_width * 3, 3), np.uint8)  # Image of the map
        self.img_scale = 5                                         # Scaling the map
        self.current_package_index = 0
        self.current_stage_index = 0

        # Mapping drone DATA
        self.sensor_validity = [True, True, True, True]
        self.mapping_enable_flag = True
        self.range_top_dist = [float('inf'), float('inf'), float('inf'), float('inf'),float('inf')]  # [front,right,back,left,top]
        self.range_bottom_data = 0.0
        self.drone_position = [0.0, 0.0, 0.0]
        self.prev_drone_position = [0.0, 0.0, 0.0]
        self.meter_conv = [110692.0702932625, 105292.0089353767, 1] 				#Factor for degrees to meters conversion(lat,long,alt to x,y,z)
        self.map_sensor_data = [0.0, 0.0, 0.0, 0.0, 0.0]                             # Sensor data [Front, Right, Back, Left, Bottom]
        self.prev_map_sensor_data = [float('inf'), float('inf'), float('inf'), float('inf'),float('inf')]            # Previous map sensor data [Front, Right, Back, Left]

        # Height control variables
        self.padding_height = 2                             # Height added to altitude data stored
        self.prev_pickup_height = 1000                      # stores the previous pickup height in stage 2. Used to avoid oscillations.
        self.pickup_pos_flag = True


        # Path planning
        self.destination_coords = [0.0, 0.0, 0.0]              # Destination coordinates of drone
        self.path_planner_flag = False
        self.path_planning_complete = False
        self.controller_info = '0000'
        self.prev_controller_info = '0000'

        # Variables used for path planning part
        self.start_node = []
        self.target_node = []
        self.grid = self.map_list
        self.grid_size = [self.grid_width - 1, self.grid_length - 1]
        self.open_list = []  # Nodes that are yet to be evaluated
        self.closed_list = []  # Nodes that have already been evaluated

        self.g_cost_list = []  # For storing g-cost of each node
        self.h_cost_list = []  # For storing h-cost of each node
        self.f_cost_list = []  # For storing f-cost of each node
        self.parent_list = {}  # For storing the parent of each node
        self.current = []  # For node in open with lowest f-cost

        self.found_path = []  # Path found from A*
        self.smoothened_path = []  # Final path obtaining after smoothening

        self.a_star_index = 0
        self.a_star_flag = False

        self.a_star_nav_completed_flag = True  # Indicates whether A* path destination reached
        self.a_star_comp_completed_flag = True
        self.a_star_path_index = 1  # Current index of A* path list
        self.a_star_coords = [0.0, 0.0, 0.0]  # A* coordinates published to position_controller
        # Marker detection
        self.enable_setpoint_pub_flag = 1  # enabling setpoint pub initially true

        self.obstacle_cell_constant = [4,4]                           # Minimum number of cells in building

        self.a_star_alt_index = 0

        self.current_action = "D"

        ####change
        #altitude bug
        self.drone_velocity = [0, 0, 0]     # x, y, z
        self.travel_direction = [0, 0]      # L/R, F/B

        self.bug_found_dir = ''

        self.bug_started = False

        self.bug_setpoint = [0,0,0]

        self.alt_bug_done = False

        self.bug_enable_flag = True
        self.navigation_case = 1

        self.drone_orientation_quaternion = [0, 0, 0, 0]
        self.drone_orientation_euler = [0, 0, 0]


######################################################################################################################
    # Subscribers and publishers

        self.setpoint_publisher = rospy.Publisher('/set_setpoint', Vector3, queue_size = 1)   # Setpoints publisher

        rospy.Subscriber('/edrone/range_finder_top', LaserScan, self.range_top)                 # Sensor data
        rospy.Subscriber('/edrone/range_finder_bottom', LaserScan, self.range_bottom)                 # Sensor data
        rospy.Subscriber('/edrone/gps', NavSatFix, self.edrone_position)                        # Current GPS of drone
        rospy.Subscriber('/destination_coordinates', Vector3, self.handle_destination_coords) # Destination from control node
        rospy.Subscriber('/controller_info', String, self.controller_info_func)  # Package info callback
        rospy.Subscriber('/enable_setpoint_pub_mapping_script', Float32, self.enable_setpoint_pub)
        rospy.Subscriber('/edrone/imu/data', Imu, self.imu_callback)

        ####change
        rospy.Subscriber('/edrone/gps_velocity', Vector3Stamped, self.velocity_callback)
        ####

#######################################################################################################################
# Callback and publisher functions

    def imu_callback(self, msg):
        self.drone_orientation_quaternion[0] = msg.orientation.x
        self.drone_orientation_quaternion[1] = msg.orientation.y
        self.drone_orientation_quaternion[2] = msg.orientation.z
        self.drone_orientation_quaternion[3] = msg.orientation.w

        (self.drone_orientation_euler[0], self.drone_orientation_euler[1],
         self.drone_orientation_euler[2]) = tf.transformations.euler_from_quaternion(
            [self.drone_orientation_quaternion[0], self.drone_orientation_quaternion[1],
             self.drone_orientation_quaternion[2], self.drone_orientation_quaternion[3]])

    def enable_setpoint_pub(self,enable):
        self.enable_setpoint_pub_flag = enable.data

    # Callback for package information
    def controller_info_func(self, data):
        list_a = ['A', 'B', 'C', "D", 'E', 'F', 'G', 'H']
        self.controller_info = str(data.data)
        if self.controller_info[0].isalpha():
            self.current_package_index = list_a.index(self.controller_info[0]) + 10
        else:
            self.current_package_index = int(self.controller_info[0])

        if self.controller_info[1].isalpha():
            dict_a = {'z': 2.5, 'y': 6.5}
            self.current_stage_index = dict_a[self.controller_info[1]]
        else:
            self.current_stage_index = int(self.controller_info[1])

        if self.controller_info[2] == "Y":
            self.a_star_flag = True
            if self.controller_info[0:2] != self.prev_controller_info[0:2]:
                self.a_star_comp_completed_flag = False
                self.a_star_nav_completed_flag = False
                #print(self.controller_info)
        else:
            self.a_star_flag = False
            self.a_star_nav_completed_flag = True

        self.current_action = self.controller_info[3]

        self.prev_controller_info = data.data

    # Publisher for setpoints to position controller
    def setpoint_pub_func(self):
        self.setpoint_pub_object.x = float(self.position_setpoint[0])
        self.setpoint_pub_object.y = float(self.position_setpoint[1])
        self.setpoint_pub_object.z = float(self.position_setpoint[2])
        self.setpoint_publisher.publish(self.setpoint_pub_object)

    # Planar sensor reading callback
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

    # Bottom sensor reading callback
    def range_bottom(self,range_bottom):
        self.range_bottom_data = range_bottom.ranges[0]

        #print("Bottom ",self.range_bottom_data)
        if(self.range_bottom_data == float('inf')):
            self.map_sensor_data[4] = 50.0 # 25
        elif (self.range_bottom_data<0.5):
            self.map_sensor_data[4] = 0.0
        else:
            zmr = self.range_bottom_data*math.cos(self.drone_orientation_euler[0])
            zmp = self.range_bottom_data*math.cos(self.drone_orientation_euler[1])
            self.map_sensor_data[4] = min(zmr, zmp)

    # Drone GPS callback function
    def edrone_position(self, gps):
        self.prev_drone_position = copy.copy(self.drone_position)
        self.drone_position[0] = gps.latitude
        self.drone_position[1] = gps.longitude
        self.drone_position[2] = gps.altitude
        self.drone_cell_index = self.find_cell_index([self.drone_position[0], self.drone_position[1]])

        #print("maplist idx : ", self.map_list[self.drone_cell_index[0]][self.drone_cell_index[1]])

    # Store incoming destination data
    def handle_destination_coords(self, coords):
        self.destination_coords[0] = coords.x
        self.destination_coords[1] = coords.y
        self.destination_coords[2] = coords.z

    ####change
    def velocity_callback(self, vel):
        self.drone_velocity = [vel.vector.x, vel.vector.y, vel.vector.z]
        for i in range(2):
            if abs(self.drone_velocity[i]) < 0.01:
                self.travel_direction[i] = 0
            elif self.drone_velocity[i] > 0:
                self.travel_direction[i] = 1 # Right, Front
            else:
                self.travel_direction[i] = -1 # Left Back
    ####



############################################################################################################################
# Algorithm

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
        #print(len(self.delivery_grid_coordinates))
        #print(len(self.return_grid_coordinates))
        #print(self.delivery_grid_coordinates[2][2])
        #print(self.return_grid_coordinates[2][2])

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
                    self.pickup_coordinate_list.append(self.delivery_grid_coordinates[grid_index[0]][grid_index[1]])
                    self.drop_coordinate_list.append([float(coords[0]),float(coords[1]),float(coords[2])])
                else:
                    grid_index = [int(row[2][1])-1,int(ord(row[2][0])-88)]
                    coords = row[1].split(';')
                    self.drop_coordinate_list.append(self.return_grid_coordinates[grid_index[0]][grid_index[1]])
                    self.pickup_coordinate_list.append([float(coords[0]),float(coords[1]),float(coords[2])])

        #print(self.pickup_coordinate_list)
        #print(self.drop_coordinate_list)

    # Checking if drone is inside the window for picking packages wrt coords and error window
    def package_window_check(self, coords, error):
        if(abs(self.drone_position[0]-coords[0])<(error[0]/self.meter_conv[0])):
            if(abs(self.drone_position[1]-coords[1])<(error[1]/self.meter_conv[1])):
                if(abs(self.drone_position[2]-coords[2])<(error[2]/self.meter_conv[2])):
                    return 1
        else:
            return 0

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


    # To compute the boundaries of grid map using drone pos, package pick and drop positions
    def boundaries_compute_func(self):
        min_latitude = 100.0
        max_latitude = 0.0
        min_longitude = 100.0
        max_longitude = 0.0

        for coordinate in self.pickup_coordinate_list:
            if(coordinate[0]>max_latitude):
                max_latitude = coordinate[0]
            if(coordinate[0]<min_latitude):
                min_latitude = coordinate[0]
            if(coordinate[1]>max_longitude):
                max_longitude = coordinate[1]
            if(coordinate[1]<min_longitude):
                min_longitude = coordinate[1]

        for coordinate in self.drop_coordinate_list:
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

        padding = 10

        self.boundaries_final[0] = [self.boundaries_computed[0][0]-(padding*self.package_cell_size[0]), self.boundaries_computed[0][1]-(padding*self.package_cell_size[1])]
        self.boundaries_final[1] = [self.boundaries_computed[1][0]+(padding*self.package_cell_size[0]), self.boundaries_computed[1][1]+(padding*self.package_cell_size[1])]

        max_latitude = (int((self.boundaries_final[1][0]-self.boundaries_final[0][0])/self.package_cell_size[0])*self.package_cell_size[0])+self.boundaries_final[0][0]
        max_longitude = (int((self.boundaries_final[1][1]-self.boundaries_final[0][1])/self.package_cell_size[1])*self.package_cell_size[1])+self.boundaries_final[0][1]

        self.boundaries_final[1] = [max_latitude, max_longitude]

        self.grid_length = math.ceil((self.boundaries_final[1][0]-self.boundaries_final[0][0])/self.package_cell_size[0])
        self.grid_width = math.ceil((self.boundaries_final[1][1]-self.boundaries_final[0][1])/self.package_cell_size[1])

        print("grid_length", self.grid_length)
        print("grid_width", self.grid_width)
        print("boundaries computed", self.boundaries_computed)
        print("boundaries final", self.boundaries_final)

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

    # Extract data of grid cell and return
    def cell_height_data(self,data):
        cell_info_data = 'X'
        cell_height_data = 200
        if(data == 500.0):
            cell_info_data = 'X'  # unmapped
            cell_height_data = 500
        if(data == 200.0):
            cell_info_data = 'O'  # obstacle
            cell_height_data = 200
        elif (data>=100.0 and data<200.0):
            cell_info_data = 'S'   # safe flying
            cell_height_data = data - 100
        elif (data<100.0):
            cell_info_data = 'A'  # got from bottom sensor
            cell_height_data = data
        return [cell_info_data,cell_height_data]

    # Updating the map image's individual cells
    def map_img_cell_update(self, cell, sign):
        cell_info = ['X',200]
        cell_info = self.cell_height_data(sign)
        if cell_info[0] == 'X':
            color = (0, 0, 255)
        elif cell_info[0] == 'S':
            color = (0, 0, 0)
        elif cell_info[0] == 'A':
            color = (255, 255, 255)
        elif cell_info[0] == 'O':
            color = (0,255,0)

        x, y = cell
        self.map_img[self.img_scale * x: self.img_scale * x + self.img_scale,self.img_scale * y: self.img_scale * y + self.img_scale] = color

    # Updating the map using map_list
    def map_img_list_update(self):
        for i in range(int(self.grid_width)):
            for j in range(int(self.grid_length)):
                self.map_img_cell_update([i, j], self.map_list[i][j])

        cv2.imwrite(self.img_loc, self.map_img)

    def mapping_height_check(self):
        window_check1 = self.package_window_check(self.pickup_coordinate_list[self.current_package_index],[5,5,float('inf')])
        window_check2 = self.package_window_check(self.drop_coordinate_list[self.current_package_index],[5,5,float('inf')])

        if(window_check1 == True):
            if(self.drone_position[2]>self.pickup_coordinate_list[self.current_package_index][2]+5):
                return True
            else:
                return False

        elif(window_check2 == True):
            if(self.drone_position[2]>self.drop_coordinate_list[self.current_package_index][2]+5):
                return True
            else:
                return False

        else:
            return True

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

    ####change
    def alt_bug(self, calc_h):
        # range_find_top - Front, right, back, left thana?
        if self.travel_direction[0] == 1 and self.map_sensor_data[1] < 4: # right
            return calc_h + 5
        elif self.travel_direction[0] == -1 and self.map_sensor_data[3] < 4: # left
            return calc_h + 5
        elif self.travel_direction[1] == 1 and self.map_sensor_data[0] < 4: # Front
            return calc_h + 5
        elif self.travel_direction[1] == -1 and self.map_sensor_data[2] < 4: # Back
            return calc_h + 5

        else:
            return calc_h

########################################################################################################################
# Mapping algorithm

    # inital setup function
    def initial_setup_func(self):
        self.grid_coordinates_compute()
        self.file_read_func()
        self.boundaries_compute_func()

        row = []
        for _ in range(int(self.grid_length)):
            row.append(500.0)
        for _ in range(int(self.grid_width)):
            self.map_list.append(list(row))
        del row

        cell_data = [0,0]
        for i in self.pickup_coordinate_list:
            cell_data = self.find_cell_index([i[0], i[1]])
            self.pickup_coordinate_list_cells.append(cell_data)
        for i in self.drop_coordinate_list:
            cell_data = self.find_cell_index([i[0], i[1]])
            self.drop_coordinate_list_cells.append(cell_data)

        count = 0
        for i in self.pickup_coordinate_list_cells:
            self.map_list[i[0]][i[1]] = self.pickup_coordinate_list[count][2] + self.padding_height
            count+=1
        count = 0
        for i in self.drop_coordinate_list_cells:
            self.map_list[i[0]][i[1]] = self.drop_coordinate_list[count][2] + self.padding_height
            count+=1

        self.map_img_initialize()

    # Map list updation, mode= 1-front/back, mode=0-left/right
    def map_update_func(self, new_data, mode):
        cell_info = ['X',200]
        if (self.mapping_enable_flag == False):
            return
        ####change
        try:
            if(mode == 0):
                if(new_data[1]>=self.drone_cell_index[1]):
                    for i in range(int(self.drone_cell_index[1]), int(new_data[1])):
                        cell_info = self.cell_height_data(self.map_list[self.drone_cell_index[0]][i])
                        if(cell_info[0] != 'A'):
                            if(cell_info[1] - self.padding_height > self.drone_position[2]) and ([self.drone_cell_index[0], i] not in self.found_path):
                                for j in range(3):
                                    cell_info = self.cell_height_data(self.map_list[self.drone_cell_index[0]-1+j][i])
                                    if(cell_info[0]!='A'):
                                        if(cell_info[0]!='O' or j==2 ):
                                            if [self.drone_cell_index[0]-1+j, i] not in self.found_path:
                                                self.map_list[self.drone_cell_index[0]-1+j][i] = 100 + self.drone_position[2]

                    for i in range(int(new_data[1])-1, int(new_data[1])+self.obstacle_cell_constant[0]):
                        if (i < self.grid_length and self.map_sensor_data[1] < 25.0):
                            cell_info = self.cell_height_data(self.map_list[self.drone_cell_index[0]][i])
                            if(cell_info[0]=='X'):
                                self.map_list[self.drone_cell_index[0]][i] = 200

                else:
                    for i in range(int(new_data[1]+1), int(self.drone_cell_index[1])+1):
                        cell_info = self.cell_height_data(self.map_list[self.drone_cell_index[0]][i])
                        if(cell_info[0] != 'A'):
                            if(cell_info[1] - self.padding_height > self.drone_position[2]) and ([self.drone_cell_index[0], i] not in self.found_path):
                                for j in range(3):
                                    cell_info = self.cell_height_data(self.map_list[self.drone_cell_index[0]-1+j][i])
                                    if(cell_info[0]!='A'):
                                        if(cell_info[0]!='O' or j==2 ):
                                            if [self.drone_cell_index[0] - 1 + j, i] not in self.found_path:
                                                self.map_list[self.drone_cell_index[0]-1+j][i] = 100 + self.drone_position[2]

                    for i in range(int(new_data[1])-self.obstacle_cell_constant[0], int(new_data[1])+1):
                        if (i >= 0 and self.map_sensor_data[3] < 25.0):
                            cell_info = self.cell_height_data(self.map_list[self.drone_cell_index[0]][i])
                            if(cell_info[0]=='X'):
                                self.map_list[self.drone_cell_index[0]][i] = 200

            else:
                if(new_data[0]>=self.drone_cell_index[0]):
                    for i in range(int(self.drone_cell_index[0]), int(new_data[0])):
                        cell_info = self.cell_height_data(self.map_list[i][self.drone_cell_index[1]])
                        if(cell_info[0] != 'A'):
                            if(cell_info[1] - self.padding_height > self.drone_position[2]) and ([i, self.drone_cell_index[1]] not in self.found_path):
                                for j in range(3):
                                    cell_info = self.cell_height_data(self.map_list[i][self.drone_cell_index[1]-1+j])
                                    if(cell_info[0]!='A'):
                                        if(cell_info[0]!='O' or j==2 ):
                                            if [i, self.drone_cell_index[1] - 1 + j] not in self.found_path:
                                                self.map_list[i][self.drone_cell_index[1]-1+j] = 100 + self.drone_position[2]

                    for i in range(int(new_data[0])-1, int(new_data[0])+self.obstacle_cell_constant[0]):
                        if (i < self.grid_width and self.map_sensor_data[2] < 25.0):
                            cell_info = self.cell_height_data(self.map_list[i][self.drone_cell_index[1]])
                            if(cell_info[0]=='X'):
                                self.map_list[i][self.drone_cell_index[1]] = 200

                else:
                    for i in range(int(new_data[0]), int(self.drone_cell_index[0])+1):
                        cell_info = self.cell_height_data(self.map_list[i][self.drone_cell_index[1]])
                        if(cell_info[0] != 'A'):
                            if(cell_info[1] -self.padding_height > self.drone_position[2]) and ([i, self.drone_cell_index[1]] not in self.found_path):
                                for j in range(3):
                                    cell_info = self.cell_height_data(self.map_list[i][self.drone_cell_index[1]-1+j])
                                    if(cell_info[0]!='A'):
                                        if(cell_info[0]!='O' or j==2 ):
                                            if [i, self.drone_cell_index[1] - 1 + j] not in self.found_path:
                                                self.map_list[i][self.drone_cell_index[1]-1+j] = 100 + self.drone_position[2]

                    for i in range(int(new_data[0])-self.obstacle_cell_constant[0], int(new_data[0])+1):
                        if (i >= 0 and self.map_sensor_data[0] < 25.0):
                            cell_info = self.cell_height_data(self.map_list[i][self.drone_cell_index[1]])
                            if(cell_info[0]=='X'):
                                self.map_list[i][self.drone_cell_index[1]] = 200

        except IndexError:
            pass
            #print("Indexerror")

    # Main map generation functions
    def mapping_func(self):
        self.mapping_enable_flag = self.mapping_height_check()
        # sensor data is [front, right, back, left]
        sensor_deg_data = [[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]]

        # Mapping front
        sensor_deg_data[0][1] = self.drone_position[1] - (self.map_sensor_data[0]/self.meter_conv[1])
        sensor_deg_data[0][0] = self.drone_position[0]
        self.current_cell_index = self.find_cell_index(sensor_deg_data[0])

        self.map_update_func(self.current_cell_index, 1)

        # Mapping right
        sensor_deg_data[1][1] = self.drone_position[1]
        sensor_deg_data[1][0] = self.drone_position[0] + (self.map_sensor_data[1]/self.meter_conv[0])
        self.current_cell_index = self.find_cell_index(sensor_deg_data[1])


        self.map_update_func(self.current_cell_index, 0)

        # Mapping back
        sensor_deg_data[2][1] = self.drone_position[1] + (self.map_sensor_data[2]/self.meter_conv[1])
        sensor_deg_data[2][0] = self.drone_position[0]
        self.current_cell_index = self.find_cell_index(sensor_deg_data[2])


        self.map_update_func(self.current_cell_index, 1)

        # Mapping left
        sensor_deg_data[3][1] = self.drone_position[1]
        sensor_deg_data[3][0] = self.drone_position[0] - (self.map_sensor_data[3]/self.meter_conv[0])
        self.current_cell_index = self.find_cell_index(sensor_deg_data[3])


        self.map_update_func(self.current_cell_index, 0)

        # Mapping bottom
        cell_info = ['X', 200]
        cell_info = self.cell_height_data(self.map_list[self.drone_cell_index[0]][self.drone_cell_index[1]])
        if(self.map_sensor_data[4]>0.0 and self.map_sensor_data[4]<50.0): # 25

            for i in range(3):
                for j in range(3):
                    cell_info = self.cell_height_data(self.map_list[self.drone_cell_index[0]-1+i][self.drone_cell_index[1]-1+j])
                    if cell_info[0] != 'A':
                        self.map_list[self.drone_cell_index[0]-1+i][self.drone_cell_index[1]-1+j] = self.drone_position[2] - self.map_sensor_data[4] + self.padding_height

        elif((self.map_sensor_data[4] == 50.0 or self.map_sensor_data[4] == 0)): # and cell_info[0]!='A'): # 25
            if(self.drone_position[2] - self.map_sensor_data[4] < cell_info[1] - self.padding_height):
                self.map_list[self.drone_cell_index[0]][self.drone_cell_index[1]] = 100 + self.drone_position[2] - self.map_sensor_data[4] + self.padding_height

        self.map_img_list_update()

    # Functions used for path planning
    def find_next_nodes(self, node):

        next_nodes = []
        gn = []

        if node[0] - 1 >= 0 and (self.grid[node[0] - 1][node[1]] != 200):  # f
            next_nodes.append([node[0] - 1, node[1]])
            gn.append(1)

        if node[1] + 1 <= self.grid_size[1] and node[0] - 1 >= 0 and (self.grid[node[0] - 1][node[1] + 1] != 200):  # fr
            next_nodes.append([node[0] - 1, node[1] + 1])
            gn.append(1.41)
        if node[1] + 1 <= self.grid_size[1] and (self.grid[node[0]][node[1] + 1] != 200):  # r
            next_nodes.append([node[0], node[1] + 1])
            gn.append(1)

        if node[0] + 1 <= self.grid_size[0] and node[1] + 1 <= self.grid_size[1] and (self.grid[node[0] + 1][node[1] + 1] != 200):  # br
            next_nodes.append([node[0] + 1, node[1] + 1])
            gn.append(1.41)

        if node[0] + 1 <= self.grid_size[0] and (self.grid[node[0] + 1][node[1]] != 200):  # b
            next_nodes.append([node[0] + 1, node[1]])
            gn.append(1)

        if node[1] - 1 >= 0 and node[0] + 1 <= self.grid_size[0] and (self.grid[node[0] + 1][node[1] - 1] != 200):  # bl
            next_nodes.append([node[0] + 1, node[1] - 1])
            gn.append(1.41)

        if node[1] - 1 >= 0 and (self.grid[node[0]][node[1] - 1] != 200):  # l
            next_nodes.append([node[0], node[1] - 1])
            gn.append(1)

        if node[0] - 1 >= 0 and node[1] - 1 >= 0 and (self.grid[node[0] - 1][node[1]] != 200):  # fl
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
        # print("RetracinGGGG ")
        path = []
        curr_node = self.target_node

        while curr_node != self.start_node:
            path.append(curr_node)
            curr_node = self.parent_list[str(curr_node)]
        path.append(self.start_node)
        self.found_path = path[::-1]

        self.smoothened_path = self.smoothing(path)  # returns the path indices
        #self.path_update_on_map(self.found_path, "Fpath")
        #self.path_update_on_map(self.smoothened_path, "Spath")

        self.a_star_flag = False
        # print("SMOOTHIE -1 : ", self.smoothened_path)
        # print("PAATHU PA : ", self.found_path)
        # print("Retraced pa")

    def path_update_on_map(self, path, val):
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
    def a_star(self, start_node, target_node):
        # print(" A-STAR STARTED")
        self.clean_path_on_map()  # for cleaning the path from the map

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
                    self.f_cost_list.append(
                        neigh_node_cost + self.get_dist(next_possible_nodes[i], self.target_node))
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
                # print("Terminating")
                break
            try:
                m_1 = (path[i + 1][0] - current_node[0]) / (path[i + 1][1] - current_node[1])  # First slope
            except:
                m_1 = 'v'

            try:
                m_2 = (path[i + 2][0] - path[i + 1][0]) / (path[i + 2][1] - path[i + 1][1])  # second slope
            except:
                m_2 = 'v'

            if m_1 == m_2:
                path.pop(i + 1)  # remove the node from the path
            else:
                current_node = path[i + 1]  # Don't remove it. Go to the next node in the path

            # print(path)
        # print("modified path is (part 1) ", path)
        # print("Part - 1 output : ", path)

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
        # print("Smoothing done")
        return path

    # Walkable function- sampling points along a line from point A to B. To be used in smoothening function
    def walkable(self, node_a, node_b):
        # print(node_a, node_b)
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

            if self.map_list[index[0]][index[1]] == 200:
                flag = False
                # print("obstable detected at", index)
                break

        return flag


    def compute_height(self):

        calculated_height = 0

        if self.current_stage_index == 0:
            self.a_star_alt_index = 0
            self.pickup_pos_flag = True
            print("CH S 0")
            pass

        elif self.current_stage_index == 1:
            # ctr node pub - pickup h + 0.5

            self.prev_pickup_height = 1000

            print("CH S 1")
            calculated_height = self.destination_coords[2]
            drone_neib_list = []
            # finding neighbours of drone cell
            for i in range(3):
                for j in range(3):
                    drone_neib_list.append([self.drone_cell_index[0] -1 + i, self.drone_cell_index[1] -1 + j])

            if len(self.found_path) > 0:
                x_diff = self.found_path[-1][1] - self.drone_cell_index[1]
                y_diff = self.found_path[-1][0] - self.drone_cell_index[0]

            for cell in drone_neib_list:
                if cell in self.found_path:
                    #print("drone in path", self.a_star_alt_index)
                    self.a_star_alt_index = self.found_path.index(cell)
                    break

            for j in self.found_path[self.a_star_alt_index:]:
                # if (self.found_path[-1][1] - j[1]) * x_diff > 0 and (self.found_path[-1][0] - j[0]) * y_diff > 0:
                info = self.cell_height_data(self.map_list[j[0]][j[1]])
                #print(" first guy info : ", info[1])
                if info[0] != 'X' and info[0] != 'O':
                    if (info[1] > calculated_height):
                        calculated_height = info[1]

            # calculated_height += 2

            dist_x = (self.destination_coords[0] - self.drone_position[0]) * self.meter_conv[0]
            dist_y = (self.destination_coords[1] - self.drone_position[1]) * self.meter_conv[1]
            dist = math.sqrt(dist_x ** 2 + dist_y ** 2)

            if self.current_action == "D":
                calculated_height = self.destination_coords[2]

            elif dist < 7:
                if self.drone_position[2] - self.pickup_coordinate_list[self.current_package_index][2] > 0.8 and self.pickup_pos_flag:
                    calculated_height = self.pickup_coordinate_list[self.current_package_index][2] + 0.3
                elif self.drone_position[2] - self.pickup_coordinate_list[self.current_package_index][2] < 0.6 and self.pickup_pos_flag:
                    calculated_height = self.pickup_coordinate_list[self.current_package_index][2] + 3
                elif self.drone_position[2] - self.pickup_coordinate_list[self.current_package_index][2] > 0.2:
                    calculated_height = self.pickup_coordinate_list[self.current_package_index][2] + 0.3
                    self.pickup_pos_flag = False

            elif dist < 20 and (calculated_height - self.destination_coords[2] > 8 ):
                return self.destination_coords[2] + 5

            else:
                return calculated_height + 2


        elif self.current_stage_index == 2:

            print("CH S 2")
            self.pickup_pos_flag = True
            self.a_star_alt_index = 0

            calculated_height = min(self.destination_coords[2] + 0.3, self.prev_pickup_height)
            if (self.package_window_check(
                    [self.destination_coords[0], self.destination_coords[1], self.destination_coords[2]],
                    [0.1, 0.1, float('inf')])):
                # print("Going True")
                window = 0
                while True:
                    if (self.package_window_check(
                            [self.destination_coords[0], self.destination_coords[1], self.destination_coords[2] - 3],
                            [0.15, 0.15, 3.5 - window])):
                        calculated_height -= 0.02
                        window += 0.01
                        # print("window : ", window)
                    else:
                        window = 0
                        break

                calculated_height = min(calculated_height, self.prev_pickup_height)
                self.prev_pickup_height = calculated_height


        elif self.current_stage_index == 2.5:
            self.prev_pickup_height = 1000
            print("CH S 2.5")
            self.a_star_alt_index = 0
            calculated_height = self.destination_coords[2]



        elif self.current_stage_index == 3:
            print("CH S 3")

            calculated_height = self.destination_coords[2]
            drone_neib_list = []
            # finding neighbours of drone cell
            for i in range(3):
                for j in range(3):
                    drone_neib_list.append([self.drone_cell_index[0] - 1 + i, self.drone_cell_index[1] - 1 + j])

            x_diff = self.found_path[-1][1] - self.drone_cell_index[1]
            y_diff = self.found_path[-1][0] - self.drone_cell_index[0]

            for cell in drone_neib_list:
                if cell in self.found_path:
                    #print("drone in path", self.a_star_alt_index)
                    self.a_star_alt_index = self.found_path.index(cell)
                    break

            for j in self.found_path[self.a_star_alt_index:]:
                # if (self.found_path[-1][1] - j[1])*x_diff > 0 and (self.found_path[-1][0] - j[0])*y_diff > 0:
                info = self.cell_height_data(self.map_list[j[0]][j[1]])
                #print(" first guy info : ", info[1])
                if info[0] != 'X' and info[0] != 'O':
                    if (info[1] > calculated_height):
                        calculated_height = info[1]

            # calculated_height += 2

            dist_x = (self.destination_coords[0] - self.drone_position[0]) * self.meter_conv[0]
            dist_y = (self.destination_coords[1] - self.drone_position[1]) * self.meter_conv[1]
            dist = math.sqrt(dist_x ** 2 + dist_y ** 2)



            if dist < 10:
                #print("Radius Dist : ", dist)
                if self.current_action == "R":
                    calculated_height = self.destination_coords[2] + 0.2
                else:
                    calculated_height = self.destination_coords[2] + 7

            elif dist > 185: # rough
                return 35
            else:
                return calculated_height + 2



        elif self.current_stage_index == 4:
            print("CH S 4")
            calculated_height = self.destination_coords[2]
            pass



        elif self.current_stage_index == 5:
            print("CH S 5")
            calculated_height = self.destination_coords[2]
            pass



        elif self.current_stage_index == 6:
            print("CH S 6")
            cell_data = self.find_cell_index([self.destination_coords[0], self.destination_coords[1]])
            self.a_star_alt_index = 0
            if (self.map_list[cell_data[0]][cell_data[1]] < 100):
                calculated_height = self.map_list[cell_data[0]][cell_data[1]] - 2
                # print("unakenappa...", calculated_height)
            else:
                calculated_height = self.destination_coords[2]

        elif self.current_stage_index == 6.5:
            print("CH S 6.5")
            calculated_height = self.destination_coords[2]

        elif self.current_stage_index == 7:
            print("CH S 7")
            calculated_height = self.destination_coords[2]

        elif self.current_stage_index == 8:
            print("CH S 8")
            calculated_height = self.destination_coords[2]

        elif self.current_stage_index == 9:
            print("CH S 9")
            calculated_height = self.destination_coords[2]
            pass

        else:
            calculated_height = self.destination_coords[2]
            #print("Error")

        # calculated_height += 2
        print("Computed height : ", calculated_height)
        return calculated_height



    # Enable or disable bug algo
    def bug_decide(self):
        flag = True
        destination_cell_index = self.smoothened_path[-1]
        list_no = len(self.smoothened_path)
        for i in range(0,3):
            for j in range(0,3):
                try:
                    if(self.map_list[destination_cell_index[0]-1+i][destination_cell_index[1]-1+j]!=500):
                        if(i!=1 and j!=1):
                            flag = False
                            break
                except:
                    print("Error ")
        if(flag):   # unmapped / partially mapped
            if(list_no > 2):
                # self.bug_enable_flag = False
                self.navigation_case = 2
            elif(list_no == 2):
                # self.bug_enable_flag = True
                self.navigation_case = 1
        else:       # mapped
            # self.bug_enable_flag = False
            self.navigation_case = 3

        #print("BEF : ", self.bug_enable_flag)
        self.bug_enable_flag = True

    # Bug Algorithm
    def bug_algo(self):
        if(self.bug_enable_flag == False):
            return

    def bug_boi(self):

        if self.bug_enable_flag == True:

            available_direction = ['f', 'r', 'b', 'l']

            if self.bug_started == False:

                obs_direction = ''
                pos_dir = []

                # range_find_top - Front, right, back, left thana?
                if self.travel_direction[0] == 1 and self.range_top_dist[1] < 8 and self.range_top_dist[1] > 0.35:  # right
                    obs_direction = 'r'
                    #available_direction.pop(1)
                elif self.travel_direction[0] == -1 and self.range_top_dist[3] < 8 and self.range_top_dist[3] > 0.35:  # left
                    obs_direction = 'l'
                    #available_direction.pop(3)
                elif self.travel_direction[1] == 1 and self.range_top_dist[0] < 8 and self.range_top_dist[0] > 0.35:  # Front
                    obs_direction= 'f'
                    #available_direction.pop(0)
                elif self.travel_direction[1] == -1 and self.range_top_dist[2] < 8 and self.range_top_dist[2] > 0.35:  # Back
                    obs_direction = 'b'
                    #available_direction.pop(2)

                if obs_direction != '':
                    self.bug_started = True
                    self.bug_setpoint = list(self.position_setpoint)
                    self.position_setpoint = list(self.drone_position)
                    #print("BUGGING")
                    self.position_setpoint[2] += 10
                    self.setpoint_pub_func()
                    self.bug_found_dir = self.bug_direction_find(obs_direction)


            else:
                if self.alt_bug_done == False:
                    if  (self.bug_setpoint[2] - self.drone_position[2] > -0.5) and (self.bug_setpoint[2] - self.drone_position[2] < 5):
                        self.bug_in_action('up',available_direction.index(self.bug_found_dir))
                    else:
                        self.alt_bug_done = True

                else:
                    self.bug_in_action(self.bug_found_dir, available_direction.index(self.bug_found_dir))




    def bug_in_action(self,direction,sensor_index):
        if direction == 'up':
            if self.map_sensor_data[sensor_index] == float('inf') or self.map_sensor_data[sensor_index] <= 0.3:
                self.position_setpoint = list(self.drone_position)
                self.position_setpoint[2] += 3
                self.setpoint_pub_func()
                self.bug_started = False

        # elif direction == 'r':



            # if abs(self.drone_position[2] - self.bug_setpoint[2]) < 0.4




    def bug_direction_find(self,obs_direction):

        all_dir = ['f', 'r', 'b', 'l']  # {'f': 0, 'r': 1, 'b': 2, 'l': 3}

        if all_dir.index(obs_direction) == 0:
            pos_dir = [all_dir[len(all_dir) - 1], all_dir[all_dir.index(obs_direction) + 1]]
        elif all_dir.index(obs_direction) == len(all_dir) - 1:
            pos_dir = [all_dir[all_dir.index(obs_direction) - 1], all_dir[0]]
        else:
            pos_dir = [all_dir[all_dir.index(obs_direction) - 1], all_dir[all_dir.index(obs_direction) + 1]]

        dest_pos = np.array(list(self.bug_setpoint[0:2])) - np.array(list(self.drone_position[0:2]))
        dest_pos.tolist()

        if ('r' in pos_dir) and ('l' in pos_dir):
            if dest_pos[0] >= 0 and self.map_sensor_data[all_dir.index('r')] > 8:
                return 'r'
            elif dest_pos[0] >= 0 and self.map_sensor_data[all_dir.index('r')] > 0.3 and self.map_sensor_data[all_dir.index('r')] < 8:
                return 'l'
            elif dest_pos[0] < 0 and self.map_sensor_data[all_dir.index('l')] > 0.3 and self.map_sensor_data[all_dir.index('l')] < 8:
                return 'r'
            else:
                return 'l'
        else:
            if dest_pos[1] >= 0 and self.map_sensor_data[all_dir.index('b')] > 8:
                return 'b'
            elif dest_pos[1] >= 0 and self.map_sensor_data[all_dir.index('b')] > 0.3 and self.map_sensor_data[all_dir.index('b')] < 8:
                return 'f'
            elif dest_pos[1] < 0 and self.map_sensor_data[all_dir.index('f')] > 0.3 and self.map_sensor_data[all_dir.index('f')] < 8:
                return 'b'
            else:
                return 'f'


    #
    # def bug_direction_find(self,available_direction):
    #     #cur_pos = list(self.drone_position)
    #     new_available_direction  = list(available_direction)
    #     move_direction = []
    #     dict_b = {'f':0,'r':1,'b':2,'l':3}
    #     dest_pos = np.array(list(self.position_setpoint[0:2])) - np.array(list(self.drone_position[0:2]))
    #     dest_pos.tolist()
    #
    #     for i in available_direction:
    #         if self.map_sensor_data[dict_b[i]] < 8 and self.map_sensor_data[dict_b[i]] > 0.3:
    #             new_available_direction.pop(dict_b[i])
    #
    #     if dest_pos[0] > 0 and dest[2] > 0:
    #         move_direction.extend(['r','b'])
    #     elif dest_pos[0] > 0 and dest[2] < 0:
    #         move_direction.extend(['r', 'f'])
    #     elif dest_pos[0] < 0 and dest[2] > 0:
    #         move_direction.extend(['l', 'b'])
    #     elif dest_pos[0] < 0 and dest[2] < 0:
    #         move_direction.extend(['l', 'f'])
    #
    #     if (move_direction[0] in new_available_direction) and (move_direction[1] in new_available_direction):
    #         if abs(dest_pos[0]) < abs(dest_pos[1]):
    #             return move_direction[1]
    #         else:
    #             return move_direction[0]
    #
    #     elif (move_direction[0] in new_available_direction):
    #         return move_direction[0]
    #     elif (move_direction[1] in new_available_direction):
    #         return move_direction[1]
    #     elif len(new_available_direction) == 1:
    #         return new_available_direction[0]
    #     elif len(new_available_direction) == 2:
    #         if abs(dest_pos[0]) < abs(dest_pos[1]) :
    #             move_direction.pop(1)
    #         else:
    #             move_direction.pop(0)

    # Main function of this node
    def path_planner(self):        #print("FLAGS __________________")
        #print("a_star_flag", self.a_star_flag)
        #print("a_start_computed flag", self.a_star_comp_completed_flag)
        #print("a_star_nav_completed", self.a_star_nav_completed_flag)
        # print("computed_height :",self.compute_height())
        # print("P.S : ", self.position_setpoint)


        if self.a_star_flag == False or self.a_star_nav_completed_flag == True:
            self.mapping_func()
            cell_data = [0,0]
            self.position_setpoint[0] = self.destination_coords[0]
            self.position_setpoint[1] = self.destination_coords[1]
            self.position_setpoint[2] = self.compute_height()
            if self.enable_setpoint_pub_flag == 1:
                self.setpoint_pub_func()
            # print("Direct publish")

        else:
            #print("In a-star section")
            self.mapping_enable_flag = False
            self.mapping_func()

            if (self.a_star_comp_completed_flag == False):
                try:
                    if self.destination_coords[0] != 0.0:
                        self.a_star(self.find_cell_index(self.drone_position[0:2]), self.find_cell_index(self.destination_coords[0:2]))
                        # print("indices for a-star",self.find_cell_index(self.drone_position[0:2]), self.find_cell_index(self.destination_coords[0:2]))
                        self.a_star_comp_completed_flag = True
                        #print("A Star happening ")
                        # print("destination co-ords", self.destination_coords[0:2])
                        self.a_star_path_index = 1 # 0
                        self.bug_decide()
                        #print("Bug decision algorithm active")
                except:
                    pass

            else:
                # try:
                # print("SMOOTHIE : ", self.smoothened_path)
                self.bug_enable_flag = True
                self.a_star_coords[0:2] = self.get_center_coordinate(self.smoothened_path[self.a_star_path_index])
                # self.a_star_coords[0] /= self.meter_conv[0]
                # self.a_star_coords[1] /= self.meter_conv[1]
                # if(self.bug_altitude_flag == False):
                self.a_star_coords[2] = self.compute_height() + 2
                # else:
                #     self.a_star_coords[2] = self.bug_added_height

                self.position_setpoint = list(self.a_star_coords)
                self.setpoint_pub_func()
                # print("position : ", self.position_setpoint)
                # print("Navigation happening")

                if len(self.smoothened_path) - self.a_star_path_index == 1:
                    self.a_star_nav_completed_flag = True
                    if self.navigation_case == 2:
                        self.bug_enable_flag = True

                elif (self.package_window_check(self.position_setpoint, [1.5, 0.7, float('inf')])):
                    if self.a_star_path_index < len(self.smoothened_path)-1:
                        if(self.navigation_case==2 and len(self.smoothened_path)-self.a_star_path_index == 2):
                            self.bug_enable_flag = True
                        self.a_star_path_index += 1
                    else:
                        self.a_star_path_index = 1 # 0
                        self.a_star_nav_completed_flag = True

                #print("NAV Ongoing !!!")

                #self.bug_algo()
                self.bug_boi()
                # except:
                #     pass

##########################################################################################################################
# Main function

if __name__ == "__main__":
    drone_boi = mapping()
    r = rospy.Rate(50)
    drone_boi.initial_setup_func()
    while not rospy.is_shutdown():
        drone_boi.path_planner()
        #drone_boi.mapping_func()
        cv2.imshow("img",drone_boi.map_img)
        cv2.waitKey(1)
        r.sleep()
#############################################################################################################################
