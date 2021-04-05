#! /usr/bin/env python

'''
# Team ID:           2537
# Theme:             Vitarana Drone
# Author List:       Jai Kesav, Aswin Sreekumar, Girish K, Greeshwar R S
# Filename:          Task_6_VD_2537_Mapping_node.py

# Functions:         imu_callback, enable_setpoint_pub, controller_info_func, setpoint_pub_func, range_top,range_bottom
#                    edrone_position, handle_destination_coords, velocity_callback, grid_coordinates_compute, file_read_func
#                    package_window_check, find_cell_index, boundaries_compute_func, cell_height_data, mapping_height_check
#                    obs_validity, initial_setup_func, map_update_func, mapping_func, find_next_nodes, get_dist, retrace
#                    a_star, get_center_coordinate, smoothing,walkable, compute_height, path_planner
                     map_img_initialize, map_img_cell_update, map_img_list_update

# Global variables:  start_node,target_node,grid,grid_size,open_list,closed_list,g_cost_list,h_cost_list,f_cost_list,parent_list
                     current,found_path,smoothened_path,a_star_index,a_star_flag,a_star_nav_completed_flag ,a_star_comp_completed_flag
                     a_star_path_index ,a_star_coords
                     img_loc, , manifest_csv_loc, A1_coordinates, X1_coordinates, delivery_grid_coordinates, return_grid_coordinates, pickup_coordinate_list
                     drop_coordinate_list, drone_inital_position, position_setpoint, package_cell_size, grid_length, grid_width, grid_cell_length, boundaries_computed
                     boundaries_final, map_list, current_cell_index, drone_cell_index, drop_coordinate_list_cells, pickup_coordinate_list_cells, map_img_len,
                     map_img_width, map_img, map_img_1, img_scale, current_package_index, current_stage_index, sensor_validity, mapping_enable_flag, range_top_dist
                     range_bottom_data, drone_position, prev_drone_position, meter_conv, map_sensor_data, prev_map_sensor_data, padding_height, prev_pickup_height
                     pickup_pos_flag, destination_coords, path_planner_flag, path_planning_complete, controller_info, prev_controller_info, enable_setpoint_pub_flag
                     obstacle_cell_constant, a_star_alt_index, current_action

'''

'''
What this script does:
The mapping script subscribes to the coordinates from control node and performs integrated A* based on conditions and mapping is enabled based on thresholds.
It uses an hybrid-continous scale mapping technique, compared to binary mapping we used in TASK4.
The entire arena is a grid divided into cells and each cell stores a floating point number based on key.
The sensor readings of drone are used to compute the safe flying distance of each cell and is updated continously.
During each navigation condition, the altitudes values in path cells are processed and a suitable flying height is assigned to the drone.
Integrated A* is applied during navigation segments and basic bug algorithm has been enabled in case of initial unmapped arena condition.
The bug algorithm gives a shot in raised altitude control before performing the required bug algorithm itself.
The position setpoint computed is published to position controller script
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

######################################################################################################################

# mapping(): Class of mapping node
class mapping():

    def __init__(self):

        # node definition and name
        rospy.init_node('mapping_pathplanner', anonymous=True)

        # variables used for mapping part

        # Local path of JAGG
        # JK
        self.img_loc = '/home/jk56/PycharmProjects/pythonProject/map_img.jpg'
        self.manifest_csv_loc = '/home/jk56/catkin_ws/src/vitarana_drone/scripts/final_scripts/sequenced_manifest_original.csv'
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
        self.A1_coordinates = [18.9998102845, 72.000142461, 16.757981]
        self.X1_coordinates = [18.9999367615, 72.000142461, 16.757981]
        #delivery_grid_coordinates: Coordinates of the delivery grid pad
        self.delivery_grid_coordinates = list()
        self.return_grid_coordinates = list()
        self.pickup_coordinate_list = list()
        self.drop_coordinate_list = list()
        self.drone_inital_position = [18.9998887906, 72.0002184402, 16.7579714806]
        self.position_setpoint = [0.0,0.0,0.0]
        #setpoint_pub_object: # Object for setpoints publisher
        self.setpoint_pub_object = Vector3()
        self.package_cell_size = [1.5/110692.0702932625, 1.5/105292.0089353767]

        # Grid map generation code snippet
        self.grid_length = 0.0
        self.grid_width = 0.0
        self.grid_cell_length = 1.50
        #boundaries_computed: Boundaries obtained [[LEFT-TOP],[RIGHT-BOTTOM]]
        self.boundaries_computed = [[0.0,0.0],[0.0,0.0]]
        #boundaries_final- Boundaries approx [[LEFT-TOP],[RIGHT-BOTTOM]]
        self.boundaries_final = [[0.0,0.0],[0.0,0.0]]

        # Grid map cell data
        #map_list: List containing altitude data of the map
        self.map_list = list()
        #current_cell_index: List containing Cell lat,long index
        self.current_cell_index = [0, 0]
        #drone_cell_index: Cell index of drone position GPS
        self.drone_cell_index = [0, 0]
        # drop_coordinate_list_cells- Package drop cell index
        self.drop_coordinate_list_cells = list()
        #pickup_coordinate_list_cells: Package pickup cell
        self.pickup_coordinate_list_cells = list()

        # Map image and related variables
        self.map_img_len = 1
        self.map_img_width = 1
        self.map_img = np.zeros((self.map_img_len*3, self.map_img_width*3, 3), np.uint8)
        self.map_img_1 = np.zeros((self.map_img_len * 3, self.map_img_width * 3, 3), np.uint8)
        self.img_scale = 5
        #current_package_index:Current package handled in index form
        self.current_package_index = 0
        #current_stage_index: Current task stage
        self.current_stage_index = 0
        # Mapping drone DATA
        #sensor_validity: Sensor filtering during high pitch roll
        self.sensor_validity = [True, True, True, True]
        #mapping_enable_flag: Flag for Enabling/disabling mapping
        self.mapping_enable_flag = True
         #range_top_dist- Sensor readings- [front,right,back,left,top]
        self.range_top_dist = [float('inf'), float('inf'), float('inf'), float('inf'),float('inf')]
        #range_bottom_data: Bottom sensor readings
        self.range_bottom_data = 0.0
        self.drone_position = [0.0, 0.0, 0.0]
        self.prev_drone_position = [0.0, 0.0, 0.0]
        self.meter_conv = [110692.0702932625, 105292.0089353767, 1]
        #map_sensor_data: Sensor data [Front, Right, Back, Left, Bottom]
        self.map_sensor_data = [0.0, 0.0, 0.0, 0.0, 0.0]
        #prev_map_sensor_data: Previous map sensor data [Front, Right, Back, Left]
        self.prev_map_sensor_data = [float('inf'), float('inf'), float('inf'), float('inf'),float('inf')]

        # Height control variables
        #padding_height: Height added to altitude data stored as a safety factor
        self.padding_height = 2
        #prev_pickup-height: stores the previous pickup height in stage 2. Used to avoid oscillations.
        self.prev_pickup_height = 1000
        self.pickup_pos_flag = True


        # Path planning
        #destination_coords: Destination coordinates of drone
        self.destination_coords = [0.0, 0.0, 0.0]
        #path_planner_flag: Enable or disable path planner algorithm
        self.path_planner_flag = False
        #path_planning_complete: Whether path planner algorithm has finished computing path
        self.path_planning_complete = False
        #controller_info: Control node info subscribed
        self.controller_info = '0000'
        self.prev_controller_info = '0000'                     # Previous info subscribed from control node

        # Variables used for path planning part
        self.start_node = []
        self.target_node = []
        self.grid = self.map_list
        self.grid_size = [self.grid_width - 1, self.grid_length - 1]
        #open_list:Nodes that are yet to be evaluated in a_star
        self.open_list = []
        #closed_list:Nodes that have already been evaluated by a_star
        self.closed_list = []
        #g_cost_list: Storing g-cost of each node
        self.g_cost_list = []
        #h-cost-list: For storing h-cost of each node
        self.h_cost_list = []
        #f_cost_list: For storing f-cost of each node, f_cost=g_cost+h_cost
        self.f_cost_list = []
        #parent_list: For storing the parent of each node
        self.parent_list = {}
        #current: For node in open with lowest f-cost
        self.current = []

        #found_path: Path found from A*
        self.found_path = []
        #smoothened_path:  # Final path obtaining after smoothening of A* path
        self.smoothened_path = []
        #self.a_star_index = 0
        #a_star_flag: Flag for knowing whether A* computation is happening
        self.a_star_flag = False
        #a_star_nav_completed_flag: Flag Indicating whether A* path destination reached
        self.a_star_nav_completed_flag = True
        #a_star_comp_completed_flag: Indicates whether Path has been computed
        self.a_star_comp_completed_flag = True
        #a_star_path_index: Current index of A* path list
        self.a_star_path_index = 1
        #a_star_coords: A* coordinates published to position_controller
        self.a_star_coords = [0.0, 0.0, 0.0]

        # Marker detection
        self.enable_setpoint_pub_flag = 1  # enabling setpoint pub initially true

        self.obstacle_cell_constant = [4,4] # Minimum number of cells in building for padding
        self.a_star_alt_index = 0           # Index of navigation A* found path
        self.current_action = "D"           # Currently doing a delivery or return

        # Hybrid bug algorithm variables
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

        self.bug_stage_flag = 1
        self.im_breaking_init = Float32()
        self.obs_direction = ""
        self.bug_stop_setpoint = []
        self.bug_velocity_direction = [0,0]  # L/R, F/B

        self.obs_cord = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.prev_obs_cord = [[0.00001, 0.0000001, 1], [1, 0.000001, 1], [0.000001, 1, 1], [1, 1, 0.000001]]
        self.obs_estimate = ['n', 'n', 'n', 'n']
        self.snapshot_a = [[0, 0], [0, 0]]  # [lat, lon], [vx, vy]
        self.snapshot_b = [[1, 1], [1, 1]]  # [lat, lon], [vx, vy]

        self.bug_count = 0

        self.pickup_down_flag = True




######################################################################################################################
    # Subscribers and publishers
        #setpoint_publisher: # Setpoints publisher
        self.setpoint_publisher = rospy.Publisher('/set_setpoint', Vector3, queue_size = 1)
        self.im_breaking_init_pub = rospy.Publisher('/start_breaking', Float32, queue_size=1)

        rospy.Subscriber('/edrone/range_finder_top', LaserScan, self.range_top)                     # Sensor data
        rospy.Subscriber('/edrone/range_finder_bottom', LaserScan, self.range_bottom)               # Sensor data
        rospy.Subscriber('/edrone/gps', NavSatFix, self.edrone_position)                            # Current GPS of drone
        rospy.Subscriber('/destination_coordinates', Vector3, self.handle_destination_coords)       # Destination from control node
        rospy.Subscriber('/controller_info', String, self.controller_info_func)                     # Package info callback
        rospy.Subscriber('/enable_setpoint_pub_mapping_script', Float32, self.enable_setpoint_pub)  # Used to stop setpoint publisher when marker detector is active
        rospy.Subscriber('/edrone/imu/data', Imu, self.imu_callback)                                # IMU sensor data
        rospy.Subscriber('/edrone/gps_velocity', Vector3Stamped, self.velocity_callback)            # GPS velocity subscriber

#######################################################################################################################
# Callback and publisher functions

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

        self.drone_orientation_quaternion[0] = msg.orientation.x
        self.drone_orientation_quaternion[1] = msg.orientation.y
        self.drone_orientation_quaternion[2] = msg.orientation.z
        self.drone_orientation_quaternion[3] = msg.orientation.w

        (self.drone_orientation_euler[0], self.drone_orientation_euler[1],
         self.drone_orientation_euler[2]) = tf.transformations.euler_from_quaternion(
            [self.drone_orientation_quaternion[0], self.drone_orientation_quaternion[1],
             self.drone_orientation_quaternion[2], self.drone_orientation_quaternion[3]])

    def enable_setpoint_pub(self,enable):
        '''
		Purpose:
		---
		Enable or disable setpoint publishing. Enabled in all stages except during marker detetion, where the position controlller directly takes control
        of the setpoints to the drone based on marker detection data.

		Input Arguments:
		---
		`enable` :  [ Float32 ]
			Subscribes to enable/disable setpoint publishing from this script to position controller node

		Returns:
		---
		None

		Example call:
		---
		Called automatically by Subscriber of /enable_setpoint_pub_mapping_script
		'''

        self.enable_setpoint_pub_flag = enable.data

    # Callback for package information
    def controller_info_func(self, data):
        '''
		Purpose:
		---
		Subscribe to controller info comprising of the current package index, stage number and enable/disable path planner algorithm.
        corresponding flags are set based on the received info data. As more than 10 packages are there, letters from A to H have been used
        to indicate packages 10 to 18.
        If path planner needs to be enabled, the flag must be set to one along with a stage change. This ensures that the algorithm for computing
        path runs only once in each navigation stage.

		Input Arguments:
		---
		`data` :  [ String ]
			Subscribes to controller info data from control node in format of <Current package index><Current stage index><Path palnenr enable/disable>

		Returns:
		---
		None

		Example call:
		---
		Called automatically by Subscriber of /controller_info
		'''

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
        '''
		Purpose:
		---
		Publishes the required setpoint coordinates to the position controller node.

		Input Arguments:
		---
		None

		Returns:
		---
		None

		Example call:
		---
		setpoint_pub_func()
        Called in control_flow()
		'''
        self.setpoint_pub_object.x = float(self.position_setpoint[0])
        self.setpoint_pub_object.y = float(self.position_setpoint[1])
        self.setpoint_pub_object.z = float(self.position_setpoint[2])
        self.setpoint_publisher.publish(self.setpoint_pub_object)

    # Planar sensor reading callback
    def range_top(self, range_top_data):
        '''
		Purpose:
		---
		Subscribes to the sensor readings of the drone from topic /edrone/range_finder_top

		Input Arguments:
		---
		`range_top_data` : ranges [ Float32 ]
            Contains the 5 sensor readings from the drone in planar direction and top

		Returns:
		---
		None

		Example call:
		---
        Called automatically by subscriber of /edrone/range_finder_top
		'''
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
        '''
		Purpose:
		---
		Subscribes to the sensor readings of the drone from topic /edrone/range_finder_top

		Input Arguments:
		---
		`range_bottom` : ranges [ Float32 ]
            Contains the sensor reading of the bottom sensor of drone
		Returns:
		---
		None

		Example call:
		---
        Called automatically by subscriber of /edrone/range_finder_bottom
		'''

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

        self.prev_drone_position = copy.copy(self.drone_position)
        self.drone_position[0] = gps.latitude
        self.drone_position[1] = gps.longitude
        self.drone_position[2] = gps.altitude
        self.drone_cell_index = self.find_cell_index([self.drone_position[0], self.drone_position[1]])

        #print("maplist idx : ", self.map_list[self.drone_cell_index[0]][self.drone_cell_index[1]])

    # Store incoming destination data
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

    def velocity_callback(self, vel):
        '''
		Purpose:
		---
		Subscribes to the current drone velocity through Vector3Stamped message subscribed from /edrone/gps_velocity

		Input Arguments:
		---
		`vel` :  [ Vector3Stamped ]
			Holds drone velocity in all directions

		Returns:
		---
		None

		Example call:
		---
		Called automatically by Subscriber /edrone/gps_velocity
		'''

        # Callback for velocity of drone
        self.drone_velocity = [vel.vector.x, vel.vector.y, vel.vector.z]
        for i in range(2):
            if abs(self.drone_velocity[i]) < 0.1:
                self.travel_direction[i] = 0
            elif self.drone_velocity[i] > 0:
                self.travel_direction[i] = 1 # Right, Front
            else:
                self.travel_direction[i] = -1 # Left Back

        if self.travel_direction[0] == 0: # L/R
            if self.position_setpoint[0] > self.drone_position[0]:
                self.travel_direction[0] = 1
            else:
                self.travel_direction[0] = -1

        if self.travel_direction[1] == 0: # F/B
            if self.position_setpoint[1] > self.drone_position[1]:
                self.travel_direction[0] = -1
            else:
                self.travel_direction[0] = 1


############################################################################################################################
# Algorithm

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
		Called by file_read_func() in initial_setup_func()
		'''

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
        flag = self.package_window_check([self.pickup_coordinate_list[self.current_package_index][0],self.pickup_coordinate_list[self.current_package_index][1],0], [0.45,0.45,float('inf')])
		'''

        if(abs(self.drone_position[0]-coords[0])<(error[0]/self.meter_conv[0])):
            if(abs(self.drone_position[1]-coords[1])<(error[1]/self.meter_conv[1])):
                if(abs(self.drone_position[2]-coords[2])<(error[2]/self.meter_conv[2])):
                    return 1
        else:
            return 0

    # Finding index values of cell given a coordinates into self.position_coordinates
    def find_cell_index(self, position_coordinates):
        '''
		Purpose:
		---
		Finds the cell index wrt the grid computed from the passed coordinates data

		Input Arguments:
		---
		`position_coordinates`: list [ Float32 ]
            Coordinates whose cell index has to be found wrt the grid computed

		Returns:
		---
		'current_index' : list [ int, int ]
            i,j format specifies the computed cell index of the passed coordinates

		Example call:
		---
        [a,b] = self.find_cell_index(self.pickup_coordinate_list[0],self.pickup_coordinate_list[1])
		'''
        current_index = [0, 0]
        try:
            current_index[1] = int(math.floor(((position_coordinates[0] - self.boundaries_final[0][0]) / self.package_cell_size[0]) - 0))
            current_index[0] = int(math.floor(((position_coordinates[1] - self.boundaries_final[0][1]) / self.package_cell_size[1]) - 0))

            #To make sure the computed cell indices are within the range of the arena
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
        '''
		Purpose:
		---
		Computes the boundaries of the grid using the package pickup/drop data and starting drone position. A small offset in outward direction
        is als given to make sure the drone doesnt go out of the arena causing a 'list index out of range error'.
        The grid for mapping requires proper boundaries set (same as a rectangle with top left and bottom right coordinates mentioned).
        Also converts into a grid map based on grid cell size and restructures the grid to multiples of cell size (also maintaining a decent offset)

		Input Arguments:
		---
		None

		Returns:
		---
		None

		Example call:
		---
        boundaries_compute_func()
        Called in initial_setup_func()
		'''

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

        #boundaries consist of lat,long of top left point and bottom right point of the region of interest in the arena
        self.boundaries_computed = [[min_latitude, min_longitude], [max_latitude, max_longitude]]
        #boundaries offset
        padding = 20

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
        '''
		Purpose:
		---
		Initialise the map image for representation and debugging purposes. Creates a scaled down map with each cell represented in a particular colour
        based on its value from map and key associated with it.

		Input Arguments:
		---
		None

		Returns:
		---
		None

		Example call:
		---
        map_img_initialize()
        Called in initial_setup_func()
		'''

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
        '''
		Purpose:
		---
		A particular format has been used for storing the altitude in the map list.
        0 - 100 represents absolute altitude meaning, they have been obtained from bottom sensor
        101 - 199 represents safe flying height which can be updated if found lower. This is obtained using the planar sensors of drone.
        200 represents obstacle
        500 represents unmapped region (default value)

        This function returns the absolute height of a particular cell passed and also a corresponding character associated with the heiht using the key above.

		Input Arguments:
		---
		`data` : [ Float32 ]
            Value or altitude to be checked

		Returns:
		---
		`[cell_info_data, cell_height_data]` : [ int, int ]
            Contains the absoulte height or altitude stored in the particular cell of the map passed

		Example call:
		---
        [a,b] = self.cell_height_data([1,2])
		'''

        cell_info_data = 'X'
        cell_height_data = 200
        if(data == 500.0):
            cell_info_data = 'X'  # unmapped
            cell_height_data = 500
        if(data == 200.0):
            cell_info_data = 'O'  # obstacle
            cell_height_data = 200
        elif (data>=100.0 and data<200.0):
            cell_info_data = 'S'   # safe flying altitude
            cell_height_data = data - 100
        elif (data<100.0):
            cell_info_data = 'A'  # got from bottom sensor-absolute height
            cell_height_data = data
        return [cell_info_data,cell_height_data]

    # Updating the map image's individual cells
    def map_img_cell_update(self, cell, sign):
        '''
		Purpose:
		---
		Updates the cell color in the map image used based on the altitude value stored and the key assumed.

		Input Arguments:
		---
		`cell` : list [ int, int ]
            The cell index whose map cell needs to be updated

        `sign` : [ Float32 ]
            The value or altitude of a particular cell in the map list

		Returns:
		---
		None

		Example call:
		---
        self.map_img_cell_update([1,2],self.map_list[1][2])
		'''

        # Key: WHITE - Absolute data; RED - Unmapped; BLACK - Safe flying height; GREEN - Obstacle; BLUE - A* path; YELLOW - Smoothened path;
        cell_info = ['X',200]
        cell_info = self.cell_height_data(sign)
        if cell in self.found_path or cell in self.smoothened_path:
            if cell in self.found_path and cell not in self.smoothened_path:
                color = (255,0,0)
            else:
                color = (0,255,255)
        else:
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
        '''
		Purpose:
		---
		UPdates all cells of the map image each time

		Input Arguments:
		---
		None

		Returns:
		---
		None

		Example call:
		---
        self.map_img_list_update()
		'''

        for i in range(int(self.grid_width)):
            for j in range(int(self.grid_length)):
                self.map_img_cell_update([i, j], self.map_list[i][j])

        cv2.imwrite(self.img_loc, self.map_img)

    def mapping_height_check(self):
        '''
		Purpose:
		---
		Used to enable or disable mapping algorithm based on height of drone and the reference height computed based on starting and ending altitides
        When the drone is within a particular window wrt starting or ending coordinates, mapping is disabled so that false readings are avoided
        from getting mapped.

		Input Arguments:
		---
		None

		Returns:
		---
		`True` or `False`: [bool]

		Example call:
		---
        self.mapping_height_check()
		'''

        window_check1 = self.package_window_check(self.pickup_coordinate_list[self.current_package_index],[5,5,float('inf')])
        window_check2 = self.package_window_check(self.drop_coordinate_list[self.current_package_index],[5,5,float('inf')])

        #checking window wrt starting coordinate
        if(window_check1 == True):
            if(self.drone_position[2]>self.pickup_coordinate_list[self.current_package_index][2]+10):
                return True
            else:
                return False
        #checking window wrt destination coordinate
        elif(window_check2 == True):
            if(self.drone_position[2]>self.drop_coordinate_list[self.current_package_index][2]+10):
                return True
            else:
                return False

        else:
            return True

    def obs_validity(self):
        '''
		Purpose:
		---
		Drone's orientation is used to find whether the sensor readings obtained are proper or have inaccuracies due to tilt of drone.

		Input Arguments:
		---
		None

		Returns:
		---
		`True` or `False`: [bool]

		Example call:
		---
		self.obs_validity()
		'''

        # sensor data is [front, right, back, left]

        lat_diff = self.drone_position[0] - self.prev_drone_position[0]
        long_diff = self.drone_position[1] - self.prev_drone_position[1]
        sensor_val_change = [0,0,0,0]
        for i in range(0,4):
            sensor_val_change[i] = self.map_sensor_data[i] - self.prev_map_sensor_data[i]
            if self.map_sensor_data[i] == self.prev_map_sensor_data[i]:
                sensor_val_change[i] = 25.1

        self.sensor_validity = [0, 0, 0, 0]
        #Trigonometric function of angle
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

    # Increase altitude on obstacle
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
        '''
		Purpose:
		---
		Initial setup of the node including computing all required pickup drop coordinates and reading the sequenced_manifest file, computing the map boundaries and diving the same into cells.
        Also initialises the map with default altitude values.
        Map image for representation and debuggin purposes is also invoked here.

		Input Arguments:
		---
		None

		Returns:
		---
		None

		Example call:
		---
		initial_setup_func()
		'''

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
    # def map_update_func(self, new_data, mode):
    #     '''
    # 	Purpose:
    # 	---
    #     updates the required cells between the drone and the obstacle/free space (along with 2 lines of padding) with the safe flying height based on conditions.
    #     Also maps obstacles with padding if detected.
    #     Based on sensor location [mode] and relative position between drone and sensor reading cell, this function works
    #
    # 	Input Arguments:
    # 	---
    # 	`new_data` : list [ int ]
    #         The cell till which free space was recorded by the sensors of drone
    #
    #     `mode` : [ int ]
    #         Indicates whether the sensor was front/back/left/right so that the reference can be taken accordingly
    #
    # 	Returns:
    # 	---
    # 	None
    #
    # 	Example call:
    # 	---
    # 	map_update_func(self.current_cell_index, 1)
    # 	'''
    #
    #     cell_info = ['X',200]
    #     if (self.mapping_enable_flag == False):
    #         return
    #     #based on mode i.e direction of sensor, the number of cells of free space is found wrt the arena grid map
    #     #and the drone's altitude is compared with the existing values.
    #     #If found less, the map list is updated along with a pre-defined padding value.
    #     #This is continously ruun and the map is more optimised through each run at a lower altitude.
    #     #The cells with absolute altitude are not over-written in this process.
    #     #The cells with obstacles are over-written only when the direct free-space is detected, and not the padding.
    #     try:
    #         if(mode == 0):
    #             if(new_data[1]>=self.drone_cell_index[1]):
    #                 for i in range(int(self.drone_cell_index[1]), int(new_data[1])):
    #                     cell_info = self.cell_height_data(self.map_list[self.drone_cell_index[0]][i])
    #                     if(cell_info[0] != 'A'):
    #                         if(cell_info[1] - self.padding_height > self.drone_position[2]) and ([self.drone_cell_index[0], i] not in self.found_path):
    #                             for j in range(3):
    #                                 cell_info = self.cell_height_data(self.map_list[self.drone_cell_index[0]-1+j][i])
    #                                 if(cell_info[0]!='A'):
    #                                     if(cell_info[0]!='O' or j==2 ):
    #                                         if [self.drone_cell_index[0]-1+j, i] not in self.found_path:
    #                                             self.map_list[self.drone_cell_index[0]-1+j][i] = 100 + self.drone_position[2]
    #
    #                 for i in range(int(new_data[1])-1, int(new_data[1])+self.obstacle_cell_constant[0]):
    #                     if (i < self.grid_length and self.map_sensor_data[1] < 25.0):
    #                         cell_info = self.cell_height_data(self.map_list[self.drone_cell_index[0]][i])
    #                         if(cell_info[0]=='X'):
    #                             self.map_list[self.drone_cell_index[0]][i] = 200
    #
    #             else:
    #                 for i in range(int(new_data[1]+1), int(self.drone_cell_index[1])+1):
    #                     cell_info = self.cell_height_data(self.map_list[self.drone_cell_index[0]][i])
    #                     if(cell_info[0] != 'A'):
    #                         if(cell_info[1] - self.padding_height > self.drone_position[2]) and ([self.drone_cell_index[0], i] not in self.found_path):
    #                             for j in range(3):
    #                                 cell_info = self.cell_height_data(self.map_list[self.drone_cell_index[0]-1+j][i])
    #                                 if(cell_info[0]!='A'):
    #                                     if(cell_info[0]!='O' or j==2 ):
    #                                         if [self.drone_cell_index[0] - 1 + j, i] not in self.found_path:
    #                                             self.map_list[self.drone_cell_index[0]-1+j][i] = 100 + self.drone_position[2]
    #
    #                 for i in range(int(new_data[1])-self.obstacle_cell_constant[0], int(new_data[1])+1):
    #                     if (i >= 0 and self.map_sensor_data[3] < 25.0):
    #                         cell_info = self.cell_height_data(self.map_list[self.drone_cell_index[0]][i])
    #                         if(cell_info[0]=='X'):
    #                             self.map_list[self.drone_cell_index[0]][i] = 200
    #
    #         else:
    #             if(new_data[0]>=self.drone_cell_index[0]):
    #                 for i in range(int(self.drone_cell_index[0]), int(new_data[0])):
    #                     cell_info = self.cell_height_data(self.map_list[i][self.drone_cell_index[1]])
    #                     if(cell_info[0] != 'A'):
    #                         if(cell_info[1] - self.padding_height > self.drone_position[2]) and ([i, self.drone_cell_index[1]] not in self.found_path):
    #                             for j in range(3):
    #                                 cell_info = self.cell_height_data(self.map_list[i][self.drone_cell_index[1]-1+j])
    #                                 if(cell_info[0]!='A'):
    #                                     if(cell_info[0]!='O' or j==2 ):
    #                                         if [i, self.drone_cell_index[1] - 1 + j] not in self.found_path:
    #                                             self.map_list[i][self.drone_cell_index[1]-1+j] = 100 + self.drone_position[2]
    #
    #                 for i in range(int(new_data[0])-1, int(new_data[0])+self.obstacle_cell_constant[0]):
    #                     if (i < self.grid_width and self.map_sensor_data[2] < 25.0):
    #                         cell_info = self.cell_height_data(self.map_list[i][self.drone_cell_index[1]])
    #                         if(cell_info[0]=='X'):
    #                             self.map_list[i][self.drone_cell_index[1]] = 200
    #
    #             else:
    #                 for i in range(int(new_data[0]), int(self.drone_cell_index[0])+1):
    #                     cell_info = self.cell_height_data(self.map_list[i][self.drone_cell_index[1]])
    #                     if(cell_info[0] != 'A'):
    #                         if(cell_info[1] -self.padding_height > self.drone_position[2]) and ([i, self.drone_cell_index[1]] not in self.found_path):
    #                             for j in range(3):
    #                                 cell_info = self.cell_height_data(self.map_list[i][self.drone_cell_index[1]-1+j])
    #                                 if(cell_info[0]!='A'):
    #                                     if(cell_info[0]!='O' or j==2 ):
    #                                         if [i, self.drone_cell_index[1] - 1 + j] not in self.found_path:
    #                                             self.map_list[i][self.drone_cell_index[1]-1+j] = 100 + self.drone_position[2]
    #
    #                 for i in range(int(new_data[0])-self.obstacle_cell_constant[0], int(new_data[0])+1):
    #                     if (i >= 0 and self.map_sensor_data[0] < 25.0):
    #                         cell_info = self.cell_height_data(self.map_list[i][self.drone_cell_index[1]])
    #                         if(cell_info[0]=='X'):
    #                             self.map_list[i][self.drone_cell_index[1]] = 200
    #
    #     except IndexError:
    #         pass
    #         #print("Indexerror")

        # Map list updation, mode= 1-front/back, mode=0-left/right
    def map_update_func(self, new_data, mode):
        '''
        Purpose:
        ---
        updates the required cells between the drone and the obstacle/free space (along with 2 lines of padding) with the safe flying height based on conditions.
        Also maps obstacles with padding if detected.
        Based on sensor location [mode] and relative position between drone and sensor reading cell, this function works

        Input Arguments:
        ---
        `new_data` : list [ int ]
            The cell till which free space was recorded by the sensors of drone

        `mode` : [ int ]
            Indicates whether the sensor was front/back/left/right so that the reference can be taken accordingly

        Returns:
        ---
        None

        Example call:
        ---
        map_update_func(self.current_cell_index, 1)
        '''

        cell_info = ['X', 200]
        if (self.mapping_enable_flag == False):
            return
        ####change
        # try:
        sensor_offset = 0.5
        if (mode == 0):
            if (new_data[1] >= self.drone_cell_index[1]):
                for i in range(int(self.drone_cell_index[1]), int(new_data[1])):
                    cell_info = self.cell_height_data(self.map_list[self.drone_cell_index[0]][i])
                    if (cell_info[0] != 'A'):
                        if (cell_info[1] - self.padding_height > self.drone_position[2] + sensor_offset) and (
                                [self.drone_cell_index[0], i] not in self.found_path):
                            for j in range(3):
                                cell_info = self.cell_height_data(
                                    self.map_list[self.drone_cell_index[0] - 1 + j][i])
                                if (cell_info[0] != 'A'):
                                    if (cell_info[0] != 'O' or j == 2):
                                        if [self.drone_cell_index[0] - 1 + j, i] not in self.found_path:
                                            self.map_list[self.drone_cell_index[0] - 1 + j][i] = 100 + self.drone_position[2] + sensor_offset

                for i in range(int(new_data[1]) - 1, int(new_data[1]) + self.obstacle_cell_constant[0]):
                    if (i < self.grid_length and self.map_sensor_data[1] < 25.0):
                        cell_info = self.cell_height_data(self.map_list[self.drone_cell_index[0]][i])
                        if (cell_info[0] == 'X'):
                            self.map_list[self.drone_cell_index[0]][i] = 200
                        else:
                            if (
                            self.package_window_check([0, 0, self.map_list[self.drone_cell_index[0] - 1][i]],
                                                      [float('inf'), float('inf'), 2])):
                                self.map_list[self.drone_cell_index[0]][i] = 200
                            # elif (
                            # self.package_window_check([0, 0, self.map_list[self.drone_cell_index[0] + 1][i]],
                            #                           [float('inf'), float('inf'), 2])):
                            #     self.map_list[self.drone_cell_index[0]][i] = 200

            else:
                for i in range(int(new_data[1] + 1), int(self.drone_cell_index[1]) + 1):
                    cell_info = self.cell_height_data(self.map_list[self.drone_cell_index[0]][i])
                    if (cell_info[0] != 'A'):
                        if (cell_info[1] - self.padding_height > self.drone_position[2] + sensor_offset) and (
                                [self.drone_cell_index[0], i] not in self.found_path):
                            for j in range(3):
                                cell_info = self.cell_height_data(
                                    self.map_list[self.drone_cell_index[0] - 1 + j][i])
                                if (cell_info[0] != 'A'):
                                    if (cell_info[0] != 'O' or j == 2):
                                        if [self.drone_cell_index[0] - 1 + j, i] not in self.found_path:
                                            self.map_list[self.drone_cell_index[0] - 1 + j][i] = 100 + self.drone_position[2] + sensor_offset

                for i in range(int(new_data[1]) - self.obstacle_cell_constant[0], int(new_data[1]) + 1):
                    if (i >= 0 and self.map_sensor_data[3] < 25.0):
                        cell_info = self.cell_height_data(self.map_list[self.drone_cell_index[0]][i])
                        if (cell_info[0] == 'X'):
                            self.map_list[self.drone_cell_index[0]][i] = 200
                        else:
                            if (
                            self.package_window_check([0, 0, self.map_list[self.drone_cell_index[0] - 1][i]],
                                                      [float('inf'), float('inf'), 2])):
                                self.map_list[self.drone_cell_index[0]][i] = 200
                            # elif (
                            # self.package_window_check([0, 0, self.map_list[self.drone_cell_index[0] + 1][i]],
                            #                           [float('inf'), float('inf'), 2])):
                            #     self.map_list[self.drone_cell_index[0]][i] = 200

        else:
            if (new_data[0] >= self.drone_cell_index[0]):
                for i in range(int(self.drone_cell_index[0]), int(new_data[0])):
                    cell_info = self.cell_height_data(self.map_list[i][self.drone_cell_index[1]])
                    if (cell_info[0] != 'A'):
                        if (cell_info[1] - self.padding_height > self.drone_position[2] + sensor_offset) and (
                                [i, self.drone_cell_index[1]] not in self.found_path):
                            for j in range(3):
                                cell_info = self.cell_height_data(
                                    self.map_list[i][self.drone_cell_index[1] - 1 + j])
                                if (cell_info[0] != 'A'):
                                    if (cell_info[0] != 'O' or j == 2):
                                        if [i, self.drone_cell_index[1] - 1 + j] not in self.found_path:
                                            self.map_list[i][self.drone_cell_index[1] - 1 + j] = 100 + self.drone_position[2] + sensor_offset

                for i in range(int(new_data[0]) - 1, int(new_data[0]) + self.obstacle_cell_constant[0]):
                    if (i < self.grid_width and self.map_sensor_data[2] < 25.0):
                        cell_info = self.cell_height_data(self.map_list[i][self.drone_cell_index[1]])
                        if (cell_info[0] == 'X'):
                            self.map_list[i][self.drone_cell_index[1]] = 200
                        else:
                            if (
                            self.package_window_check([0, 0, self.map_list[i][self.drone_cell_index[1] - 1]],
                                                      [float('inf'), float('inf'), 2])):
                                self.map_list[i][self.drone_cell_index[1]] = 200
                            # elif (
                            # self.package_window_check([0, 0, self.map_list[i][self.drone_cell_index[1] + 1]],
                            #                           [float('inf'), float('inf'), 2])):
                            #     self.map_list[i][self.drone_cell_index[1]] = 200

            else:
                for i in range(int(new_data[0]), int(self.drone_cell_index[0]) + 1):
                    cell_info = self.cell_height_data(self.map_list[i][self.drone_cell_index[1]])
                    if (cell_info[0] != 'A'):
                        if (cell_info[1] - self.padding_height > self.drone_position[2] + sensor_offset) and (
                                [i, self.drone_cell_index[1]] not in self.found_path):
                            for j in range(3):
                                cell_info = self.cell_height_data(
                                    self.map_list[i][self.drone_cell_index[1] - 1 + j])
                                if (cell_info[0] != 'A'):
                                    if (cell_info[0] != 'O' or j == 2):
                                        if [i, self.drone_cell_index[1] - 1 + j] not in self.found_path:
                                            self.map_list[i][self.drone_cell_index[1] - 1 + j] = 100 + self.drone_position[2] + sensor_offset

                for i in range(int(new_data[0]) - self.obstacle_cell_constant[0], int(new_data[0]) + 1):
                    if (i >= 0 and self.map_sensor_data[0] < 25.0):
                        cell_info = self.cell_height_data(self.map_list[i][self.drone_cell_index[1]])
                        if (cell_info[0] == 'X'):
                            self.map_list[i][self.drone_cell_index[1]] = 200
                        else:
                            if (
                            self.package_window_check([0, 0, self.map_list[i][self.drone_cell_index[1] - 1]],
                                                      [float('inf'), float('inf'), 2])):
                                self.map_list[i][self.drone_cell_index[1]] = 200
                            # elif (
                            # self.package_window_check([0, 0, self.map_list[i][self.drone_cell_index[1] + 1]],
                            #                           [float('inf'), float('inf'), 2])):
                            #     self.map_list[i][self.drone_cell_index[1]] = 200

        # except IndexError:
        #     pass
            # print("Indexerror")

    # Main map generation functions
    def mapping_func(self):
        '''
    	Purpose:
    	---
    	Updates map continously based on sensor readings of drone. The corresponding cell in map is computed and the safe flying height is updated based on conditions.
        Updates absolute heights also using bottom sensor data.

    	Input Arguments:
    	---
    	None

    	Returns:
    	---
    	None

    	Example call:
    	---
    	mapping_func()
    	'''

        self.mapping_enable_flag = self.mapping_height_check()
        # sensor data is [front, right, back, left]
        sensor_deg_data = [[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]]

        #The sensor readings is converted into degrees and the corresponding cell index is computed w.r.t the main grid map
        #This cell index data is passed on to map_update_func() for map list updation.
        # Mapping front
        sensor_deg_data[0][1] = self.drone_position[1] - (self.map_sensor_data[0]*math.cos(self.drone_orientation_euler[0])/self.meter_conv[1])
        sensor_deg_data[0][0] = self.drone_position[0]
        self.current_cell_index = self.find_cell_index(sensor_deg_data[0])

        self.map_update_func(self.current_cell_index, 1)

        # Mapping right
        sensor_deg_data[1][1] = self.drone_position[1]
        sensor_deg_data[1][0] = self.drone_position[0] + (self.map_sensor_data[1]*math.cos(self.drone_orientation_euler[1])/self.meter_conv[0])
        self.current_cell_index = self.find_cell_index(sensor_deg_data[1])


        self.map_update_func(self.current_cell_index, 0)

        # Mapping back
        sensor_deg_data[2][1] = self.drone_position[1] + (self.map_sensor_data[2]*math.cos(self.drone_orientation_euler[0])/self.meter_conv[1])
        sensor_deg_data[2][0] = self.drone_position[0]
        self.current_cell_index = self.find_cell_index(sensor_deg_data[2])


        self.map_update_func(self.current_cell_index, 1)

        # Mapping left
        sensor_deg_data[3][1] = self.drone_position[1]
        sensor_deg_data[3][0] = self.drone_position[0] - (self.map_sensor_data[3]*math.cos(self.drone_orientation_euler[0])/self.meter_conv[0])
        self.current_cell_index = self.find_cell_index(sensor_deg_data[3])


        self.map_update_func(self.current_cell_index, 0)

        # Mapping bottom
        #The bottom sensor picks up the exact altitude of buildings and therefore is called absolute height/altitude
        #This is not over-written by safe flying height data but will be overwritten with the same sensor itself due to 
        #sensor issues.
        #A padding of 8 cells around the main cell is given because all macro objects/obstacles are greater than 1.5 sq. m
        cell_info = ['X', 200]
        cell_info = self.cell_height_data(self.map_list[self.drone_cell_index[0]][self.drone_cell_index[1]])
        if(self.map_sensor_data[4]>0.0 and self.map_sensor_data[4]<50.0): # 25

            for i in range(3):
                for j in range(3):
                    cell_info = self.cell_height_data(self.map_list[self.drone_cell_index[0]-1+i][self.drone_cell_index[1]-1+j])
                    if cell_info[0] != 'A': # and (i == 1 and j == 1): ######
                        self.map_list[self.drone_cell_index[0]-1+i][self.drone_cell_index[1]-1+j] = self.drone_position[2] - self.map_sensor_data[4] + self.padding_height

        elif((self.map_sensor_data[4] == 50.0 or self.map_sensor_data[4] == 0)): # and cell_info[0]!='A'): # 25
            if(self.drone_position[2] - self.map_sensor_data[4] < cell_info[1] - self.padding_height):
                self.map_list[self.drone_cell_index[0]][self.drone_cell_index[1]] = 100 + self.drone_position[2] - self.map_sensor_data[4] + self.padding_height

        self.map_img_list_update()

    # Functions used for path planning
    def find_next_nodes(self, node):
        '''
        Purpose:
        ---
        Finds the index and cost value of the neighbouring nodes (8 nodes surrounding one node).
        For example, for vertical and horizontal neighbouring nodes, cost=1
        Diagonal, cost= 1.41

        Input Arguments:
        ---
        'node` :  [list]
            The node for which we want to calculate the cost to neighbouring nodes (list form)

        Returns:
        ---
        'next_nodes` :  [list]
             returns the indices of the neighbouring nodes of the given input node.


        Example call:
        ---
        find_next_nodes([3,5])
        '''


        next_nodes = []
        gn = []
        '''
        Given the index of a node, the cost of neighbouring nodes are computed.
        If there is an obstacle in the path, it will not append any value (remains at infinity)
        '''

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



        return next_nodes

    def get_dist(self, node1, node2):


        '''
        Purpose:
        ---
        Finds the heuristic cost (Euclidian Distance between two given nodes).
        Basically this is updated in the h_cost_list

        Input Arguments:
        ---
        'node1` :  [list]
            Index of the first node

        'node' :  [list]
            Index of the second node

        Returns:
        ---
        < name of 1st return argument >` :  [ < type of 1st return argument > ]
            < one-line description of 1st return argument >

        < name of 2nd return argument >` :  [ < type of 2nd return argument > ]
            < one-line description of 2nd return argument >

        Example call:
        ---
        get_dist(start_node,end_node)
        '''

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

        '''
        Purpose:
        ---
        To find the A* path (inverted) using parent_nodes list.
        Loops from the target node to the start node via parent node.
        Appends the value to found_path

        Input Arguments:
        ---
        NONE

        Returns:
        ---
        NONE

        Example call:
        ---
        retrace()
        '''

        path = []
        curr_node = self.target_node

        while curr_node != self.start_node:
            path.append(curr_node)
            curr_node = self.parent_list[str(curr_node)]
        path.append(self.start_node)
        self.found_path = path[::-1]

        self.smoothened_path = self.smoothing(path)  # returns the path indices
        self.a_star_flag = False


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

        '''
        Purpose:
        ---
        To implement A* algorithm for path planning between two given nodes from a map.
        Minimizes the f-cost.
        Combination of Djikshtra and greedy-search algorithm.

        Input Arguments:
        ---
        'start_node` :  [list]
            index of the first node

        'target_node` :  [list]
            index of the second node

        Returns:
        ---
        NONE

        Example call:
        ---
        a_star([2,3],[5,6])
        '''

        # print(" A-STAR STARTED")
        self.clean_path_on_map()  # for cleaning the path from the map

        
        self.start_node = start_node
        self.target_node = target_node
        self.grid = self.map_list
        self.grid_size = [self.grid_width - 1, self.grid_length - 1]
        self.open_list = [self.start_node]  
        self.closed_list = []  

        self.g_cost_list = [0]  
        self.h_cost_list = [self.get_dist(self.start_node, self.target_node)]  
        self.f_cost_list = [0 + self.get_dist(self.start_node, self.target_node)]  
        self.parent_list = {}  
        self.current = []  

        self.found_path = []  
        self.smoothened_path = []  

        while len(self.open_list) > 0:
            # for calculating the node with min fn
            self.current = self.open_list[0]
            min_id = 0
            for i in range(1, len(self.f_cost_list)):
                if (self.f_cost_list[i] < self.f_cost_list[min_id]) or (self.f_cost_list[i] == self.f_cost_list[min_id] and self.h_cost_list[i] < self.h_cost_list[min_id]):
                    min_id = i

            self.current = self.open_list[min_id]
            
            self.open_list.pop(min_id)
            curr_f_cost = self.f_cost_list.pop(min_id)
            curr_g_cost = self.g_cost_list.pop(min_id)
            curr_h_cost = self.h_cost_list.pop(min_id)
            self.closed_list.append(self.current)

            if self.current == self.target_node:
                self.retrace()
                break
            #finding costs of neighbouring nodes
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

                
        print("A* algorithm done")

    # Function for getting center coordinate given the node index as a list
    def get_center_coordinate(self, node):
        '''
        Purpose:
        ---
        Function for getting the (x,y) coordinates of te centre of a given node.
        This coordinates will be published to the drone.
        Centre coordinate is used because of convenience.

        Input Arguments:
        ---
        'node` :  [list]
            indices of the node whose centre coordinate is to be found


        Returns:
        ---
        '[x,y]` : [list]
            returns the (x,y) coordinates in list form.


        Example call:
        ---
        get_center_coordinate([2,3])
        '''

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
    def smoothing(self, computed_path):
        '''
        Purpose:
        ---
        Function that will smoothen the path found by A* function to achieve realistic pathfinding.
        PART 1 of smoothening- removing collinear points
        PART 2- Further smoothening by sampling the line between 2 points.
        Out of 3 given points, it will sample the first and last point. If no obstacle in the middle,
        it will remove the second point.

        Input Arguments:
        ---
        'computed_path` : [list]
            path found by A* algorithm.

        Returns:
        ---
        'path` :  [list]
            Indices of the smoothened path



        Example call:
        ---
        <smoothing(found_path) (or) smoothing([[1,2],[2,3],[5,6],[8,10]])>
        '''

        path = list(computed_path)  # dummy variable for storing the path and working on it

        # PART-1 of smoothening- Removing collinear points (By comparing slopes)
         # Number of coordinates in the initial path
        path_length = len(path) 
        
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
                 # remove the node from the path if slopes are equal
                path.pop(i + 1) 
            else:
                # Don't remove it. Go to the next node in the path
                current_node = path[i + 1]  

           

        # PART-2 of smoothening- Further smoothening to remove zig zaggedness
        # initially it is the starting point in the path
        checkpoint = path[0]  
        # initializing it with the next node in the path
        current_point = path[1]  
        # End node of the path
        end_point = path[len(path) - 1]  

        while (current_point != end_point):
            # The node next to current_point
            current_point_next = path[path.index(current_point) + 1]  
            
            if self.walkable(checkpoint, current_point_next):
                #if the sampling function returns True
                temp = current_point
                current_point = current_point_next
                # remove from the list
                path.remove(temp)  
                
            else:
                checkpoint = current_point
                current_point = current_point_next
        #reversing the list- start point to end point
        path = path[::-1]
        # print("Smoothing done")
        return path

    # Walkable function- sampling points along a line from point A to B. To be used in smoothening function
    def walkable(self, node_a, node_b):
        '''
        Purpose:
        ---
        To sample the line connecting two nodes.
        Used in smoothing function.
        Given two nodes, it will sample all points that lie in the line connecting them.If there is an obstacle,
        it will return False, otherwise True
        Input Arguments:
        ---
        'node_a` : [list]
            index of the first node

        'node_b` : [list]
            index of the next node

        Returns:
        ---
        'flag` :  [ boolean]
            True if no obstacle, false otherwise

        Example call:
        ---
        walkable([1,2],[3,4])
        '''

        # print(node_a, node_b)
        # (x,y) of node a
        coor_a = self.get_center_coordinate(node_a)  
        # (x,y) of node b
        coor_b = self.get_center_coordinate(node_b)  
        #This code snippet is for generalizing the sampling algorithm
        if coor_a[0] < coor_b[0]:
            t = coor_a
            coor_a = coor_b
            coor_b = t
        # Slope i.e tan(theta)
        tan = ((coor_b[1] - coor_a[1]) * self.meter_conv[1]) / (
                (coor_b[0] - coor_a[0]) * self.meter_conv[0])  
        theta = math.atan(tan)
        #Finding sine and cosine. Based on this, the coordinates will be updated.
        cosgree = math.cos(theta)
        singree = math.sin(theta)
        #finding the distance between the points to be sampled
        distance = math.sqrt(math.pow((coor_b[1] - coor_a[1]) * self.meter_conv[1], 2) + math.pow(
            (coor_b[0] - coor_a[0]) * self.meter_conv[0], 2))
        # Initial
        current_node = list(coor_a) 
        # initially Flag variable is set to True 
        flag = True  
        index = [0, 0] 

        #checking whether you have reached the end coordinate
        while math.sqrt(math.pow((coor_a[1] - current_node[1]) * self.meter_conv[1], 2) + math.pow(
                (coor_a[0] - current_node[0]) * self.meter_conv[0], 2)) < distance:
            #sampling points evenly spaced at a distance of 0.3 m
            current_node[0] -= 0.3 * cosgree / self.meter_conv[0]
            current_node[1] -= 0.3 * singree / self.meter_conv[1]
            index = self.find_cell_index(current_node)

            if self.map_list[index[0]][index[1]] == 200:
                #If obstacle detected, set Flag as false and break the loop
                flag = False
                
                break

        return flag


    def compute_height(self):
        '''
    	Purpose:
    	---
    	Altitude control algorithm. Main element of the designed algorithm is to maintain the safe flying altitude for each navigation, once the arena has been mapped.
        So based on each task stage, the altitude is controlled in this function. During pickup/drop, the drone descends until it picks up package/ is inside delivery threshold
        and thereby making the building height redundant for the same. During drone navigation, the map is refered to altitide values of path computed and the maximum altitude
        between the drone's position to destination. Thereby ensuring optimied, efficient and generalised solution to the problem statement.

    	Input Arguments:
    	---
    	None

    	Returns:
    	---
    	None

    	Example call:
    	---
    	compute_height()
    	'''

        calculated_height = 0

        if self.current_stage_index == 0:
            self.a_star_alt_index = 0
            self.pickup_pos_flag = True
            print("CH S 0")
            pass

        elif self.current_stage_index == 1:
            # ctr node pub - pickup h + 0.5

            self.prev_pickup_height = 1000
            self.pickup_down_flag = True

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
                calculated_height = self.destination_coords[2] + 0.5

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

            elif dist > 90:  # rough
                return 35
            else:
                return calculated_height + 2


        elif self.current_stage_index == 2:

            print("CH S 2")
            self.pickup_pos_flag = True
            self.a_star_alt_index = 0
            try:
                dist_x = (self.destination_coords[0] - self.drop_coordinate_list[self.current_package_index-1][0]) * self.meter_conv[0]
                dist_y = (self.destination_coords[1] - self.drop_coordinate_list[self.current_package_index-1][1]) * self.meter_conv[1]
                dist = math.sqrt(dist_x ** 2 + dist_y ** 2)
            except IndexError:
                dist = 7

            if dist > 70:
                wind = 0.05
                vel_wind = 0.1
            else:
                wind = 0.1
                vel_wind = 0.5

            vel_mag = math.sqrt(self.drone_velocity[0] ** 2 + self.drone_velocity[1] ** 2)

            calculated_height = min(self.destination_coords[2] + 0.3, self.prev_pickup_height)
            if (self.package_window_check(
                    [self.destination_coords[0], self.destination_coords[1], self.destination_coords[2]],
                    [wind, wind, float('inf')])) and vel_mag < vel_wind : # or self.pickup_down_flag == False:
                # print("Going True")
                window = 0
                while True:
                    if (self.package_window_check([self.destination_coords[0], self.destination_coords[1], self.destination_coords[2] - 3],
                            [float('inf'), float('inf'), 3.5 - window])):
                        calculated_height -= 0.02
                        window += 0.01
                        # print("window : ", window)
                    else:
                        self.pickup_down_flag = False
                        window = 0
                        break

            calculated_height = min(calculated_height, self.prev_pickup_height)
            self.prev_pickup_height = calculated_height + 0


        elif self.current_stage_index == 2.5:
            self.pickup_down_flag = True
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
                    calculated_height = self.destination_coords[2] - 3 #+ 0.4
                else:
                    calculated_height = self.destination_coords[2] #+ 7

            elif dist > 70: # rough
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
        #print("Computed height : ", calculated_height)
        return calculated_height



    # Bug Algorithm
    def bug_algo(self):
        if (self.bug_enable_flag == False):
            return

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



    def obs_check(self):
        # self.prev_drone_position
        # self.drone_position
        # self.map_sensor_data - [F,R,B,L, bottom]
        # self.prev_map_sensor_data
        # self.obs_estimate - [F,R,B,L] 'v'-vertical, 'h'-horizontal, 'n'-none

        self.update_obs_cord()

        for i in range(4):

            x_diff = (self.obs_cord[i][0] - self.prev_obs_cord[i][0])*self.meter_conv[0]
            y_diff = (self.obs_cord[i][1] - self.prev_obs_cord[i][1])*self.meter_conv[1]
            h = self.obs_cord[i][2] - self.prev_obs_cord[i][2]

            length = math.sqrt(x_diff**2 + y_diff**2 + h**2)

            if length != 0:
                if abs(h/length) > 0.5 and self.map_sensor_data[i] != 25:
                    self.obs_estimate[i] = 'v'
                elif self.map_sensor_data[i] == 25:
                    self.obs_estimate[i] = 'n'
                else:
                    self.obs_estimate[i] = 'h'
                # print("sin z : ", abs(h / length))
                #
                # print("sensor idx : ", i)
                # print("x : ", x_diff, "y : ", y_diff, "z : ", h)
                # print("estimate : ", self.obs_estimate[i])
                #print("estimate out : ", self.obs_estimate)



    def update_obs_cord(self):

        # updating previous obs_cord
        self.prev_obs_cord = [[0,0,0], [0,0,0], [0,0,0], [0,0,0]]
        for i in range(4):
            for j in range(3):
                self.prev_obs_cord[i][j] = self.obs_cord[i][j] + 0.0

        # Right sensor

        self.obs_cord[1][1] = self.drone_position[1]  # longitude is same as drone
        self.obs_cord[1][2] = self.drone_position[2] - self.map_sensor_data[1] * math.sin(self.drone_orientation_euler[1])
        self.obs_cord[1][0] = self.drone_position[0] + self.map_sensor_data[1] * abs(math.cos(self.drone_orientation_euler[1]))/ self.meter_conv[0]

        # Left sensor

        self.obs_cord[3][1] = self.drone_position[1]  # longitude is same as drone
        self.obs_cord[3][2] = self.drone_position[2] + self.map_sensor_data[3] * math.sin(self.drone_orientation_euler[1])
        self.obs_cord[3][0] = self.drone_position[0] - self.map_sensor_data[3] * abs(math.cos(self.drone_orientation_euler[1]))/self.meter_conv[0]

        # Front sensor

        self.obs_cord[0][0] = self.drone_position[0]  # lattitude is same as drone
        self.obs_cord[0][2] = self.drone_position[2] + self.map_sensor_data[0] * math.sin(self.drone_orientation_euler[0])
        self.obs_cord[0][1] = self.drone_position[1] - self.map_sensor_data[0] * abs(math.cos(self.drone_orientation_euler[0]))/self.meter_conv[1]

        # Back sensor

        self.obs_cord[2][0] = self.drone_position[0]  # lattitude is same as drone
        self.obs_cord[2][2] = self.drone_position[2] - self.map_sensor_data[2] * math.sin(self.drone_orientation_euler[0])
        self.obs_cord[2][1] = self.drone_position[1] + self.map_sensor_data[2] * abs(math.cos(self.drone_orientation_euler[0]))/self.meter_conv[1]

        # print("OBS_CORD : ")
        # for i in self.obs_cord:
        #     print(i)





    def behind_the_cur_point(self,val,depth):
        cur = self.bug_stop_setpoint
        return_list = [0,0,0]
        for i in range(2):
            if self.bug_setpoint[i] > cur[i]:
                return_list[i] = cur[i] - (val/self.meter_conv[i])
            else:
                return_list[i] = cur[i] + (val/ self.meter_conv[i])
        return_list[2] = cur[2] - depth

        return return_list




    def bug_boi(self):

        idx_dict = {'f': 0, 'r': 1, 'b': 2, 'l': 3}

        if self.bug_enable_flag == True:

            available_direction = ['f', 'r', 'b', 'l']

            if self.bug_started == False:

                self.bug_stage_flag = 1
                self.obs_direction = ''
                pos_dir = []
                # polynomial variables
                a = 0  #0.0005803577
                b = 0.03097692
                c = 1.1341 #1.258851
                d = -0.130558 + 4

                vel = math.sqrt(self.drone_velocity[0]**2 + self.drone_velocity[1]**2)
                safe_dist = a*(vel**3) + b*(vel**2) + c*vel + d
                if safe_dist > 25:
                    safe_dist = 25

                #print("safe dist : ", safe_dist)
                # range_find_top - Front, right, back, left thana?
                if self.travel_direction[0] == 1 and self.range_top_dist[1] < 25 and self.range_top_dist[1] > 1 and self.obs_estimate[1] == 'v':  # right
                    if self.range_top_dist[1] < safe_dist:
                        self.obs_direction = 'r1'
                        self.snapshot_a[0] = [self.drone_position[0], self.drone_position[1]]
                        self.snapshot_a[1] = [self.drone_velocity[0], self.drone_velocity[1]]
                    elif self.obs_estimate[1] == 'v':
                        self.obs_direction = "r0"

                    #available_direction.pop(1)
                elif self.travel_direction[0] == -1 and self.range_top_dist[3] < 25 and self.range_top_dist[3] > 1 and self.obs_estimate[3] == 'v':  # left
                    if self.range_top_dist[3] < safe_dist :
                        self.obs_direction = 'l1'
                        self.snapshot_a[0] = [self.drone_position[0], self.drone_position[1]]
                        self.snapshot_a[1] = [self.drone_velocity[0], self.drone_velocity[1]]
                    else:
                        self.obs_direction = 'l0'
                    #available_direction.pop(3)
                elif self.travel_direction[1] == 1 and self.range_top_dist[0] < 25 and self.range_top_dist[0] > 1 and self.obs_estimate[0] == 'v':  # Front
                    if self.range_top_dist[0] < safe_dist :
                        self.obs_direction = 'f1'
                        self.snapshot_a[0] = [self.drone_position[0], self.drone_position[1]]
                        self.snapshot_a[1] = [self.drone_velocity[0], self.drone_velocity[1]]
                    else:
                        self.obs_direction = 'f0'
                    #available_direction.pop(0)
                elif self.travel_direction[1] == -1 and self.range_top_dist[2] < 25 and self.range_top_dist[2] > 1 and self.obs_estimate[2] == 'v':  # Back
                    if self.range_top_dist[2] < safe_dist:
                        self.obs_direction = 'b1'
                        self.snapshot_a[0] = [self.drone_position[0], self.drone_position[1]]
                        self.snapshot_a[1] = [self.drone_velocity[0], self.drone_velocity[1]]
                    else:
                        self.obs_direction = 'b0'

                    #available_direction.pop(2)


                if '1' in self.obs_direction:
                    self.bug_velocity_direction = list(self.travel_direction)  # noting the direction of velocity
                    self.bug_started = True
                    self.bug_setpoint = list(self.position_setpoint)
                    self.bug_stop_setpoint = list(self.drone_position)   # the point where drone needs to stop
                    self.position_setpoint = self.behind_the_cur_point(20,-5) #to update the setpoint 4 meters before 3 meters below #list(self.drone_position)
                    self.alt_bug_done = False
                    self.bug_count += 1
                    #print("BUGGING")
                    self.setpoint_pub_func()

                    # if self.obs_direction[0] in ['r', 'l']:
                    #     self.obs_cord[1] = self.drone_position[1] # longitude is same as drone
                    #     if self.obs_direction[0] == 'r':
                    #         self.obs_cord[0] = self.drone_position[0] + self.map_sensor_data[idx_dict[self.obs_direction[0]]]/self.meter_conv[0]
                    #     else:
                    #         self.obs_cord[0] = self.drone_position[0] - self.map_sensor_data[idx_dict[self.obs_direction[0]]] / self.meter_conv[0]
                    #
                    # else:
                    #     self.obs_cord[0] = self.drone_position[0]  # lattitude is same as drone
                    #     if self.obs_direction[0] == 'f':
                    #         self.obs_cord[1] = self.drone_position[1] - self.map_sensor_data[idx_dict[self.obs_direction[0]]]/self.meter_conv[1]
                    #     else:
                    #         self.obs_cord[1] = self.drone_position[1] + self.map_sensor_data[idx_dict[self.obs_direction[0]]] / self.meter_conv[1]


                    self.im_breaking_init.data = 35.5
                    self.im_breaking_init_pub.publish(self.im_breaking_init)

                    self.bug_found_dir = self.bug_direction_find(self.obs_direction[0])

                # elif '0' in self.obs_direction:
                #     print("roll pitch controlled")
                #     self.im_breaking_init.data = 20.5
                #     self.im_breaking_init_pub.publish(self.im_breaking_init)

            else:
                #print(" Bug stage : ", self.bug_stage_flag)
                param_list = {'f': 1, 'b': 1, 'r': 0, 'l': 0}
                vel_idx = param_list[self.obs_direction[0]]
                vel_mag = math.sqrt(self.drone_velocity[0]**2 + self.drone_velocity[1]**2)
                print("vmagh : ", vel_mag)
                # if self.snapshot_a[1][vel_idx]*self.drone_velocity[vel_idx] < 0:
                #     self.snapshot_b[0] = [self.drone_position[0], self.drone_position[1]]
                #     self.snapshot_b[1] = [self.drone_velocity[0], self.drone_velocity[1]]
                #
                #     x_diff = (self.snapshot_b[0][0] - self.snapshot_a[0][0])*self.meter_conv[0]
                #     y_diff = (self.snapshot_b[0][1] - self.snapshot_a[0][1]) * self.meter_conv[1]
                #     dist = math.sqrt(x_diff**2 + y_diff**2)
                #     init_vel = math.sqrt(self.snapshot_a[1][0]**2 + self.snapshot_a[1][1]**2)
                #     print("------values-------")
                #     print("velocity : ", self.snapshot_a[1], init_vel)
                #     print("dist : ", dist, "xdiff : ", x_diff, "ydiff : ", y_diff)

                    # self.position_setpoint[0] = float(self.drone_position[0])
                    # self.position_setpoint[1] = float(self.drone_position[1])
                    # self.setpoint_pub_func()

                if self.map_sensor_data[idx_dict[self.obs_direction[0]]] == 25 and self.bug_stage_flag != 4:
                    #print("updating the new setpoint")
                    if self.obs_direction[0] == 'r':
                        self.position_setpoint[1] = self.drone_position[1]  # longitude is same as drone
                        self.position_setpoint[2] = self.drone_position[2] + abs(self.map_sensor_data[1] * math.sin(self.drone_orientation_euler[1])) + 6
                        self.position_setpoint[0] = self.drone_position[0] + self.map_sensor_data[1] * abs(math.cos(self.drone_orientation_euler[1])) / self.meter_conv[0]

                    elif self.obs_direction[0] == 'l':
                        self.position_setpoint[1] = self.drone_position[1]  # longitude is same as drone
                        self.position_setpoint[2] = self.drone_position[2] + abs(self.map_sensor_data[3] * math.sin(self.drone_orientation_euler[1])) + 6
                        self.position_setpoint[0] = self.drone_position[0] - self.map_sensor_data[3] * abs(math.cos(self.drone_orientation_euler[1])) / self.meter_conv[0]

                    elif self.obs_direction[0] == 'f':
                        self.position_setpoint[0] = self.drone_position[0]  # lattitude is same as drone
                        self.position_setpoint[2] = self.drone_position[2] + abs(self.map_sensor_data[0] * math.sin(self.drone_orientation_euler[0])) + 6
                        self.position_setpoint[1] = self.drone_position[1] - self.map_sensor_data[0] * abs(math.cos(self.drone_orientation_euler[0])) / self.meter_conv[1]

                    elif self.obs_direction[0] == 'b':
                        self.position_setpoint[0] = self.drone_position[0]  # lattitude is same as drone
                        self.position_setpoint[2] = self.drone_position[2] + abs(self.map_sensor_data[2] * math.sin(self.drone_orientation_euler[0])) + 6
                        self.position_setpoint[1] = self.drone_position[1] + self.map_sensor_data[2] * abs(math.cos(self.drone_orientation_euler[0])) / self.meter_conv[1]

                    for i in range(2):
                        if self.bug_setpoint[i] - self.drone_position[i] > 0:
                            if self.position_setpoint[i] > self.bug_setpoint[i]:
                                self.position_setpoint[i] = float(self.bug_setpoint[i])
                        elif self.position_setpoint[i] < self.bug_setpoint[i]:
                            self.position_setpoint[i] = self.bug_setpoint[i]

                    self.setpoint_pub_func()
                    self.bug_started = False
                    self.bug_stage_flag = 4
                    #print("updated setpoint : ", self.position_setpoint)
                    self.im_breaking_init.data = 25
                    self.im_breaking_init_pub.publish(self.im_breaking_init)


                if self.bug_stage_flag == 1:

                    self.bug_vel_check_pos_update()    #to update the actual pos when drone stopped
                    self.im_breaking_init.data = 35.5
                    self.im_breaking_init_pub.publish(self.im_breaking_init)

                elif self.bug_stage_flag == 2:          # stablize in setpoint updated in self.bug_vel_check_pos_update()
                    state_check = [False, False, False]
                    for idx in range(3):
                        if abs(self.position_setpoint[idx] - self.drone_position[idx]) * self.meter_conv[idx] < 2.5:
                            state_check[idx] = True

                    if state_check == [True, True, True]:
                        self.im_breaking_init.data = 20
                        self.im_breaking_init_pub.publish(self.im_breaking_init)
                        self.bug_stage_flag = 3

                elif self.bug_stage_flag == 3:
                    self.position_setpoint[2] += 5
                    self.setpoint_pub_func()
                    self.bug_stage_flag = 4


                elif self.alt_bug_done == False and  self.bug_stage_flag == 4:
                    # if (self.bug_setpoint[2] - self.drone_position[2] > -0.5) and (self.bug_setpoint[2] - self.drone_position[2] < 5):
                    #     self.bug_in_action('up',available_direction.index(self.bug_found_dir))
                    # else:
                    # when drone not clear of building height
                    if abs(self.position_setpoint[2] - self.drone_position[2]) < 1 and (self.map_sensor_data[idx_dict[self.obs_direction[0]]] < 25 and self.map_sensor_data[idx_dict[self.obs_direction[0]]] >= 0.3) :
                        self.position_setpoint[2] += 3
                        self.setpoint_pub_func()

                    # when drone clear of building height
                    elif abs(self.position_setpoint[2] - self.drone_position[2]) < 0.5:
                        #print("bug setpoint given")
                        for i in range(2):
                            self.position_setpoint[i] = float(self.bug_setpoint[i])
                        self.position_setpoint[2] += 2
                        self.setpoint_pub_func()
                        self.bug_started = False
                        self.alt_bug_done = True
                        self.bug_stage_flag = 0

                    else:
                        print("nothing ")



                else:
                    self.position_setpoint[0] = float(self.bug_setpoint[0])
                    self.position_setpoint[1] = float(self.bug_setpoint[1])
                    #print("in else")
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

    def bug_vel_check_pos_update(self):

        # self.bug_velocity_direction    # L = -1/R = 1, F = 1/B  = -1    0 if v < 0.2

        #print("travel_direction",self.travel_direction)
        #print("prev_travel_dir",self.bug_velocity_direction)

        if self.travel_direction[0]*self.bug_velocity_direction[0] <= 0 and self.travel_direction[1]*self.bug_velocity_direction[1] <= 0:
            #print("updated")
            self.im_breaking_init.data = 35.5
            self.im_breaking_init_pub.publish(self.im_breaking_init)
            self.position_setpoint = list(self.drone_position)#self.bug_stop_setpoint
            self.setpoint_pub_func()
            self.bug_stage_flag = 2





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

    # Check whether the tallest building is too close
    def tall_building_check(self):
        '''
    	Purpose:
    	---
        When the tallest building or obstacle within the computed path is too close to the starting position of the drone,
        the drone will be unable to complete the ascend to the required altitude before it encounters the obstacle.
        Due to this low ascend time, the drone will hit the obstacle.
        Hence the distance between the starting position of drone and the tallest obstacle is checked, if found close,
        the cell of the tallest obstacle is inserted into the index 1 of the smoothened_path list.

    	Input Arguments:
    	---
    	None

    	Returns:
    	---
    	None

    	Example call:
    	---
    	tall_building_check()
    	'''

        max_height_cell = self.drone_cell_index
        max_height_coords = [0.0, 0.0, 0.0]
        for i in self.found_path:
            if self.map_list[i[0]][i[1]]>self.map_list[max_height_cell[0]][max_height_cell[1]]:
                max_height_cell = i
        for i in self.found_path:
            if self.map_list[i[0]][i[1]]==self.map_list[max_height_cell[0]][max_height_cell[1]]:
                max_height_cell=i
                break

        max_height_coords = self.get_center_coordinate(max_height_cell)
        if(self.package_window_check(max_height_coords,[10, 10, float('inf')])):
            self.smoothened_path.insert(1, max_height_cell)
            #print("Inserted inserted")

        #print("MHCell : ", max_height_cell)
        #print("MHCoord : ", max_height_coords)
        #print("Drone cell index : ", self.drone_cell_index)


    # Main function of this node
    def path_planner(self):
        '''
    	Purpose:
    	---
    	Main algorithm of script. Based on task stage change and requirements, the drone navigation is controlled.
        If path planner is not required, the coordinates subscribed from control node is directly published to position controller script.
        If path planner is enabled, the required path is computed once when task stage changes. Then the path found is used as navigation list setpoints
        and drone moves through the required coordinates, avoiding obstacles if any.
        If destination is not approached even after ending the path computed, the drone moves directly to final destination setpoint.

    	Input Arguments:
    	---
    	None

    	Returns:
    	---
    	None

    	Example call:
    	---
    	Called continously in __main__
    	'''

        self.obs_check()
        # if self.current_package_index not in [2]:
        #     self.bug_boi()

        #print("mapping flag : ", self.mapping_enable_flag)
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
            self.mapping_func()

            if (self.a_star_comp_completed_flag == False):
                try:
                    if self.destination_coords[0] != 0.0:
                        self.a_star(self.find_cell_index(self.drone_position[0:2]), self.find_cell_index(self.destination_coords[0:2]))
                        self.tall_building_check()
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
                self.bug_enable_flag = True
                self.a_star_coords[0:2] = self.get_center_coordinate(self.smoothened_path[self.a_star_path_index])

                self.a_star_coords[2] = self.compute_height() + 2

                if self.bug_started == False:   ################################ added
                    self.position_setpoint = list(self.a_star_coords)
                self.setpoint_pub_func()


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

                # self.bug_boi()

##########################################################################################################################
# Main function

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

    drone_boi = mapping()
    r = rospy.Rate(50)
    drone_boi.initial_setup_func()
    while not rospy.is_shutdown():
        drone_boi.path_planner()
        cv2.imshow("img",drone_boi.map_img)
        # print("BEF : ", drone_boi.bug_enable_flag)
        # print("BSF : ", drone_boi.bug_started)
        # print("BUG COUNT : ", drone_boi.bug_count)
        cv2.waitKey(1)
        r.sleep()
#############################################################################################################################
