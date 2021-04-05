#! /usr/bin/env python

'''
TEAM ID: 2537
TEAM Name: JAGG
'''

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

'''


############################################################################################################################################################

Algorithm: Task 2 - position controller script add-on for scanning, trajectory planning and obstacle avoidance rerouting
ALGOS explanation: Our algorithm is based on scanning the region and determining the set-points. 
Initially, the drone is programmed to move to the package using the position controller script. 
Once it picks the package, it determines the final destination using the QR code detection script and it starts moving towards the destination. 
When the drone encounters an obstacle that is within 10 meters of any one of it's sensors, it stops, 
moves back and scans the surrounding area that is within a circle of radius 25 meters (because the maximum distance that the sensor can detect is 25 meters).
The scanning is done by changing the yaw angle from 0-90 degrees. 
This ensures a 360 degree scanning of the region. After scanning, the drone notes down the possible exit points by using the (r, theta) values 
obtained in the scanning process with a bit of algebra (with a slight offset to avoid collision). 
The drone determines the possible exit points using discontinuities in the distances of the obstacle by setting threshold values. 
The possible points are used by a "cost function" that determines which point to exit from by calculating the distance from those points to the destination point. 
Once the exit point is determined, the drone moves to that point. The important point to note that it does not scan the region again while moving to that point. 
This is done by setting some flag variables. Once it moves to that point, it again resets the set-points to the destination points and the whole process of repeated again 
( if obstacles are detected, it will again scan and determine the end point). 
Once the drone reaches the desired latitude and longitude, it starts descending till the desired altitude is reached and the gripper service is deactivated.

##################################################################################################################################################################

Algorithm : position controller [Task 1]

This is the position controller script. Based on current GPS value and the directed setpoint, error is calculated
and rcValues are calculated using PID and published to attitude controller script. This script dynamically
changes the setpoints to cover all the 4 setpoints mentioned in the problem statement.

Note: For larger heights, we have used velocity based PID control for higher accuracy. Since this task requires
only a smaller height, velocity based control will not be used but it is included in the code.

While moving to a given setpoint, the drone ensures that its current state is stable and proceeds to the setpoint.
The stability of the drone at a particular setpoint within the window is the condition for assigning the next
setpoint and proceeding
'''

'''
THE PROGRAM CAN ACHIEVE AN ACCURACY OF +/- 6*e-6 m in latitude, longitude and altitude
However we have compromised on accuracy to achieve a lower window time.
By changing the error_threshold value (find it in the set of initialized variables) we can set the required accuracy (and indirectly time window time)
We have set error_threshold value as 0.10 to get a time of flight as 19.4 sec [ after experimenting on compromise bw accuracy and time of flight]

'''

class drone_control():

    def __init__(self):

        rospy.init_node('position_control', anonymous=True)

        self.img = np.ones((1000,1000, 3), np.uint8)

        # This corresponds to your current orientation of eDrone converted in euler angles form.
        # [r,p,y]
        self.drone_orientation_euler = [0.0, 0.0, 0.0]

        self.Kp = [1638*0.04, 1638*0.04, 1.0, 1.0, 136*0.6/1.024]  # idx 0- roll, 1-pitch, 2-yaw, 3-throttle, 4-eq_throttle
        self.Ki = [.3, .3, 0.0, 0.0, 192*0.08/1.024]  # Equilibrium control as of now
        self.Kd = [749*0.3, 749*0.3, 0.0, 0.0, 385*0.3/1.024]
        self.vel_kp = [0, 0, 0, 0]  #PID parameters for velocity control
        self.vel_kd = [0, 0, 0, 0] #PID parameters for velocity control
        self.Kp1 = [0, 0, 0, 0]  # Damping in final 2 meters of velocity control
        self.Kd1 = [0, 0, 0, 0] # Damping in final 2 meters of velocity control
        self.Kp2 = [1638*0.04, 1638*0.04, 0, 1550*0.06/1.024]  # Short range control (Used in this task)
        self.Kd2 = [749*0.3, 749*0.3, 0, 502*0.3/1.024]# Short range control (Used in this task)

        self.initial_error = [0.0, 0.0, 0.0]  # idx 0-x, 1-y, 2-z
        self.initial_error_distance = 0

        #Used in velocity Control
        self.vel_setpoint = [0, 0, 0]
        self.vel_magnitude = 0
        self.vel_error = [0.0, 0.0, 0.0]
        self.vel_derivative = [0.0, 0.0, 0.0]
        self.prev_vel_error = [0, 0, 0]




        self.error = [0.0, 0.0, 0.0] #Position error in metres
        self.prev_error = [0.0, 0.0, 0.0] #Position error for derivative
        self.error_derivative = [0.0, 0.0, 0.0]
        self.sample_time = 0.2
        self.accumulator = [0, 0, 0] #Position error for integrating
        self.error_distance = 0
        self.error_threshold = [0.05, 0.05, 0.05]  # used in def stability
        self.velocity_threshold = 0.01    # used in def stability

        self.meter_conv = [110693.227, 105078.267, 1] #Factor for degrees to meters conversion(lat,long,alt to x,y,z)

        self.cosines = [0, 0, 0]#Velocity control using direction cosines

        self.proceed = True #Flag variable for checking state
        self.state = [False, False, False] #Variable for axes state checking
        self.eq_flags = [False, False, False]
        self.target = [19, 72, 0.45] #Dynamic setpoint variable (not used for this task)
        self.rcval = [1500, 1500, 1500, 1500] #Roll,pitch,yaw,throttle
        self.eq_rcval = [1500, 1500, 1500, 1496.98058] #Equilibrium base value for different loads
        self.thresh = 150 #Threshold value for Kd to avoid spikes
        self.command = edrone_cmd()

        self.drone_position = [19, 72, 0.31] #Current drone position in GPS coordinates

        #task1 setpoints
        self.setpoint_1 = [19, 72, 3]
        self.setpoint_2 = [19.0000451704, 72, 3]
        self.setpoint_3 = [19.0000451704, 72, 0.31]
        self.setpoints_task1 = [self.setpoint_1,self.setpoint_2,self.setpoint_3]
        self.counter = 0 #keeping track of setpoints


        self.box_setpoint = [19.0007046575, 71.9998955286, 22.1599967919]
        self.drone_start_pos = [19.0009248718, 71.9998318945, 22.1599967919]
        self.drone_current_pos = [19.0009248718, 71.9998318945, 22.1599967919]
        self.position_setpoint_dest = [19.0009248718, 71.9998318945, 25]
        if self.drone_current_pos[2] > self.box_setpoint[2]:
            self.setpoint_inter_curr = list([self.drone_current_pos[0], self.drone_current_pos[1], self.drone_current_pos[2] + 5])
            self.setpoint_inter_dest = list([self.box_setpoint[0], self.box_setpoint[1],self.drone_current_pos[2] + 5])
        else:
            self.setpoint_inter_curr = list([self.drone_current_pos[0], self.drone_current_pos[1], self.box_setpoint[2] + 5])
            self.setpoint_inter_dest = list([self.box_setpoint[0], self.box_setpoint[1], self.box_setpoint[2] + 5])

        self.inter_points_decision = True
        self.position_setpoint = list(self.setpoint_inter_curr)
        self.vel = [0, 0, 0]

        self.assign()

        self.around_me_reply = 0
        self.start_around_me = False
        self.update_flag = False

        # me new
        self.range_top_dist = [float('inf'), float('inf'), float('inf'), float('inf'),float('inf')]  # [front,right,back,left,top]
        self.obs_flag = False
        self.L = 0
        self.LL = 0
        self.LLL = 0
        self.oi = 0
        self.roll_prev = 0
        self.pitch_prev = 0
        self.range_find_top_prev = [0,0,0,0,0]
        self.prev_sensor_idx = [0, 0, 0, 0]
        self.p = -1
        self.q = -1

        # from skeleton
        self.qr_position = [0, 0, 0]
        self.task_stage_flag = 0
        self.qr_start = Float32()
        self.qr_start.data = 0
        self.gripper_check_flag = "False"

        # SUBSCRIBERS

        rospy.Subscriber('/edrone/gps', NavSatFix, self.edrone_position)
        #rospy.Subscriber('/set_setpoint', SetPoints, self.set_setpoints)
        rospy.Subscriber('/edrone/gps_velocity', Vector3Stamped, self.get_drone_vel)
        rospy.Subscriber('/edrone/imu/data', Imu, self.imu_callback)
        rospy.Subscriber('/edrone/range_finder_top', LaserScan, self.range_top)
        rospy.Subscriber('/around_me_reply', Float32, self.around_me_reply_func)
        rospy.Subscriber('/path_plan_coords', String,self.path_plan_coords_set)
        rospy.Subscriber('/qr_scan_data', String, self.set_qr_setpoint)
        rospy.Subscriber('/obs_status', Float32, self.obs_status)
        #rospy.Subscriber('/pid_tuning_roll', PidTune, self.roll_set_pid)
        #rospy.Subscriber('/pid_tuning_pitch', PidTune, self.pitch_set_pid)
        #rospy.Subscriber('/pid_tuning_yaw', PidTune, self.yaw_set_pid)

        # PUBLISHERS

        self.qr_initiate = rospy.Publisher('/qr_initiate', Float32, queue_size=1)
        self.drone_command = rospy.Publisher('/drone_command', edrone_cmd, queue_size = 1)
        self.around_me_pub = rospy.Publisher('/around_me', Float32, queue_size=1)
        self.cost_dest_update_pub = rospy.Publisher('/cost_dest_update', String, queue_size = 1)
        self.around_me_val = Float32()
        self.around_me_val.data = -1
        self.cost_dest = String()


    def imu_callback(self, msg):
        drone_orientation_quaternion = [0,0,0,0]
        drone_orientation_quaternion[0] = msg.orientation.x
        drone_orientation_quaternion[1] = msg.orientation.y
        drone_orientation_quaternion[2] = msg.orientation.z
        drone_orientation_quaternion[3] = msg.orientation.w

        (self.drone_orientation_euler[0], self.drone_orientation_euler[1], self.drone_orientation_euler[2]) = tf.transformations.euler_from_quaternion([drone_orientation_quaternion[0], drone_orientation_quaternion[1], drone_orientation_quaternion[2], drone_orientation_quaternion[3]])

        self.drone_orientation_euler[0] *= 180 / math.pi
        self.drone_orientation_euler[1] *= 180 / math.pi
        self.drone_orientation_euler[2] *= 180 / math.pi


    def path_plan_coords_set(self,data):

        found = []
        for i in data.data.split():
            found.append(float(i))
        #print("the setpoints updated :", found, "@" * 10)
        self.position_setpoint = list(found)
        self.update_flag = True



    def around_me_reply_func(self,data):
        self.around_me_reply = data.data

    def range_top(self, range_top):

        self.range_top_dist = range_top.ranges
        #print(self.range_top_dist)


    def scan_init(self):
        self.around_me_val.data = 1

    def obs_status(self, status):
        if self.LL == 0:
            if status.data == 1:
                self.obs_flag = True
                self.LL = 1
            else:
                self.obs_flag = False
                self.LL = 0


    #Function for controlling position
    def position_control(self):

        for i in range(3):
            self.stability(i)

        if self.obs_flag and self.L == 0:

            if self.LLL == 0:
                self.position_setpoint = list(self.drone_position)
                self.LLL = 1

            if self.state == [True, True, True] :
                self.scan_init()
                print("started scanning __________________")
                self.L = 1
                self.LLL = 0

        elif self.obs_flag and self.L == 1:
            pass
            #print("im scanning __________________")

        if self.state == [True, True, True] and self.update_flag:
            self.LL = 0
            self.update_flag = False
            self.position_setpoint = list(self.setpoint_inter_dest)
            self.cost_dest.data = str(self.position_setpoint[0]) + " " + str(self.position_setpoint[1]) + " " + str(self.position_setpoint[2])
            print("MID POINT detected")

        self.travel()

        if self.around_me_reply == -1:

            self.around_me_val.data = -1
            self.around_me_reply = 0
            print("scanning done---------------------------------------------------")
            #
            self.obs_flag = False
            self.L = 0

        self.around_me_pub.publish(self.around_me_val)
        #print(self.around_me_val)

        for i in range(3):
            self.stability(i)
        #print(self.state)
        self.cost_dest_update_pub.publish(self.cost_dest)

    ################################################################################################################################
    # Decision to pickup or drop the package will be sent here
    def gripper_function(self,data):
        self.gripper_check_flag = data.data
        if self.task_stage_flag == 2:
            if(self.gripper_check_flag == "True"):
                self.gripper_service_client(True)
        elif self.task_stage_flag == 4:
            self.gripper_service_client(False)

    # Gets the setpoint from the qr code detecting node and gives it to the required variable here
    def set_qr_setpoint(self,set_point):
        qr_list = []
        for i in set_point.data.split(','):
            qr_list.append(float(i))

        self.qr_position = list(qr_list)
        self.position_setpoint_dest = list(qr_list)

    # Broadcasts the activation and deactivation commands to the service
    def gripper_service_client(self,activate_gripper):
        rospy.wait_for_service('/edrone/activate_gripper') #waiting for the service to be advertised
        try:
            self.gripper = rospy.ServiceProxy('edrone/activate_gripper', Gripper) #setting up a local proxy for using the service. Arguments are the name of the service and the service type
            self.req = self.gripper(activate_gripper)

            #print("service active")
            #print(self.req.result)

            if(self.req.result == True):
                    self.task_stage_flag = 3

        except rospy.ServiceException as e:
            pass
            #print("Service call failed: %s"%e)

    # Controls the over-all flow of the problem statement
    def control_flow(self):
        if(self.task_stage_flag == 0):   # go to package altitude
            if not self.state == [True, True, True] and self.position_setpoint == self.setpoint_inter_curr:
                self.position_control()
            else:
                self.position_setpoint = self.setpoint_inter_dest
                self.position_control()

                if(self.drone_position[0]>19.000703 and self.drone_position[0]<19.000705):  ########## need to change
                    if(self.drone_position[1]>71.999895 and self.drone_position[1]<71.999896):
                        self.task_stage_flag = 1 ########## make as 1

        if(self.task_stage_flag == 1):   # scan qr code and descend
            self.position_setpoint = list(self.box_setpoint)
            self.position_control()
            self.qr_start.data = 2.0
            self.qr_initiate.publish(self.qr_start)
            #print(self.qr_position)
            if(self.qr_position[0]>0.0 and self.qr_position[1]>0 and self.qr_position[2]>0):
                self.task_stage_flag = 2 ####################################################### put to 2

        if(self.task_stage_flag == 2):   # pick up package
            rospy.Subscriber('/edrone/gripper_check',String,self.gripper_function)
            self.position_control()


        if(self.task_stage_flag == 3):   # go to destination
            if self.inter_points_decision:
                if self.drone_position[2] > self.position_setpoint_dest[2]:
                    self.setpoint_inter_curr = list([self.drone_position[0], self.drone_position[1], self.drone_position[2] + 5])
                    self.setpoint_inter_dest = list([self.position_setpoint_dest[0], self.position_setpoint_dest[1], self.drone_position[2] + 5])
                else:
                    self.setpoint_inter_curr = list([self.drone_position[0], self.drone_position[1], self.position_setpoint_dest[2] + 5])
                    self.setpoint_inter_dest = list([self.position_setpoint_dest[0], self.position_setpoint_des[1], self.setpoint_inter_dest[2] + 5])
                self.inter_points_decision = False ################ make it true when destination is reached!!!!
                self.position_setpoint = self.setpoint_inter_curr

            if self.position_setpoint == self.setpoint_inter_curr:
                self.position_control()
                if self.state == [True, True, True]:
                    self.position_setpoint = self.setpoint_inter_dest
            elif self.position_setpoint == self.setpoint_inter_dest:
                self.position_control()
                if self.state == [True, True, True]:
                    self.position_setpoint = self.position_setpoint_dest
            elif self.position_setpoint == self.position_setpoint_dest:
                self.position_control()
                if self.state == [True, True, True] or (self.state[0:2] == [True, True] and abs(self.drone_position[2]- self.position_setpoint[2]) < 0.2):
                    print("Target acheived .....")
                    self.task_stage_flag = 4




        if(self.task_stage_flag == 4):   # drop package
            self.position_control()

        #print(self.task_stage_flag)
        self.cost_dest.data = str(self.position_setpoint[0]) + " " + str(self.position_setpoint[1]) + " " + str(self.position_setpoint[2])
        self.cost_dest_update_pub.publish(self.cost_dest)

    ################################################################################################################################################3

    #callback function for drone position
    def edrone_position(self, gps):

        self.drone_position[0] = gps.latitude
        self.drone_position[1] = gps.longitude
        self.drone_position[2] = gps.altitude


    #Callback function for publishing setpoints (Used during coding the script to check for different positions) (not used here)
    def set_setpoints(self, setpoint):

        setpoints = [setpoint.latitude, setpoint.longitude, setpoint.altitude]

        if self.position_setpoint != setpoints:
            self.proceed = False
            self.initial_error_distance = 0

            for i in range(3):

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


    #Function for dynamically assigning one setpoint after the other using flags and drone position
    def set_setpoints_auto(self):

        if self.counter < 3:
            self.position_setpoint = self.setpoints_task1[self.counter]

        if self.state == [True,True,True] and self.counter < 3:
            self.counter += 1
            #self.proceed = False
            self.initial_error_distance = 0

            if self.counter < 3:
                for i in range(3):

                    self.position_setpoint[i] = self.setpoints_task1[self.counter][i]
                    self.state[i] = False

                    if i == 2:
                        self.target[i] = self.drone_position[i] + 0.05
                    else:
                        self.target[i] = self.drone_position[i]

                    self.initial_error[i] = (self.setpoints_task1[self.counter][i] - self.drone_position[i])*self.meter_conv[i]
                    self.initial_error_distance += math.pow(self.initial_error[i], 2)#velocity control

            #delay = rospy.Rate(0.333)
            #delay.sleep()

            self.initial_error_distance = math.pow(self.initial_error_distance, 0.5) #velocity control


        elif self.counter == 3:
            rospy.loginfo("task done")


        else:
            pass

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

    #Function for PID calculation of rcValues
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

        #This block is the threshold absolute distance for velocity control(Not executed for this task,will be used for future tasks)
        if abs(self.initial_error_distance) > 500:  # a value of 100 is given for for now, so as to make sure that the else statement is always initiated for this task
                                                    # we haven't tuned the PID values for velocity control yet, haven't had time.
            if abs(self.error_distance) > 2:  # If the absolute value of the error is greater that 2m, velocity is controlled ( its x, y z components controller individually)
                for i in range(3):
                    self.vel_setpoint[i] = self.vel_magnitude*self.cosines[i]
                    self.vel_error[i] = self.vel[i] - self.vel_setpoint[i]
                    self.vel_derivative[i] = (self.vel_error[i] - self.prev_vel_error[i])/self.sample_time
                    self.prev_vel_error[i] = self.vel_error[i]

                for i in range(2):  # at the last 2 meters of velocity control, position control is used to accurately stop the drone at the given position (again, PID values not tuned yet)
                    self.rcval[i] = self.confine(self.eq_rcval[i] + self.vel_kp[i] * self.error[i] + self.vel_kd[i] * self.vel_derivative[i])

                self.rcval[3] = self.confine(self.eq_rcval[3] + self.vel_kp[3] * self.error[2] + self.vel_kd[3] * self.vel_derivative[2])

                self.assign()
                self.drone_command.publish(self.command)



            else:

                for i in range(2):
                    self.rcval[i] = self.confine(self.eq_rcval[i] + self.Kp1[i] * self.error[i] + self.Kd1[i] * self.error_derivative[i])

                self.rcval[3] = self.confine(self.eq_rcval[3] + self.Kp1[3] * self.error[2] + self.Kd1[3] * self.error_derivative[2])

                self.assign()
                self.drone_command.publish(self.command)

            print("In velocity control")

        #This block is always executed in this task as absolute distance is less.
        else:

            for i in range(2):
                self.rcval[i] = self.confine(self.eq_rcval[i] + self.Kp2[i] * self.error[i] + self.Kd2[i] * self.error_derivative[i])

            self.rcval[3] = self.confine(self.eq_rcval[3] + self.Kp2[3] * self.error[2] + self.Kd2[3] * self.error_derivative[2])

            self.assign()
            self.drone_command.publish(self.command)

            print("In postion control")


        self.error_distance = 0


    #Function for assigning rcValues into object
    def assign(self):
        self.command.rcRoll = self.rcval[1]
        self.command.rcPitch = self.rcval[0]
        self.command.rcYaw = self.rcval[2]
        self.command.rcThrottle = self.rcval[3]

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


    #Callback function for getting current velocity(not used here)
    def get_drone_vel(self, v):

        self.vel[0] = v.vector.x
        self.vel[1] = v.vector.y
        self.vel[2] = v.vector.z


    #Function to check whether it has reached the setpoint stable.
    # It has been modified to reduce complexity as the equilibrium function is not used
    def stability(self, idx):

        # if self.proceed:
        if abs(self.position_setpoint[idx] - self.drone_position[idx])*self.meter_conv[idx] < self.error_threshold[idx] and abs(self.vel[idx]) < self.velocity_threshold:
            self.state[idx] = True
        else:
            self.state[idx] = False
        #
        # else:
        #     if abs(self.target[idx] - self.drone_position[idx])*self.meter_conv[idx] < self.error_threshold[idx] and abs(self.vel[idx]) < self.velocity_threshold:
        #         self.state[idx] = True
        #     else:
        #         self.state[idx] = False

    #Tuning parameters during development and debugging
    def roll_set_pid(self, roll):
        self.Ki[0] = roll.Ki * 0.04  # This is just for an example. You can change the ratio/fraction value accordingly
        self.Ki[1] = roll.Ki * 0.04
        self.Ki[2] = roll.Kp * 0.04

    def pitch_set_pid(self, pitch):
        self.vel_kp[0] = pitch.Kp * 0.04  # This is just for an example. You can change the ratio/fraction value accordingly
        self.vel_ki[0] = pitch.Ki * 0.00
        self.vel_kd[0] = pitch.Kd * 0.3

    def yaw_set_pid(self, yaw):
        self.Kp1[0] = yaw.Kp * 0.04  # This is just for an example. You can change the ratio/fraction value accordingly
        self.Ki1[0] = yaw.Ki * 0.00
        self.Kd1[0] = yaw.Kd * 0.3


#main function
if __name__ == "__main__":

    drone_boi = drone_control()
    r = rospy.Rate(5)
    while not rospy.is_shutdown():
        if drone_boi.LL == 0:
            drone_boi.control_flow()
        else:
            drone_boi.position_control()

        r.sleep()

# END OF POSITION CONTROLLER PROGRAM
