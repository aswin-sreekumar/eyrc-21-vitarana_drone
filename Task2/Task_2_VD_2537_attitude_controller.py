#!/usr/bin/env python

'''
TEAM ID: 2537
TEAM NAME: JAGG
'''

'''
This python file runs a ROS-node of name attitude_control which controls the roll pitch and yaw angles of the eDrone.
This node publishes and subsribes the following topics:
        PUBLICATIONS            SUBSCRIPTIONS
        /roll_error             /pid_tuning_altitude
        /pitch_error            /pid_tuning_pitch
        /yaw_error              /pid_tuning_roll
        /edrone/pwm             /edrone/imu/data
                                /edrone/drone_command

Rather than using different variables, use list. eg : self.setpoint = [1,2,3], where index corresponds to x,y,z ...rather than defining self.x_setpoint = 1, self.y_setpoint = 2
CODE MODULARITY AND TECHNIQUES MENTIONED LIKE THIS WILL HELP YOU GAINING MORE MARKS WHILE CODE EVALUATION.
'''

# Importing the required libraries

from vitarana_drone.msg import *
from pid_tune.msg import PidTune
from sensor_msgs.msg import *
from std_msgs.msg import *
import rospy
import time
import tf
import math
import numpy as np
import cv2
import copy


'''
Algorithm implemented in this node:

Checks for obstacle in trajectory of drone, when obstacle is detected,
Yaw rotated by 90 deg to scan 360 deg data and compute possible setpoints, filter them and implement the calculated set point
Obstacle avoidance, sudden obstacle jerking, filtering data based on quadrant with respect to shortest path has been considered

Refer the position controller node script for detailed combined algorithm

This node performs obstacle detection, scanning of surroundings and dynamic path planning to compute middle set point

This node basically also performs attitude control based on roll, pitch and yaw published by position controller

'''
def return2(i):
    return i[1]

class Edrone():
    """docstring for Edrone"""
    def __init__(self):
        rospy.init_node('attitude_controller')  # initializing ros node with name drone_control

        # This corresponds to your current orientation of eDrone in quaternion format. This value must be updated each time in your imu callback
        # [x,y,z,w]
        self.drone_orientation_quaternion = [0.0, 0.0, 0.0, 0.0]

        # This corresponds to your current orientation of eDrone converted in euler angles form.
        # [r,p,y]
        self.drone_orientation_euler = [0.0, 0.0, 0.0]

        # This is the setpoint that will be received from the drone_command in the range from 1000 to 2000
        # [r_setpoint, p_setpoint, y_setpoint]
        self.setpoint_cmd = [1500, 1500, 1500]
        self.setpoint_cmd_throttle = 1000


        # The setpoint of orientation in euler angles at which you want to stabilize the drone
        # [r_setpoint, p_psetpoint, y_setpoint]
        self.setpoint_euler = [0.0, 0.0, 0.0]
        # throttle value
        self.setpoint_throttle = 0.0

        # Declaring pwm_cmd of message type prop_speed and initializing values
        # Hint: To see the message structure of prop_speed type the following command in the terminal
        # rosmsg show vitarana_drone/prop_speed

        self.pwm_cmd = prop_speed()
        self.pwm_cmd.prop1 = 0.0
        self.pwm_cmd.prop2 = 0.0
        self.pwm_cmd.prop3 = 0.0
        self.pwm_cmd.prop4 = 0.0

        self.roll_error = Float32()
        self.roll_error.data = 0.0
        self.pitch_error = Float32()
        self.pitch_error.data = 0.0
        self.yaw_error = Float32()
        self.yaw_error.data = 0.0
        self.zero_error_0 = Float32()
        self.zero_error_0.data = 0.0




        # initial setting of Kp, Kd and ki for [roll, pitch, yaw]. eg: self.Kp[2] corresponds to Kp value in yaw axis
        # after tuning and computing corresponding PID parameters, change the parameters
        self.Kp = [0.006*500, 0.006*500, 0.019*1212]
        self.Ki = [0, 0, 0]
        self.Kd = [ 0.03*300*2, 0.03*300*2, 0.08*980*2]
        # -----------------------Add other required variables for pid here ----------------------------------------------
        #
        self.prev_error = [0,0,0]  #[roll, pitch, yaw]
        self.error = [0,0,0] #[roll, pitch, yaw]
        self.Iterm = [0,0,0] #[roll, pitch, yaw]
        self.output = [0,0,0] #[roll, pitch, yaw]

        self.max_values = [1024, 1024, 1024, 1024] #[prop1, prop2, prop3, prop4]
        self.min_values = [0, 0, 0, 0]

        # STUFF FOR around_me and around_me_exe function
        self.range_top_dist = [float('inf'), float('inf'), float('inf'), float('inf'), float('inf')]  # [front,right,back,left,top]
        self.obs_coords_range_top = [float('inf'), float('inf'), float('inf'), float('inf')]  # [front,right,back,left]
        self.img = np.ones((1020, 1020, 3), np.uint8)
        self.around_me_flag = False
        self.obstacle_datapoints = []
        self.a = 0
        self.b = 350
        self.k = 500*90/120
        self.stop_flag = True
        self.eq_yaw = 1500
        self.prev_data = -1
        self.prev_angle = 180

        self.d = 4
        self.d2 = 1.5
        self.prev_r = [25, 0]

        #####new
        self.drone_position = [19, 72, 0.31]  # Current drone position in GPS coordinates
        self.meter_conv = [110693.227, 105078.267, 1]  # Factor for degrees to meters conversion(lat,long,alt to x,y,z)

        self.obs_flag = False
        self.roll_prev = 0
        self.pitch_prev = 0
        self.range_find_top_prev = [0, 0, 0, 0, 0]
        self.prev_sensor_idx = [0, 0, 0, 0]
        self.p = -1
        self.q = -1
        self.obs_status = Float32()

        # Hint : Add variables for storing previous errors in each axis, like self.prev_values = [0,0,0] where corresponds to [roll, pitch, yaw]
        #        Add variables for limiting the values like self.max_values = [1024, 1024, 1024, 1024] corresponding to [prop1, prop2, prop3, prop4]
        #                                                   self.min_values = [0, 0, 0, 0] corresponding to [prop1, prop2, prop3, prop4]
        #
        # ----------------------------------------------------------------------------------------------------------

        # # This is the sample time in which you need to run pid. Choose any time which you seem fit. Remember the stimulation step time is 50 ms
        self.sample_time = 10  # in seconds

        # Publishing /edrone/pwm, /roll_error, /pitch_error, /yaw_error
        self.pwm_pub = rospy.Publisher('/edrone/pwm', prop_speed, queue_size=1)
        self.roll_error_pub = rospy.Publisher('/roll_error',Float32, queue_size=1)
        self.pitch_error_pub = rospy.Publisher('/pitch_error',Float32, queue_size=1)
        self.yaw_error_pub = rospy.Publisher('/yaw_error',Float32, queue_size=1)
        self.zero_error_pub = rospy.Publisher('/zero_error',Float32, queue_size=1)
        self.around_me_reply_pub = rospy.Publisher('/around_me_reply',Float32,queue_size = 1)
        #for publishing found coords after scan
        self.path_plan_coords_pub = rospy.Publisher('/path_plan_coords', String, queue_size=1)
        #to publish obs_check values
        self.obs_status_pub = rospy.Publisher('/obs_status', Float32, queue_size = 1)

        self.path_plan_coords = String()
        self.path_plan_coords.data = ""

        self.around_me_reply = Float32()
        self.around_me_reply.data = -1

        self.dest = [0, 0, 0]
        self.o = 0



        # ------------------------Add other ROS Publishers here-----------------------------------------------------

        # -----------------------------------------------------------------------------------------------------------

        # Subscribing to /drone_command, imu/data, /pid_tuning_roll, /pid_tuning_pitch, /pid_tuning_yaw
        rospy.Subscriber('/drone_command', edrone_cmd, self.drone_command_callback)
        rospy.Subscriber('/edrone/imu/data', Imu, self.imu_callback)
        rospy.Subscriber('/edrone/gps', NavSatFix, self.edrone_position)
        rospy.Subscriber('/edrone/range_finder_top', LaserScan, self.range_top)
        rospy.Subscriber('/around_me', Float32, self.around_me_exe)
        rospy.Subscriber('/cost_dest_update', String, self.cost_dest_update)

        #rospy.Subscriber('/pid_tuning_roll', PidTune, self.roll_set_pid)
        #rospy.Subscriber('/pid_tuning_pitch', PidTune, self.pitch_set_pid)
        #rospy.Subscriber('/pid_tuning_yaw', PidTune, self.yaw_set_pid)
        # -------------------------Add other ROS Subscribers here----------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------

    # Imu callback function
    # The function gets executed each time when imu publishes /edrone/imu/data

    # Note: The imu publishes various kind of data viz angular velocity, linear acceleration, magnetometer reading (if present),
    # but here we are interested in the orientation which can be calculated by a complex algorithm called filtering which is not in the scope of this task,
    # so for your ease, we have the orientation published directly BUT in quaternion format and not in euler angles.
    # We need to convert the quaternion format to euler angles format to understand the orienataion of the edrone in an easy manner.
    # Hint: To know the message structure of sensor_msgs/Imu, execute the following command in the terminal
    # rosmsg show sensor_msgs/Imu

    # checking for obstacles in direction of trajectory
    def obs_check(self):    
        sensor_idx = [0, 0, 0, 0]
        sensor_validity = [0, 0, 0, 0]

        roll_curr = copy.copy(self.drone_orientation_euler[0])
        pitch_curr = copy.copy(self.drone_orientation_euler[1])
        range_find_top_curr = list(self.range_top_dist[0:4])
        self.obs_flag = False
        if self.o < 90:
            sensor_validity = [1, 0, 0, 1]
        elif self.o < 180:
            sensor_validity = [0, 0, 1, 1]
        elif self.o < 270:
            sensor_validity = [0, 1, 1, 0]
        else:
            sensor_validity = [1, 1, 0, 0]

        for i in range(4):

            if self.range_top_dist[i] <= 13 and self.range_top_dist[i] > 0.3:
                sensor_idx[i] = 1
            else:
                sensor_idx[i] = 0

        for x in range(4): # checking all sensors
            if sensor_idx[x] and self.prev_sensor_idx[x] and sensor_validity[x]:
                if x%2 == 0:
                    if (roll_curr - self.roll_prev)*(range_find_top_curr[x] - self.range_find_top_prev[x])*roll_curr > 0 or abs(math.sin(roll_curr*180/math.pi)*range_find_top_curr[x]) < 0.4:
                        self.obs_flag = True
                        self.LL = 1
                        self.p = abs(math.sin(roll_curr * 180 / math.pi) * range_find_top_curr[x])

                else:

                    if (pitch_curr - self.pitch_prev)*(range_find_top_curr[x] - self.range_find_top_prev[x])*pitch_curr >0 or abs(math.sin(pitch_curr*180/math.pi)*range_find_top_curr[x]) < 0.4:
                        self.obs_flag = True
                        self.LL = 1
                        self.q = abs(math.sin(pitch_curr*180/math.pi)*range_find_top_curr[x])

        # print("p -- ",self.p)
        # print("q -- ",self.q)
        # print(roll_curr - self.roll_prev, pitch_curr - self.pitch_prev, "=" * 5)

        self.roll_prev = copy.copy(roll_curr)
        self.pitch_prev = copy.copy(pitch_curr)
        self.range_find_top_prev = list(range_find_top_curr)
        self.prev_sensor_idx = list(sensor_idx)

        if self.obs_flag:
            self.obs_status.data = 1
        else:
            self.obs_status.data = 0

        self.obs_status_pub.publish(self.obs_status)





    # computing costs for setpoints computed
    def cost_dest_update(self, data):
        self.dest = data.data.split()
        for i in range(3):
            self.dest[i] = float(self.dest[i])
        y = -(self.dest[0] - self.drone_position[0]) * self.meter_conv[0]
        x = -(self.dest[1] - self.drone_position[1]) * self.meter_conv[1]
        self.o = math.atan(y / x)

        if self.o > 0:
            self.o = self.o*180/math.pi
        elif self.o < 0:
            self.o = 360 + (self.o*180/math.pi)

    # call back for sensor subscriber
    def range_top(self, range_top):

        self.range_top_dist = range_top.ranges
        #print(self.range_top_dist)

    # PLotting and computing sensor data around the drone
    def around_me_exe(self, data):
        # print("in")
        # print("-"*10)
        if data.data == 1 and data.data*self.prev_data == -1:
            self.around_me_flag = True
            self.obstacle_datapoints = []
            self.a = 0
            self.around_me()
            self.stop_flag = False
            self.img = np.ones((1020, 1020, 3), np.uint8)


        elif data.data == 1:
            if not self.stop_flag:
                self.around_me()
                #print(self.stop_flag)
            else:

                self.around_me_flag = False
                self.obstacle_datapoints = sorted(self.obstacle_datapoints, key=return2)
                self.analysis()
                if self.obstacle_datapoints[0][0] == float('inf'):
                    self.prev_r = [25.1, self.obstacle_datapoints[0][1]]
                else:
                    self.prev_r = self.obstacle_datapoints[0]
                self.around_me_reply_pub.publish(self.around_me_reply)
                for i in self.obstacle_datapoints:
                    if not i[0] == float('inf'):
                        cv2.circle(self.img, (int(i[0]*20*math.cos(i[1]*math.pi/180) )+ 510, -int(i[0]*20*math.sin(i[1]*math.pi/180))+ 510), 1, (255, 255, 0), 2)
                    else:
                        cv2.circle(self.img, (int(25 * 20 * math.cos(i[1] * math.pi / 180)) + 510, -int(25 * 20 * math.sin(i[1] * math.pi / 180)) + 510), 1,(255, 255, 255), 2)
                cv2.circle(self.img, (510, 510), 50, (255, 0, 0), 2)
                cv2.line(self.img, (510,510), (560,510), (255, 0, 0), 2)

                self.b = -self.b #####
                #cv2.imwrite("/home/jk56/PycharmProjects/pythonProject/img1.jpg", self.img)
                # please dont consider the circle printing codes, it was used for debgging the scanning



        self.prev_data = data.data

    # scans for the obstacles around the drone
    def around_me(self):

        front, right, back, left, top = self.range_top_dist
        theta = self.drone_orientation_euler[2]
        angle_values = [0, 90, 180, 270]
        if theta > 0:
            for i in range(4):
                angle_values[i] += theta
        else:
            for i in range(1,4):
                angle_values[i] += theta
            angle_values[0] = 360 + theta

        self.obstacle_datapoints.extend([[front, angle_values[0]], [left, angle_values[1]], [back, angle_values[2]], [right, angle_values[3]]])

        if self.a != 0:
            self.stop_flag = False

        if (theta >= self.a*0.24 - 20) and (theta <= self.a*0.24 + 20):
            self.a += self.b
            print("SCANNING IN PROCESS ....")

            if self.a >= self.k:
                self.b = -self.b
        if theta >= 90 and self.a > self.k:
            self.a += self.b

        #elif self.a < -1 * self.k:
         #   self.b = -self.b

        if not self.stop_flag and abs(theta) < 2 and self.b < 0:
            self.stop_flag = True




    # data analysis of raw data 
    def analysis(self):
        gap_list = [([0,0], [0,0]), ([0,0], [0,0])]
        exit_list = []
        final_list = []
        dist = 5

        for i in self.obstacle_datapoints:
            if i[0] == float('inf'):
                i = [25.1, i[1]]
                dist = 1.5
            if abs(i[0] - self.prev_r[0]) > dist and self.prev_r not in [gap_list[-1][0], gap_list[-1][0], gap_list[-2][0], gap_list[-2][1]]:   ############################################# Condition to stop it from oscillating between two points !!
                gap_list.append((self.prev_r, i))
            elif abs(i[0] - self.prev_r[0]) > dist and self.prev_r in [gap_list[-1][0], gap_list[-1][0], gap_list[-2][0], gap_list[-2][1]]:
                gap_list.pop(-1)
            dist = 5
            self.prev_r = i

        gap_list.remove(([0,0], [0,0]))
        gap_list.remove(([0, 0], [0, 0]))

        if self.obstacle_datapoints[-1][0] == 25.1 or self.obstacle_datapoints[0][0] == 25.1:
            dist = 1.5

        if abs(min(25.1, self.obstacle_datapoints[-1][0]) - min(25.1, self.obstacle_datapoints[0][0])) > dist:
            gap_list.append(([min(25.1, self.obstacle_datapoints[-1][0]), self.obstacle_datapoints[-1][1]], [min(25.1, self.obstacle_datapoints[0][0]), self.obstacle_datapoints[0][1]]))

        x = 0
        removal = []
        for i in range(len(self.obstacle_datapoints)):
            if x == len(gap_list):
                break
            else:
                if self.obstacle_datapoints[i][1] > gap_list[x][1][1]:
                    try:
                        if self.obstacle_datapoints[i][0] < gap_list[x][1][0] and self.obstacle_datapoints[i - 1][0] < gap_list[x][1][0]:
                            removal.append(gap_list[x])

                    except IndexError:
                        if self.obstacle_datapoints[i][0] < gap_list[x][1][0] and self.obstacle_datapoints[-1][0] < gap_list[x][1][0]:
                            removal.append(gap_list[x])
                    x += 1
                elif self.obstacle_datapoints[i][1] > gap_list[x][0][1]:
                    try:
                        if self.obstacle_datapoints[i][0] < gap_list[x][0][0] and self.obstacle_datapoints[i - 1][0] <gap_list[x][0][0]:
                            removal.append(gap_list[x])

                    except IndexError:
                        if self.obstacle_datapoints[i][0] < gap_list[x][0][0] and self.obstacle_datapoints[-1][0] < gap_list[x][0][0]:
                            removal.append(gap_list[x])
                    x += 1

        if x == len(gap_list) - 1:

            if self.obstacle_datapoints[-1][0] < gap_list[x][1][0] and self.obstacle_datapoints[0][0] < gap_list[x][1][0]:
                removal.append(gap_list[x])

            if self.obstacle_datapoints[-1][0] < gap_list[x][0][0] and self.obstacle_datapoints[0][0] < gap_list[x][0][0]:
                removal.append(gap_list[x])

        for i in removal:
            try:
                gap_list.remove(i)
            except ValueError:
                pass

        for i, j in gap_list:

            cv2.circle(self.img, (int(i[0] * 20 * math.cos(i[1] * math.pi / 180) + 510), -int(i[0] * 20 * math.sin(i[1] * math.pi / 180)) + 510), 20, (0, 0, 255), 2)

            cv2.circle(self.img, (int(j[0] * 20 * math.cos(j[1] * math.pi / 180) + 510),  -int(j[0] * 20 * math.sin(j[1] * math.pi / 180)) + 510), 20, (0, 0, 255), 2)
            cv2.line(self.img, (int(i[0] * 20 * math.cos(i[1] * math.pi / 180) + 510), -int(i[0] * 20 * math.sin(i[1] * math.pi / 180)) + 510), (int(j[0] * 20 * math.cos(j[1] * math.pi / 180) + 510), -int(j[0] * 20 * math.sin(j[1] * math.pi / 180)) + 510), (0, 0, 255), 2)
        #print("final gap")
        for i in gap_list:
            print("Detected possiblities ...")
            print(i)

        if len(gap_list) == 0:
            print("Free to move")

        elif len(gap_list) == 2:
            for i in gap_list:
                if i[0][0] == 25.1:
                    final_list.append((i[1][0] + 2, i[1][1] - 10)) # interchanged because of inverted y axis
                else:
                    final_list.append((i[0][0] + 2, i[0][1] + 10))

        else:
            x = 0
            for i in range(len(gap_list)-1): # joining discontiunities at infinity
                if x ==0:
                    if gap_list[i][1][0] == 25.1 and gap_list[i+1][0][0] == 25.1 :
                        exit_list.append((gap_list[i][0], gap_list[i+1][1]))
                        x = 1
                    else:
                        exit_list.append(gap_list[i])
                else:
                    x = 0

            if gap_list[-1][1][0] == 25.1 and gap_list[0][0][0] == 25.1:
                exit_list.append((gap_list[-1][0], gap_list[0][1]))
                exit_list.pop(0)

            removal = []
            for i in range(len(exit_list)-1):
                p = exit_list[i]
                q = exit_list[i+1]
                if abs(p[0][1] - p[1][1]) < 2 and abs(q[0][1] - q[1][1]) < 2 and len(exit_list) - len(removal) > 2:
                    if (min(q[0][0], q[1][0]) + min(p[0][0], p[1][0])) * abs(p[1][1] - q[0][1]) * math.pi/360 < self.d2:
                        removal.append(q)

            for i in removal:
                exit_list.remove(i)


            for i, j in exit_list:
                cv2.line(self.img, (int(i[0] * 20 * math.cos(i[1] * math.pi / 180) + 510), -int(i[0] * 20 * math.sin(i[1] * math.pi / 180)) + 510), (int(j[0] * 20 * math.cos(j[1] * math.pi / 180) + 510), -int(j[0] * 20 * math.sin(j[1] * math.pi / 180)) + 510), (0, 255, 0), 2)

            for i in range(len(exit_list)):
                p, q = exit_list[i]
                x = p[0]*math.cos(p[1]) - q[0]*math.cos(q[1])
                y = p[0] * math.sin(p[1]) - q[0] * math.sin(q[1])
                d = math.sqrt(x**2 + y**2)

                o = 180/math.pi
                if d > 2*self.d and abs(p[1] - q[1]) > 2:
                    if p[0] < 25.05:
                        if p[1] + self.d*o/p[0] > 0:
                            final_list.append((p[0] +2 , p[1] + self.d*o/p[0]))
                        else:
                            final_list.append((p[0] + 2, p[1] + self.d*o/p[0] + 360))
                    if q[0] < 25.05:
                        if q[1] - self.d*o/q[0] > 0:
                            final_list.append((q[0] +2 , q[1] - self.d*o/q[0]))
                        else:
                            final_list.append((q[0] + 2, q[1] - self.d*o/q[0] + 360))

                   # print("multi- point", p, q)

                elif d > 2*self.d:
                    if p[0] > q[0] :
                        if q[1] - self.d*o/q[0]>0:
                            final_list.append((q[0] + 2, q[1] - self.d*o/q[0]))
                        else:
                            final_list.append((q[0] + 2, q[1] - self.d*o/q[0] + 360))
                    else:
                        if p[1] - self.d*o/p[0] > 0:
                            final_list.append((p[0] + 2, p[1] + self.d*o/p[0]))
                        else:
                            final_list.append((p[0] + 2, p[1] + self.d*o/p[0] + 360))
                    #print("single - point", p, q)

                elif d > self.d :
                    final_list.append((min(p[0], q[0]) - 1, (p[1] + q[1])/2 - self.d*o/min(p[0], q[0]) ))
                    
                            ################################# removing all final lines behind obstacles
        x = 0
        removal = []
        for i in range(len(self.obstacle_datapoints)):
            if x == len(final_list):
                break
            else:
                if self.obstacle_datapoints[i][1]> final_list[x][1]:
                    try:
                        if self.obstacle_datapoints[i][0] < final_list[x][1] and self.obstacle_datapoints[i-1][0] < final_list[x][0]:
                            removal.append(final_list[x])
                        x += 1
                    except IndexError:
                        if self.obstacle_datapoints[i][0] < final_list[x][1] and self.obstacle_datapoints[-1][0] < final_list[x][0]:
                            removal.append(final_list[x])
                        x += 1

        if x == len(final_list)-1:
            if self.obstacle_datapoints[-1][0] < final_list[x][1] and self.obstacle_datapoints[0][0] < final_list[x][0]:
                removal.append(final_list[x])

        for i in removal:
            final_list.remove(i)



        # for i in exit_list:
        #     print(i)

        print("Final Data", final_list)


        try:
            index = self.cost_dist(final_list)
            cv2.circle(self.img, (int(final_list[index][0] * 20 * math.cos(final_list[index][1] * math.pi / 180) + 510), -int(final_list[index][0] * 20 * math.sin(final_list[index][1] * math.pi / 180)) + 510), 20, (0, 255, 0), 2)

        except ValueError:
            pass
            #print("VALUE ERROR in Data scan")

        for i in final_list:
            cv2.circle(self.img, (int(i[0] * 20 * math.cos(i[1] * math.pi / 180) + 510), -int(i[0] * 20 * math.sin(i[1] * math.pi / 180) )+ 510), 10, (255, 0, 255), 2)
            cv2.line(self.img, (510, 510), (int(i[0] * 20 * math.cos(i[1] * math.pi / 180)) + 510, -int(i[0] * 20 * math.sin(i[1] * math.pi / 180))+ 510), (255, 0, 255), 2)

            # append (r, theta) as per conditions you see fit. Should be a good collection of appropriate values of radius.
            # this condition is satisfied if there are are obstacles at all sides including or except the direction of entry also.






    # cost computing for filtered setpoints
    def cost_dist(self,final):
        cost_list = []

        setpoint_dist = math.sqrt(((self.drone_position[0] - self.dest[0])*self.meter_conv[0])**2 + ((self.drone_position[1] - self.dest[1])*self.meter_conv[1])**2)
        #print("o : ", self.o)
        #print("setpoint_dist : ", setpoint_dist)
        #print("dest : ", self.dest)

        for i in final:
            final_set = [(self.drone_position[0]*self.meter_conv[0]) - (i[0] * math.sin(i[1]* math.pi / 180)),(self.drone_position[1]* self.meter_conv[1]) - (i[0] * math.cos(i[1]* math.pi / 180))]
            # dist between drone_curr_pos and final_list point
            dist_g = math.sqrt(((self.drone_position[0] * self.meter_conv[0]) - final_set[0])**2 + ((self.drone_position[1] * self.meter_conv[1]) - final_set[1])**2)
            # dist between destination and final_list point
            dist_f = math.sqrt((final_set[0] - (self.dest[0]*self.meter_conv[0]))**2 + (final_set[1] - (self.dest[1]*self.meter_conv[1]))**2)

            dist_o = abs(i[1] - self.o)*10/360

            cost = dist_f*100 + dist_g*100 + dist_o
            cost_list.append(cost)

        min_index = cost_list.index(min(cost_list))

        decided_coord = [(self.drone_position[0]) - ((final[min_index][0] / self.meter_conv[0]) * math.sin(final[min_index][1]* math.pi / 180)),(self.drone_position[1]) - ((final[min_index][0] / self.meter_conv[1]) * math.cos(final[min_index][1]* math.pi / 180))]

        print("found :",decided_coord)

        for i in decided_coord:
            self.path_plan_coords.data += str(i) + " "

        self.path_plan_coords.data += str(self.dest[2])  # 25 -- altitude
        #self.path_plan_coords.data = f"{decided_coord[0]} {decided_coord[1]} {25}"   # 25 -- altitude

        self.path_plan_coords_pub.publish(self.path_plan_coords)  #publishing found coordinates to pos
        self.path_plan_coords.data = ""

        o = 0

        return min_index











    def edrone_position(self, gps):

        self.drone_position[0] = gps.latitude
        self.drone_position[1] = gps.longitude
        self.drone_position[2] = gps.altitude





    def imu_callback(self, msg):

        self.drone_orientation_quaternion[0] = msg.orientation.x
        self.drone_orientation_quaternion[1] = msg.orientation.y
        self.drone_orientation_quaternion[2] = msg.orientation.z
        self.drone_orientation_quaternion[3] = msg.orientation.w

        (self.drone_orientation_euler[0], self.drone_orientation_euler[1], self.drone_orientation_euler[2]) = tf.transformations.euler_from_quaternion([self.drone_orientation_quaternion[0], self.drone_orientation_quaternion[1], self.drone_orientation_quaternion[2], self.drone_orientation_quaternion[3]])

        self.drone_orientation_euler[0] *= 180 / math.pi
        self.drone_orientation_euler[1] *= 180 / math.pi
        self.drone_orientation_euler[2] *= 180 / math.pi

        # --------------------Set the remaining co-ordinates of the drone from msg----------------------------------------------

    def drone_command_callback(self, msg):
        self.setpoint_cmd[0] = msg.rcRoll
        self.setpoint_cmd[1] = msg.rcPitch
        self.setpoint_cmd_throttle = msg.rcThrottle

        if not self.around_me_flag:
            self.setpoint_cmd[2] = msg.rcYaw
        else:
            self.eq_yaw = msg.rcYaw
            self.setpoint_cmd[2] = self.eq_yaw + self.a


        # ---------------------------------------------------------------------------------------------------------------

    # Callback function for /pid_tuning_roll
    # This function gets executed each time when /tune_pid publishes /pid_tuning_roll
    def roll_set_pid(self, roll):
        self.Kp[0] = roll.Kp * 0.006    # This is just for an example. You can change the ratio/fraction value accordingly
        self.Ki[0] = roll.Ki * 0.008
        self.Kd[0] = roll.Kd * 0.03

    def pitch_set_pid(self, pitch):
        self.Kp[1] = pitch.Kp * 0.006   # This is just for an example. You can change the ratio/fraction value accordingly
        self.Ki[1] = pitch.Ki * 0.008
        self.Kd[1] = pitch.Kd * 0.03

    def yaw_set_pid(self, yaw):
        self.Kp[2] = yaw.Kp * 0.019  #  0.019 - 1212 or 4192  This is just for an example. You can change the ratio/fraction value accordingly
        self.Ki[2] = yaw.Ki * 0.008
        self.Kd[2] = yaw.Kd * 0.08       #0.08 - 980


    # ----------------------------Define callback function like roll_set_pid to tune pitch, yaw--------------

    # ----------------------------------------------------------------------------------------------------------------------



    def pid(self):
        # -----------------------------Write the PID algorithm here--------------------------------------------------------------

        # Steps:
        #   1. Convert the quaternion format of orientation to euler angles
        #   2. Convert the setpoin that is in the range of 1000 to 2000 into angles with the limit from -10 degree to 10 degree in euler angles
        #   3. Compute error in each axis. eg: error[0] = self.setpoint_euler[0] - self.drone_orientation_euler[0], where error[0] corresponds to error in roll...
        #   4. Compute the error (for proportional), change in error (for derivative) and sum of errors (for integral) in each axis. Refer "Understanding PID.pdf" to understand PID equation.
        #   5. Calculate the pid output required for each axis. For eg: calcuate self.out_roll, self.out_pitch, etc.
        #   6. Use this computed output value in the equations to compute the pwm for each propeller. LOOK OUT FOR SIGN (+ or -). EXPERIMENT AND FIND THE CORRECT SIGN
        #   7. Don't run the pid continously. Run the pid only at the a sample time. self.sampletime defined above is for this purpose. THIS IS VERY IMPORTANT.
        #   8. Limit the output value and the final command value between the maximum(0) and minimum(1024)range before publishing. For eg : if self.pwm_cmd.prop1 > self.max_values[1]:
        #                                                                                                                                      self.pwm_cmd.prop1 = self.max_values[1]
        #   8. Update previous errors.eg: self.prev_error[1] = error[1] where index 1 corresponds to that of pitch (eg)
        #   9. Add error_sum to use for integral component

        # 1.Converting quaternion to euler angles

        # 2.Convertng the range from 1000 to 2000 in the range of -10 degree to 10 degree for roll axis
        self.setpoint_euler[0] = self.setpoint_cmd[0] * 0.02 - 30
        self.setpoint_euler[1] = self.setpoint_cmd[1] * 0.02 - 30
        self.setpoint_euler[2] = self.setpoint_cmd[2] *120/500 - 3*120  # -90 to 90
        #   Convertng the range of throttle from 1000 to 2000 in the range of 0-1024 (eg. 1500 - 512 pwm)
        self.setpoint_throttle = self.setpoint_cmd_throttle * 1.024 - 1024

        #   3. Compute error in each axis. eg: error[0] = self.setpoint_euler[0] - self.drone_orientation_euler[0], where error[0] corresponds to error in roll...

        #  roll

        self.error[0] = self.setpoint_euler[0] - self.drone_orientation_euler[0]
        #self.error[0] = 0 - self.drone_orientation_euler[0]
        self.Iterm[0] = self.Ki[0]*(self.Iterm[0] + self.error[0])

        #  pitch

        self.error[1] = self.setpoint_euler[1] - self.drone_orientation_euler[1]
        #self.error[1] = 0 - self.drone_orientation_euler[1]
        self.Iterm[1] = self.Ki[1]*(self.Iterm[1] + self.error[1])

        #  Yaw

        self.error[2] = self.setpoint_euler[2] - self.drone_orientation_euler[2]
        #self.error[2] = 0 - self.drone_orientation_euler[2]
        self.Iterm[2] = self.Ki[2]*(self.Iterm[2] + self.error[2])

        #   4. Compute the error (for proportional), change in error (for derivative) and sum of errors (for integral) in each axis. Refer "Understanding PID.pdf" to understand PID equation.


        #   5. Calculate the pid output required for each axis. For eg: calcuate self.out_roll, self.out_pitch, etc.
        #roll
        self.output[0] = ((self.Kp[0]*self.error[0]) + (self.Kd[0]* (self.error[0] - self.prev_error[0])) + (self.Iterm[0]))
        #pitch
        self.output[1] = ((self.Kp[1]*self.error[1]) + (self.Kd[1]* (self.error[1] - self.prev_error[1])) + (self.Iterm[1]))
        #yaw
        self.output[2] = ((self.Kp[2]*self.error[2]) + (self.Kd[2]* (self.error[2] - self.prev_error[2])) + (self.Iterm[2]))

        #3333#####################################################################################################################################################################


        if (self.output[2] < -300):
              self.output[2] = -300
        elif(self.output[2] > 300):
              self.output[2] = 300

        self.pwm_cmd.prop1 = self.setpoint_throttle + self.output[0]  - self.output[1] - self.output[2]
        self.pwm_cmd.prop2 = self.setpoint_throttle - self.output[0]  - self.output[1] + self.output[2]
        self.pwm_cmd.prop3 = self.setpoint_throttle - self.output[0]  + self.output[1] - self.output[2]
        self.pwm_cmd.prop4 = self.setpoint_throttle + self.output[0]  + self.output[1] + self.output[2]



        self.roll_error.data = self.error[0]
        self.pitch_error.data = self.error[1]
        self.yaw_error.data = self.error[2]


        #   8. Limit the output value and the final command value between the maximum(0) and minimum(1024)range before publishing. For eg : if self.pwm_cmd.prop1 > self.max_values[1]:
        #                                                                                                                                      self.pwm_cmd.prop1 = self.max_values[1]

        if self.pwm_cmd.prop1 > self.max_values[0]:
            self.pwm_cmd.prop1 = self.max_values[0]
        if self.pwm_cmd.prop1 < self.min_values[0]:
            self.pwm_cmd.prop1 = self.min_values[0]


        if self.pwm_cmd.prop2 > self.max_values[1]:
            self.pwm_cmd.prop2 = self.max_values[1]
        if self.pwm_cmd.prop2 < self.min_values[1]:
            self.pwm_cmd.prop2 = self.min_values[1]

        if self.pwm_cmd.prop3 > self.max_values[2]:
            self.pwm_cmd.prop3 = self.max_values[2]
        if self.pwm_cmd.prop3 < self.min_values[2]:
            self.pwm_cmd.prop3 = self.min_values[2]


        if self.pwm_cmd.prop4 > self.max_values[3]:
            self.pwm_cmd.prop4 = self.max_values[3]
        if self.pwm_cmd.prop4 < self.min_values[3]:
            self.pwm_cmd.prop4 = self.min_values[3]











        # Complete the equations for pitch and yaw axis

        # Also convert the range of 1000 to 2000 to 0 to 1024 for throttle here itslef

        #
        #
        #
        #
        #
        #
        #
        # ------------------------------------------------------------------------------------------------------------------------
        self.prev_error[0] = self.error[0]
        self.prev_error[1] = self.error[1]
        self.prev_error[2] = self.error[2]

        self.pwm_pub.publish(self.pwm_cmd)
        # self.roll_error_pub.publish(self.roll_error)
        # self.pitch_error_pub.publish(self.pitch_error)
        # self.yaw_error_pub.publish(self.yaw_error)
        # self.zero_error_pub.publish(self.zero_error_0)

        y = -(self.dest[0] - self.drone_position[0]) * self.meter_conv[0]
        x = -(self.dest[1] - self.drone_position[1]) * self.meter_conv[1]
        self.o = math.atan(y / x)



if __name__ == '__main__':

    e_drone = Edrone()
    r = rospy.Rate(20)  # specify rate in Hz based upon your desired PID sampling time, i.e. if desired sample time is 33ms specify rate as 30Hz
    while not rospy.is_shutdown():
        e_drone.pid()
        e_drone.obs_check()
        r.sleep()

#END OF ATTITUDE CONTROLLER CODE
