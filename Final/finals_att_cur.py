#!/usr/bin/env python

'''
# Team ID:          2537
# Theme:            Vitarana Drone
# Author List:      K.R.Jai Kesav, Greeshwar R.S, Aswin Sreekumar, K.Girish
# Filename:         Task_6_VD_2537_attitude_controller.py
# Functions:        im_breakin,range_top, edrone_position,imu_callback,drone_command_callback,pid

# Global variables:  drone_orientation_quaternion,drone_orientation_euler,self.setpoint_cmd,
                     setpoint_cmd_throttle,self.setpoint_euler,self.setpoint_throttle,Kp,Ki,
                     Kd,prev_error,error,Iterm,output,max_values,min_values,drone_position,meter_conv,rp_range

'''


# Importing the required libraries
from vitarana_drone.msg import *
from pid_tune.msg import PidTune
from sensor_msgs.msg import *
from std_msgs.msg import *
from geometry_msgs.msg import *
import rospy
import time
import tf
import math
import numpy as np
import cv2
import copy



def return2(i):
    return i[1]

class Edrone():
    """docstring for Edrone"""
    def __init__(self):
        rospy.init_node('attitude_controller')  # initializing ros node with name drone_control

        #drone_orientation_quaternion:corresponds to current orientation of eDrone in quaternion format[x,y,z,w]
        self.drone_orientation_quaternion = [0.0, 0.0, 0.0, 0.0]


        self.drone_orientation_euler = [0.0, 0.0, 0.0]
        self.prev_drone_orientation_euler = [0, 0, 0]
        self.setpoint_cmd = [1500, 1500, 1500]
        self.setpoint_cmd_throttle = 1000
        self.setpoint_euler = [0.0, 0.0, 0.0]
        self.setpoint_throttle = 0.0


        self.pwm_cmd = prop_speed()
        self.pwm_cmd.prop1 = 0.0
        self.pwm_cmd.prop2 = 0.0
        self.pwm_cmd.prop3 = 0.0
        self.pwm_cmd.prop4 = 0.0


        self.Kp = [0.006*500, 0.006*500, 0.019*1212]
        self.Ki = [0, 0, 0.35*0.008] # 0.7
        self.Kd = [ 0.03*300*2, 0.03*300*2, 0.08*980*2]

        self.prev_error = [0,0,0]  #[roll, pitch, yaw]
        self.error = [0,0,0]  # [roll, pitch, yaw]
        self.Iterm = [0,0,0]  # [roll, pitch, yaw]
        self.output = [0,0,0]  # [roll, pitch, yaw]

        #max_values: For storing maximum permissible value of rotors-[prop1, prop2, prop3, prop4]
        self.max_values = [1023, 1023, 1023, 1023]
        self.min_values = [1, 1, 1, 1]

        self.drone_position = [19, 72, 0.31]  # Current drone position in GPS coordinates
        self.meter_conv = [110693.227, 105078.267, 1]  # Factor for degrees to meters conversion(lat,long,alt to x,y,z)
        #rp_range
        self.rp_range = 2

        self.sample_time = 10  # in seconds

        self.Impulse_flag = False
        self.updated_yaw = 0
        self.prev_setpoint_yaw = 0
        self.yaw_offset = 0
        self.yaw_offset_flag = True

        self.position_setpoint = [19, 72, 10]
        self.controller_info = "0000"
        self.prev_controller_info = "0000"
        self.current_stage_index = 0
        self.unaltered_yaw_output = 0
        self.max_yaw_diff = -1
        self.prev_drone_velocity_res = 0
        self.drone_velocity_res = 0


        # Publishing /edrone/pwmr
        self.pwm_pub = rospy.Publisher('/edrone/pwm', prop_speed, queue_size=1)
        #self.roll_error_pub = rospy.Publisher('/roll_error',Float32, queue_size=1)
        #self.pitch_error_pub = rospy.Publisher('/pitch_error',Float32, queue_size=1)
        #self.yaw_error_pub = rospy.Publisher('/yaw_error',Float32, queue_size=1)
        #self.zero_error_pub = rospy.Publisher('/zero_error',Float32, queue_size=1)




        # Subscribing to /drone_command, imu/data,,/edrone/gps,/edrone/range_finder_top,/start_breaking
        rospy.Subscriber('/drone_command', edrone_cmd, self.drone_command_callback)
        rospy.Subscriber('/edrone/imu/data', Imu, self.imu_callback)
        rospy.Subscriber('/edrone/gps', NavSatFix, self.edrone_position)
        rospy.Subscriber('/edrone/range_finder_top', LaserScan, self.range_top)
        rospy.Subscriber('/start_breaking', Float32, self.im_breakin)
        rospy.Subscriber('/Impulse', Float32, self.Impulse_callback)
        rospy.Subscriber('Decided_Yaw', Float32, self.Yaw_callback )
        rospy.Subscriber('/SP', Vector3, self.Setpoint_callback)
        rospy.Subscriber('/controller_info', String, self.controller_info_func)
        rospy.Subscriber('/edrone/gps_velocity', Vector3Stamped, self.gps_velocity)

    def gps_velocity(self, gps):
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
        self.drone_velocity_res = math.sqrt((gps.vector.x) ** 2 + (gps.vector.y) ** 2)  # + (gps.vector.z)**2)


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


        self.prev_controller_info = str(self.controller_info)

        self.controller_info = str(data.data)
        if self.controller_info[1].isalpha():
            dict_a = {'z': 2.5, 'y': 6.5}
            self.current_stage_index = dict_a[self.controller_info[1]]
        else:
            self.current_stage_index = int(self.controller_info[1])


    def Setpoint_callback(self, data):
        self.position_setpoint = [data.x, data.y, data.z]

    def Yaw_callback(self, data):
        self.updated_yaw = data.data

    def Impulse_callback(self, data):
        if data.data == float(0):
            self.Impulse_flag = False
        elif data.data == float(1):
            self.Impulse_flag = True



    def im_breakin(self, start):
        '''
        Purpose:
        ---
        Callback function for braking

        Input Arguments:
        ---
        start: Float32

        Returns:
        ---
        NONE

        Example call:
        ---
        Called automatically by subscriber of /start_breaking
        '''

        self.rp_range = start.data

    # call back for sensor subscriber
    def range_top(self, range_top):
        '''
        Purpose:
        ---
        Callback function for top sensor in drone

        Input Arguments:
        ---
        range_top: Lasercan

        Returns:
        ---
        NONE

        Example call:
        ---
        Called automatically by subscriber of /edrone/range_finder_top
        '''

        self.range_top_dist = range_top.ranges



    def edrone_position(self, gps):
        '''
        Purpose:
        ---
        Callback function for GPS coordinates. Updates value of drone_position accordingly

        Input Arguments:
        ---
        gps: NavSatFix

        Returns:
        ---
        NONE

        Example call:
        ---
        Called automatically by subscriber of /edrone/gps
        '''

        self.drone_position[0] = gps.latitude
        self.drone_position[1] = gps.longitude
        self.drone_position[2] = gps.altitude

    def imu_callback(self, msg):
        '''
        Purpose:
        ---
        Callback function for IMU sensor. Converts quaternion to euler angles

        Input Arguments:
        ---
        msg: Imu

        Returns:
        ---
        NONE

        Example call:
        ---
        Called automatically by subscriber of /edrone/imu/data
        '''

        self.prev_drone_orientation_euler = list(self.drone_orientation_euler)

        self.drone_orientation_quaternion[0] = msg.orientation.x
        self.drone_orientation_quaternion[1] = msg.orientation.y
        self.drone_orientation_quaternion[2] = msg.orientation.z
        self.drone_orientation_quaternion[3] = msg.orientation.w

        (self.drone_orientation_euler[0], self.drone_orientation_euler[1], self.drone_orientation_euler[2]) = tf.transformations.euler_from_quaternion([self.drone_orientation_quaternion[0], self.drone_orientation_quaternion[1], self.drone_orientation_quaternion[2], self.drone_orientation_quaternion[3]])

        self.drone_orientation_euler[0] *= 180 / math.pi
        self.drone_orientation_euler[1] *= 180 / math.pi
        self.drone_orientation_euler[2] *= 180 / math.pi

        # print("euler : ", self.drone_orientation_euler)


    def drone_command_callback(self, msg):
        '''
        Purpose:
        ---
        Callback function for roll,pitch,yaw and throttle. Recieves roll,pitch,yaw and throttle
        setpoints from Task_6_VD_2537_Position_controller

        Input Arguments:
        ---
        msg: edrone_cmd

        Returns:
        ---
        NONE

        Example call:
        ---
        called automatically by Subscriber of /drone_command
        '''

        self.setpoint_cmd[0] = msg.rcRoll
        self.setpoint_cmd[1] = msg.rcPitch
        self.setpoint_cmd[2] = msg.rcYaw
        self.setpoint_cmd_throttle = msg.rcThrottle




    '''
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
    '''




    def pid(self):
        '''
        Purpose:
        ---
        Main PID function. Computes the error in roll, pitch and yaw.
        Output: Kp*error + Kd*(error-prev_error) + Ki*(error+(sum of previous error))
        Sampling time is constant.
        Feeds the output values of each rotation axis to the propellors as proper linear combinations.
        Publishes /edrone/pwm
        Updates prev_error accordingly

        Input Arguments:
        ---
        NONE

        Returns:
        ---
        NONE

        Example call:
        ---
        <instance of class>.pid()
        '''

        for i in range(2):
            # if self.setpoint_cmd[i] == 3000 or self.setpoint_cmd[i] == 0:
            #     if self.setpoint_cmd[i] == 3000:
            #         self.setpoint_euler[i] = 40
            #     else:
            #         self.setpoint_euler[i] = -40
            # else:
            self.setpoint_euler[i] = self.setpoint_cmd[i] * self.rp_range/ 500 - self.rp_range* 3

        # self.setpoint_euler[2] = self.setpoint_cmd[2] *120/500 - 3*120  # -120 to 120
        #Convertng the range of throttle from 1000 to 2000 in the range of 0-1024 (eg. 1500 - 512 pwm)
        self.setpoint_throttle = self.setpoint_cmd_throttle * 1.024 - 1024

        # if self.setpoint_throttle > 800:
        #     self.setpoint_throttle = 800
        # elif self.setpoint_throttle < 200:
        #     self.setpoint_throttle = 200

        if self.updated_yaw != self.setpoint_euler[2] or (self.prev_controller_info[1]!=self.controller_info[1] and self.controller_info[1] in ['z', 'y']):
            self.Iterm[2] = 0
            self.yaw_offset_flag = True
            self.yaw_offset = 0
            self.max_yaw_diff = -1
            print("updated yaw")

        self.setpoint_euler[2] = self.updated_yaw + 0
        # self.prev_setpoint_yaw = float(self.setpoint_euler[2])


        if self.Impulse_flag:
            self.setpoint_throttle = 1024
            for i in range(2):
                # self.setpoint_euler[i] = - self.drone_orientation_euler[i]
                self.setpoint_euler[i] = -self.setpoint_euler[i]


        self.error[0] = self.setpoint_euler[0] - self.drone_orientation_euler[0]

        self.Iterm[0] = self.Ki[0]*(self.Iterm[0] + self.error[0])

        #  pitch

        self.error[1] = self.setpoint_euler[1] - self.drone_orientation_euler[1]

        self.Iterm[1] = self.Ki[1]*(self.Iterm[1] + self.error[1])

        #  Yaw

        self.error[2] = self.setpoint_euler[2] - self.drone_orientation_euler[2]

        self.Iterm[2] += self.Ki[2]*self.error[2]


        kd_term = self.drone_orientation_euler[2] - self.prev_drone_orientation_euler[2]
        if abs(kd_term) > self.max_yaw_diff:
            self.max_yaw_diff = abs(kd_term)

        if abs(self.error[2]) > 1 and (abs(kd_term) < 0.0002 and abs(kd_term) < self.max_yaw_diff/5) and (self.controller_info[1] in ['z', 'y', '3', '4']) and abs(self.position_setpoint[2] - self.drone_position[2]) < 0.5 and self.drone_velocity_res < 0.3:
            self.yaw_offset_flag = False
            self.yaw_offset = float(self.output[2])

        # elif abs(self.error[2]) < 1:
        #     self.yaw_offset_flag = True
        #     self.yaw_offset = 0

        print("Iterm : ", self.Iterm)
        print("KD term : ", kd_term)
        print("yaw_offset : ", self.yaw_offset)
        print("yaw_offset_flag : ", self.yaw_offset_flag)

        #output calculation for each axis of rotation
        #roll
        self.output[0] = (self.Kp[0]*self.error[0]) + (self.Kd[0]* (self.error[0] - self.prev_error[0])) + (self.Iterm[0])
        #pitch
        self.output[1] = (self.Kp[1]*self.error[1]) + (self.Kd[1]* (self.error[1] - self.prev_error[1])) + (self.Iterm[1])
        #yaw
        self.output[2] = (self.Kp[2]*self.error[2]) + (self.Kd[2]* (self.error[2] - self.prev_error[2])) + self.yaw_offset

        self.unaltered_yaw_output = float(self.output[2])

        #3333#####################################################################################################################################################################

        # if (self.output[2] < -200):
        #     self.output[2] = -200
        # elif (self.output[2] > 200):
        #     self.output[2] = 200

        output_culmination = [self.output[0] - self.output[1] - self.output[2], - self.output[0] - self.output[1] + self.output[2], - self.output[0] + self.output[1] - self.output[2], self.output[0] + self.output[1] + self.output[2]]
        # print("culmination : ", output_culmination)
        print("initial throttle : ", self.setpoint_throttle)
        print("initial self.output[2] : ", self.output[2])

        max_negative = float('inf')
        max_positive = -float('inf')
        for i in output_culmination:
            if i < 0 and i < max_negative:
                max_negative = i
            if i >=0 and i > max_positive:
                max_positive = i

        if max_positive == -float('inf'):
            max_positive = 0
        if max_negative == float('inf'):
            max_negative = 0

        print("max_positive : ", max_positive)
        print("max_negative : ", max_negative)

        if abs(max_negative) + abs(max_positive) <= 1024:
            print("AVAILABLE !")
            if self.setpoint_throttle > abs(max_negative) and self.setpoint_throttle < 1024 - abs(max_positive):
                pass
            else:
                if self.setpoint_throttle < abs(max_negative):
                    self.setpoint_throttle = abs(max_negative) + 2
                else:
                    self.setpoint_throttle = 1024 - abs(max_positive) - 2

        else:
            print("CUTOFF!")
            cutoff = min(self.setpoint_throttle/4, (1024 - self.setpoint_throttle)/4)
            for j in range(2):
                if (self.output[j] < -cutoff):
                      self.output[j] = -cutoff
                elif(self.output[j] > cutoff):
                      self.output[j] = cutoff

            output_culmination_for_yaw = [self.output[0] - self.output[1] - self.output[2],
                                  - self.output[0] - self.output[1] + self.output[2],
                                  - self.output[0] + self.output[1] - self.output[2],
                                  self.output[0] + self.output[1] + self.output[2]]
            maxi = -float('inf')
            mini = float('inf')
            for i in range(4):
                x = output_culmination_for_yaw[i] + self.setpoint_throttle
                print(x)
                if x <= 0 and x < mini:
                    mini = x + 0
                elif x >= 1024 and x - 1024 > maxi:
                    maxi = x - 1024

            print("maxi",  maxi)
            print("mini", mini)

            if maxi == -float('inf'):
                maxi = 0
            if mini == float('inf'):
                mini = 0

            chop_off = max(abs(maxi), abs(mini))

            if self.output[2] < 0:
                self.output[2] += chop_off
            elif self.output[2] > 0:
                self.output[2] -= chop_off
            else:
                print("no probs with yaw !")

        print("final throttle : ", self.setpoint_throttle)
        print("output[2]", self.output[2])

        output_culmination_for_yaw = [self.output[0] - self.output[1] - self.output[2],
                                      - self.output[0] - self.output[1] + self.output[2],
                                      - self.output[0] + self.output[1] - self.output[2],
                                      self.output[0] + self.output[1] + self.output[2]]

        for i in range(4):
            x = output_culmination_for_yaw[i] + self.setpoint_throttle
            print(x)

        self.pwm_cmd.prop1 = self.setpoint_throttle + self.output[0]  - self.output[1] - self.output[2]
        self.pwm_cmd.prop2 = self.setpoint_throttle - self.output[0]  - self.output[1] + self.output[2]
        self.pwm_cmd.prop3 = self.setpoint_throttle - self.output[0]  + self.output[1] - self.output[2]
        self.pwm_cmd.prop4 = self.setpoint_throttle + self.output[0]  + self.output[1] + self.output[2]

        print("prop_val : ", self.pwm_cmd)

        # if self.Impulse_flag:
        #     self.pwm_cmd.prop1 = 1024
        #     self.pwm_cmd.prop2 = 1024
        #     self.pwm_cmd.prop3 = 1024
        #     self.pwm_cmd.prop4 = 1024
        #
        # print("Impulse flag : ", self.Impulse_flag)


        #self.roll_error.data = self.error[0]
        #self.pitch_error.data = self.error[1]
        #self.yaw_error.data = self.error[2]


        #Limiting the propellor values between the given range.
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



        self.prev_error[0] = float(self.error[0])
        self.prev_error[1] = float(self.error[1])
        self.prev_error[2] = float(self.error[2])

        # # DEBUG
        # self.pwm_cmd.prop1 = 422.899679389
        # prop2: 594.917183514
        # prop3: 422.899210544
        # prop4: 594.916406625


        self.pwm_pub.publish(self.pwm_cmd)
        # self.roll_error_pub.publish(self.roll_error)
        # self.pitch_error_pub.publish(self.pitch_error)
        # self.yaw_error_pub.publish(self.yaw_error)
        # self.zero_error_pub.publish(self.zero_error_0)

        # print("error : ", self.error)
        # print("Output : ", self.output)
        # print("pwm values : ", self.pwm_cmd)




if __name__ == '__main__':

    e_drone = Edrone()
    r = rospy.Rate(20)  # specify rate in Hz based upon your desired PID sampling time, i.e. if desired sample time is 33ms specify rate as 30Hz
    while not rospy.is_shutdown():
        e_drone.pid()
        r.sleep()

#END OF ATTITUDE CONTROLLER CODE
