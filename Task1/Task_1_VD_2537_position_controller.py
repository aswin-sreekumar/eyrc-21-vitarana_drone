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


'''
Algorithm :
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

        self.Kp = [1638*0.04, 1638*0.04, 1.0, 1.0, 136*0.6/1.024]  # idx 0- roll, 1-pitch, 2-yaw, 3-throttle, 4-eq_throttle
        self.Ki = [0.0, 0.0, 0.0, 0.0, 192*0.08/1.024]  # Equilibrium control as of now
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
        self.accumulator = 0 #Position error for integrating
        self.error_distance = 0
        self.error_threshold = 0.10  # used in def stability
        self.velocity_threshold = 0.10    # used in def stability

        self.meter_conv = [110693.227, 105078.267, 1] #Factor for degrees to meters conversion(lat,long,alt to x,y,z)

        self.cosines = [0, 0, 0]#Velocity control using direction cosines

        self.proceed = True #Flag variable for checking state
        self.state = [False, False, False] #Variable for axes state checking
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

        self.position_setpoint = [19 , 72 , 3 ]
        self.vel = [0, 0, 0]

        self.assign()

        # SUBSCRIBERS

        rospy.Subscriber('/edrone/gps', NavSatFix, self.edrone_position)
        #rospy.Subscriber('/set_setpoint', SetPoints, self.set_setpoints)
        rospy.Subscriber('/edrone/gps_velocity', Vector3Stamped, self.get_drone_vel)

        #rospy.Subscriber('/pid_tuning_roll', PidTune, self.roll_set_pid)
        #rospy.Subscriber('/pid_tuning_pitch', PidTune, self.pitch_set_pid)
        #rospy.Subscriber('/pid_tuning_yaw', PidTune, self.yaw_set_pid)

        # PUBLISHERS

        self.drone_command = rospy.Publisher('/drone_command', edrone_cmd, queue_size = 1)



    #Function for controlling position
    def position_control(self):
        '''
        if not self.proceed:
            self.equilibrium()

        else:
        '''

        self.travel()

        for i in range(3):
            self.stability(i)

        rospy.loginfo("Latitude, Longitude and Altitude Stabilities")
        rospy.loginfo(self.state)  # The drone is considered stable if it reaches the set setpoints and the velocity and position errors satisfy the given thresholds
        rospy.loginfo("Latitude, Longitude and Altitude Errors")
        rospy.loginfo(self.error)

        self.set_setpoints_auto()


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

    def equilibrium(self):

        for i in range(3):
            self.error[i] = (self.target[i] - self.drone_position[i]) * self.meter_conv[i]
            self.error_derivative[i] = (self.error[i] - self.prev_error[i]) / self.sample_time
            self.prev_error[i] = self.error[i]

        self.accumulator += self.error[2]* self.sample_time

        for i in range(2):
            self.rcval[i] = self.confine(self.eq_rcval[i] + self.Kp2[i] * self.error[i] + self.Kd2[i] * self.error_derivative[i])

        self.rcval[3] = self.confine(self.eq_rcval[3] + self.Kp2[3] * self.error[2] + self.Kd2[3] * self.error_derivative[2])
        self.rcval[3] = self.rcval[3] / abs(math.cos(self.drone_orientation_euler[0]) * math.cos(self.drone_orientation_euler[1]))

        if self.state[0]:
            self.eq_rcval[0] = self.command.rcRoll
        if self.state[1]:
            self.eq_rcval[1] = self.command.rcPitch
        if self.state[2]:
            self.eq_rcval[3] = self.command.rcThrottle
            self.accumulator = 0

        if self.state[0] and self.state[1] and self.state[2]:
            self.proceed = True

        self.assign()
        self.drone_command.publish(self.command)
        rospy.loginfo("In equilib.")

        '''
        for i in range(2):

            self.error[i] = (self.target[i] - self.drone_position[i])*self.meter_conv[i]
            self.error_derivative[i] = self.Kd[i]*(self.error[i] - self.prev_error[i])/self.sample_time
            self.error_derivative[i] = self.thresh_derivative(self.error_derivative[i])
            self.rcval[i] = self.confine(self.eq_rcval[i] + self.Kp[i] * self.error[i] + self.error_derivative[i])
            self.prev_error[i] = self.error[i]
            if i == 0 and self.state[i]:
                self.eq_rcval[i] = self.command.rcRoll
            elif i == 1 and self.state[i]:
                self.eq_rcval[i] = self.command.rcPitch

        self.error[2] = (self.target[2] - self.drone_position[2])*self.meter_conv[2]
        self.error_derivative[2] = self.Kd[4] * (self.error[2] - self.prev_error[2]) / self.sample_time
        self.error_derivative[2] = self.thresh_derivative(self.error_derivative[2])
        self.accumulator += self.error[2]*self.sample_time
        self.rcval[3] = self.confine(self.eq_rcval[3] + self.Kp[4]*self.error[2] + self.error_derivative[2] + self.Ki[4]*self.accumulator)
        self.prev_error[2] = self.error[2]
        if self.state[i]:
            self.eq_rcval[3] = self.command.rcThrottle
            self.accumulator = 0

        if self.state[0] and self.state[1] and self.state[2]:
            self.proceed = True

        self.assign()
        self.drone_command.publish(self.command)
        rospy.loginfo("In equilib.")

        '''



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
            self.cosines[i] = self.error[i]/self.error_distance

        #This block is the threshold absolute distance for velocity control(Not executed for this task,will be used for future tasks)
        if abs(self.initial_error_distance) > 100:  # a value of 100 is given for for now, so as to make sure that the else statement is always initiated for this task
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


    #Function to check whether it has reached the setpoint stable
    def stability(self, idx):

        if self.proceed:
            if abs(self.position_setpoint[idx] - self.drone_position[idx])*self.meter_conv[idx] < self.error_threshold and abs(self.vel[idx]) < self.velocity_threshold:
                self.state[idx] = True
            else:
                self.state[idx] = False

        else:
            if abs(self.target[idx] - self.drone_position[idx])*self.meter_conv[idx] < self.error_threshold and abs(self.vel[idx]) < self.velocity_threshold:
                self.state[idx] = True
            else:
                self.state[idx] = False

    #Tuning parameters during development and debugging
    def roll_set_pid(self, roll):
        self.Kp2[0] = roll.Kp * 0.04  # This is just for an example. You can change the ratio/fraction value accordingly
        self.Ki2[0] = roll.Ki * 0.00
        self.Kd2[0] = roll.Kd * 0.3

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
        drone_boi.position_control()
        r.sleep()

# END OF POSITION CONTROLLER PROGRAM
