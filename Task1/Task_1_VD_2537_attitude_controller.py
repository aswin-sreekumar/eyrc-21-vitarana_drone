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
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
import rospy
import time
import tf
import math


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

        self.roll_error = error_rpyt()
        self.roll_error.data = 0.0
        self.pitch_error = error_rpyt()
        self.pitch_error.data = 0.0
        self.yaw_error = error_rpyt()
        self.yaw_error.data = 0.0
        self.zero_error_0 = error_rpyt()
        self.zero_error_0.data = 0.0




        # initial setting of Kp, Kd and ki for [roll, pitch, yaw]. eg: self.Kp[2] corresponds to Kp value in yaw axis
        # after tuning and computing corresponding PID parameters, change the parameters
        self.Kp = [0.006*500, 0.006*500, 0.019*1212]
        self.Ki = [0, 0, 0]
        self.Kd = [ 0.03*300, 0.03*300, 0.08*980]
        # -----------------------Add other required variables for pid here ----------------------------------------------
        #
        self.prev_error = [0,0,0]  #[roll, pitch, yaw]
        self.error = [0,0,0] #[roll, pitch, yaw]
        self.Iterm = [0,0,0] #[roll, pitch, yaw]
        self.output = [0,0,0] #[roll, pitch, yaw]

        self.max_values = [1024, 1024, 1024, 1024] #[prop1, prop2, prop3, prop4]
        self.min_values = [0, 0, 0, 0]

        # Hint : Add variables for storing previous errors in each axis, like self.prev_values = [0,0,0] where corresponds to [roll, pitch, yaw]
        #        Add variables for limiting the values like self.max_values = [1024, 1024, 1024, 1024] corresponding to [prop1, prop2, prop3, prop4]
        #                                                   self.min_values = [0, 0, 0, 0] corresponding to [prop1, prop2, prop3, prop4]
        #
        # ----------------------------------------------------------------------------------------------------------

        # # This is the sample time in which you need to run pid. Choose any time which you seem fit. Remember the stimulation step time is 50 ms
        self.sample_time = 10  # in seconds

        # Publishing /edrone/pwm, /roll_error, /pitch_error, /yaw_error
        self.pwm_pub = rospy.Publisher('/edrone/pwm', prop_speed, queue_size=1)
        self.roll_error_pub = rospy.Publisher('/roll_error',error_rpyt, queue_size=1)
        self.pitch_error_pub = rospy.Publisher('/pitch_error',error_rpyt, queue_size=1)
        self.yaw_error_pub = rospy.Publisher('/yaw_error',error_rpyt, queue_size=1)
        self.zero_error_pub = rospy.Publisher('/zero_error',error_rpyt, queue_size=1)
        # ------------------------Add other ROS Publishers here-----------------------------------------------------

        # -----------------------------------------------------------------------------------------------------------

        # Subscribing to /drone_command, imu/data, /pid_tuning_roll, /pid_tuning_pitch, /pid_tuning_yaw
        rospy.Subscriber('/drone_command', edrone_cmd, self.drone_command_callback)
        rospy.Subscriber('/edrone/imu/data', Imu, self.imu_callback)

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

    def imu_callback(self, msg):

        self.drone_orientation_quaternion[0] = msg.orientation.x
        self.drone_orientation_quaternion[1] = msg.orientation.y
        self.drone_orientation_quaternion[2] = msg.orientation.z
        self.drone_orientation_quaternion[3] = msg.orientation.w


        # --------------------Set the remaining co-ordinates of the drone from msg----------------------------------------------

    def drone_command_callback(self, msg):
        self.setpoint_cmd[0] = msg.rcRoll
        self.setpoint_cmd[1] = msg.rcPitch
        self.setpoint_cmd[2] = msg.rcYaw
        self.setpoint_cmd_throttle = msg.rcThrottle


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
        (self.drone_orientation_euler[0], self.drone_orientation_euler[1], self.drone_orientation_euler[2]) = tf.transformations.euler_from_quaternion([self.drone_orientation_quaternion[0], self.drone_orientation_quaternion[1], self.drone_orientation_quaternion[2], self.drone_orientation_quaternion[3]])

        # 2.Convertng the range from 1000 to 2000 in the range of -10 degree to 10 degree for roll axis
        self.setpoint_euler[0] = self.setpoint_cmd[0] * 0.02 - 30
        self.setpoint_euler[1] = self.setpoint_cmd[1] * 0.02 - 30
        self.setpoint_euler[2] = self.setpoint_cmd[2] * 0.02 - 30
        #   Convertng the range of throttle from 1000 to 2000 in the range of 0-1024 (eg. 1500 - 512 pwm)
        self.setpoint_throttle = self.setpoint_cmd_throttle * 1.024 - 1024



        # radians to degree
        self.drone_orientation_euler[0] *= 180/math.pi
        self.drone_orientation_euler[1] *= 180/math.pi
        self.drone_orientation_euler[2] *= 180/math.pi
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
        self.roll_error_pub.publish(self.roll_error)
        self.pitch_error_pub.publish(self.pitch_error)
        self.yaw_error_pub.publish(self.yaw_error)
        self.zero_error_pub.publish(self.zero_error_0)




if __name__ == '__main__':

    e_drone = Edrone()
    r = rospy.Rate(e_drone.sample_time)  # specify rate in Hz based upon your desired PID sampling time, i.e. if desired sample time is 33ms specify rate as 30Hz
    while not rospy.is_shutdown():
        e_drone.pid()
        print("orientation (roll pitch yaw) :",e_drone.drone_orientation_euler)
        print("error :",e_drone.error)
        print("prop1_pwm :",e_drone.pwm_cmd.prop1)
        print("prop2_pwm :",e_drone.pwm_cmd.prop2)
        print("prop3_pwm :",e_drone.pwm_cmd.prop3)
        print("prop4_pwm :",e_drone.pwm_cmd.prop4)
        print("output_roll :",e_drone.output[0])
        print("output_pitch :",e_drone.output[1])
        print("output_yaw :",e_drone.output[2])
        print("kp :",e_drone.Kp)
        print("kd :",e_drone.Kd)
        print("ki :",e_drone.Ki)
        print("iterm :",e_drone.Iterm)
        r.sleep()

#END OF ATTITUDE CONTROLLER CODE
