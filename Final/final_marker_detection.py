#!/usr/bin/env python


'''
# Team ID:          VD_2537
# Theme:            Vitarana Drone
# Author List:      Jai Kesav, Aswin Sreekumar, Girish K, Greeshwar R S
# Filename:         Task_6_VD_2537_marker_detection

# Functions:        __init__(), imu_callback(), camera_info_update(), marker_detect_start(), build_alt_update(), file_read_func(), publish_val(), update_allow()
                    x_error(), y_error(), marker_id(), find_marker_reply_fun(), range_bottom(), edrone_position(), pixel_dist(), cascade(), coord_dec(), image_callback()

# Global variables: img_loc, manifest_csv_loc, cascade_xml_loc, image_sub, img, bridge, img_1, ROI, hfov_rad, img_width, Zm, Zm_bott, drone_position, do, second_r, one_t, allow
                    building_alt, package_drop_coordinates, total_package_number, meter_conv, logo, marker_detection, sequenced_package_index, package_index
                    prev_sequenced_package_index

'''

'''
What this script does:




'''

from vitarana_drone.msg import *
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
#from pyzbar.pyzbar import decode
import rospy
import math
import csv
import copy
import tf


class image_proc():

    # Initialise everything
    def __init__(self):

        #GREE
        # self.img_loc = '/home/greesh/PycharmProjects/chumma/map_imp.jpg'
        # self.manifest_csv_loc = '/home/greesh/catkin_ws/src/vitarana_drone/scripts/finals/sequenced_manifest_original.csv'
        # self.cascade_xml_loc = '/home/greesh/PycharmProjects/chumma/intro_cascade_classifiers_training_and_usage/data/cascade.xml'

        # JK
        self.img_loc = '/home/jk56/PycharmProjects/pythonProject/map_img.jpg'
        self.manifest_csv_loc = '/home/jk56/catkin_ws/src/vitarana_drone/scripts/manifest.csv'
        self.cascade_xml_loc = '/home/jk56/catkin_ws/src/vitarana_drone/scripts/cascade.xml'

        rospy.init_node('detector')  # Initialise rosnode
        self.image_sub = rospy.Subscriber("/edrone/camera/image_raw", Image,self.image_callback)  # Subscribing to the camera topic

        self.img = np.empty([])  # This will contain your image frame from camera
        self.bridge = CvBridge()
        self.img_1 = np.ones((8, 8, 3), np.uint8)
        self.ROI = np.ones((8, 8, 3), np.uint8)

        self.hfov_rad = 1.3962634
        self.img_width = 400
        self.Zm = 0.31          # same values as range_finder_bottom
        self.Zm_bott = 0.31  # same values as range_finder_bottom
        self.drone_position = [19, 72, 0.31]  # Current drone position in GPS coordinates

        self.do = 0    # denotes each phase of detection
        self.second_r = 0
        self.one_t = 0
        self.allow = 1
        self.building_alt = 0

        self.package_drop_coordinates = []
        self.total_package_number = 0
        # for obtaining and storing the building coordinates from csv file
        self.file_read_func()
        self.meter_conv = [110692.0702932625, 105292.0089353767, 1]   # Factor for degrees to meters conversion(lat,long,alt to x,y,z)

        self.logo = []

        self.marker_detection = 1

        self.sequenced_package_index = 0
        self.package_index = -1
        self.prev_sequenced_package_index = -1

        self.updated_yaw = 0

        # Subscribers
        rospy.Subscriber('/edrone/gps', NavSatFix, self.edrone_position)
        rospy.Subscriber('/edrone/range_finder_bottom', LaserScan, self.range_bottom)
        rospy.Subscriber('/find_marker_reply', Float32, self.find_marker_reply_fun)

        rospy.Subscriber('/err_x_m_pos', Float32, self.x_error)
        rospy.Subscriber('/err_y_m_pos', Float32, self.y_error)
        rospy.Subscriber('/curr_marker_id_pos', Float32, self.marker_id)
        rospy.Subscriber('/allow', Float32, self.update_allow)
        rospy.Subscriber('/edrone/gps', NavSatFix, self.edrone_position)

        rospy.Subscriber('/marker_detection_start', Float32, self.marker_detect_start)
        rospy.Subscriber('/current_package_index',String,self.build_alt_update)
        rospy.Subscriber('/edrone/camera/camera_info', CameraInfo, self.camera_info_update)
        rospy.Subscriber('/edrone/imu/data', Imu, self.imu_callback)
        rospy.Subscriber('Decided_Yaw', Float32, self.Yaw_callback)



        # Publishers
        self.found1_pub = rospy.Publisher('/find_marker',String,queue_size = 1)
        self.found1 = String()
        self.found1.data = ""

        self.err_x_m_pub = rospy.Publisher('/edrone/err_x_m', Float32, queue_size=1)
        self.err_y_m_pub = rospy.Publisher('/edrone/err_y_m', Float32, queue_size=1)
        self.curr_marker_id_pub = rospy.Publisher('/edrone/curr_marker_id', Float32, queue_size=1)

        self.err_x_m = Float32()
        self.err_y_m = Float32()
        self.curr_marker_id = Float32()

        self.err_x_m.data = 0
        self.err_y_m.data = 0
        self.curr_marker_id.data = 2

        #new variable for matrix based reversal
        self.P = []
        self.K_inv = []
        self.R = []
        self.D = []
        self.K = []
        self.drone_orientation_quaternion = [0, 0, 0, 0]
        self.drone_orientation_euler = [0, 0, 0]


    def Yaw_callback(self, data):
        self.updated_yaw = data.data

    def imu_callback(self, msg):
        '''
                Purpose:
                ---
                Callback function of IMU sensor

                Input Arguments:
                ---
                `msg` :  [Imu]
                Subscribes to the current quaternion coordinates of the drone from Imu sensors of Imu message

                < name of 2nd input argument >` :  [ < type of 2nd input argument > ]
                    < one-line description of 2nd input argument >

                Returns:
                ---
                NONE


                Example call:
                ---
                Called automatically by subscriber of /Imu/data
        '''

        self.drone_orientation_quaternion[0] = msg.orientation.x
        self.drone_orientation_quaternion[1] = msg.orientation.y
        self.drone_orientation_quaternion[2] = msg.orientation.z
        self.drone_orientation_quaternion[3] = msg.orientation.w

        (self.drone_orientation_euler[0], self.drone_orientation_euler[1],
         self.drone_orientation_euler[2]) = tf.transformations.euler_from_quaternion(
            [self.drone_orientation_quaternion[0], self.drone_orientation_quaternion[1],
             self.drone_orientation_quaternion[2], self.drone_orientation_quaternion[3]])

        self.drone_orientation_euler[1] *= -1


    def camera_info_update(self, ci):
        '''
                Purpose:
                ---
                for getting the camera parameters

                Input Arguments:
                ---
                `ci` :  [ CameraInfo ]
                    camera feed



                Returns:
                ---
                NONE
                Example call:
                ---
                Called automatically by subscriber of /edrone/camera/camera_info
        '''
        # self.P = [ci.P[0:4], ci.P[4:8], ci.P[8:12]]
        self.P = ci.P
        self.K = [ci.K[0:3], ci.K[3:6], ci.K[6:9]]
        self.R = ci.R
        self.D = ci.D

    def marker_detect_start(self, data):
        '''
                Purpose:
                ---
                to start or stop marker detection (that is to publish the marker coordinates or text related to marker detection stages to position control)

                Input Arguments:
                ---
                `data` :  [ Float32]
                    0 to stop detection,1 to start detection

                Returns:
                ---
                NONE

                Example call:
                ---
                Called automatically by Subscriber of /marker_detectifeed

on_start
        '''
        if data.data == 1:
            self.marker_detection = 1
        else:
            self.marker_detection = 0

    def build_alt_update(self, index):
        '''
                Purpose:
                ---
                for updating building altitude for Zm calculation in marker detection

                Input Arguments:
                ---
                `index` : [String]
                    index of packages to get the altitude of corresponding building on which marker detection is to be done.


                Returns:
                ---
                NONE

                Example call:
                ---
                gets called automatically by subscriber of /current_package_index topic
        '''
        if str(index.data[0]).isalpha():
            list_a = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            self.sequenced_package_index = 10 + list_a.index(str(index.data[0]))
        else:
            self.sequenced_package_index = int(index.data[0])

        if index.data[1] == "D" and self.prev_sequenced_package_index != self.sequenced_package_index:
            self.package_index += 1
            self.building_alt = self.package_drop_coordinates[self.sequenced_package_index][2]
            self.prev_sequenced_package_index = self.sequenced_package_index


    # Extract data from csv file
    def file_read_func(self):
        '''
                Purpose:
                ---
                for reading the csv file and storing the building coordinates(delivery coords)

                Input Arguments:
                ---
                NONE

                Returns:
                ---
                NONE
                Example call:
                ---
                file_read_func()
        '''
        with open(self.manifest_csv_loc) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[0] == "DELIVERY":
                    package_list = row[2].split(';')
                    self.package_drop_coordinates.append([float(package_list[0]),float(package_list[1]),float(package_list[2])])
                    self.total_package_number += 1
                else:
                    self.package_drop_coordinates.append('pongada')


    def publish_val(self):
        '''
                Purpose:
                ---
                < Short-text describing the purpose of this function >

                Input Arguments:
                ---
                NONE

                Returns:
                ---
                NONE

                Example call:
                ---
                publish_val()
        '''

        if self.allow == 1:
            self.err_x_m_pub.publish(self.err_x_m)
            self.err_y_m_pub.publish(self.err_y_m)
        self.curr_marker_id_pub.publish(self.curr_marker_id)

    def update_allow(self, data):
        '''
                Purpose:
                ---
                < Short-text describing the purpose of this function >

                Input Arguments:
                ---
                `data` :  [ < type of 1st input argument > ]
                    < one-line description of 1st input argument >



                Returns:
                ---
                NONE
                Example call:
                ---
                < Example of how to call this function >
                '''
        self.allow = data.data

    def x_error(self, data):
        '''
                Purpose:
                ---
                < Short-text describing the purpose of this function >

                Input Arguments:
                ---
                `data` :  [ < type of 1st input argument > ]
                    < one-line description of 1st input argument >



                Returns:
                ---
                NONE

                Example call:
                ---
                < Example of how to call this function >
        '''
        self.err_x_m = data.data

    def y_error(self, data):
        '''
                Purpose:
                ---
                < Short-text describing the purpose of this function >

                Input Arguments:
                ---
                `data` :  [ < type of 1st input argument > ]
                    < one-line description of 1st input argument >



                Returns:
                ---
                NONE

                Example call:
                ---
                < Example of how to call this function >
        '''
        self.err_y_m = data.data

    def marker_id(self, data):
        '''
                Purpose:
                ---
                < Short-text describing the purpose of this function >

                Input Arguments:
                ---
                `data` :  [ < type of 1st input argument > ]
                    < one-line description of 1st input argument >



                Returns:
                NONE

                Example call:
                ---
                ---
                < Example of how to call this function >
        '''
        self.curr_marker_id = data.data

    def find_marker_reply_fun(self,reply):
        '''
                Purpose:
                ---
                for getting the stage number from position control(find_marker function)

                Input Arguments:
                ---
                `reply` :  [Float32]
                    its from 0 -5 corresponding to the stages of marker detection



                Returns:
                ---
                NONE
                Example call:
                ---
                calls automatically by subscriber of /find_marker_reply topic
        '''
        self.do = reply.data


    def range_bottom(self,bottom):
        '''
                Purpose:
                ---
                Callback function for bottom sensor

                Input Arguments:
                ---
                `bottom` :  [ LaserScan ]
                    for getting bottom sensor obstacle distance


                Returns:
                ---
                NONE

                Example call:
                ---
                Called automatically by subscriber of /edrone/range_finder_bottom
        '''

        # if bottom.ranges[0] >= 0.4:
        #     self.Zm = bottom.ranges[0] #sum(self.prev_bottom_val)/10
        # else:
        self.Zm = self.drone_position[2] - self.building_alt


    #callback function for drone position
    def edrone_position(self, gps):
        '''
                Purpose:
                ---
                callback function for edrone GPS

                Input Arguments:
                ---
                `gps` :  [ NavSatFix]
                    Gets the GPS coordinates (latitude, longtitude, altitude)



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


    def pixel_dist(self,centre_x_pixel,centre_y_pixel):

        '''
                Purpose:
                ---
                for calculating the location of marker in terms of degrees(lat ,long)

                Input Arguments:
                ---
                `centre_x_pixel` :  [ int ]
                    x location of marker centre on the image in pixel coords

                `centre_y_pixel` :  [ int ]
                    y location of marker centre on the image in pixel coords

                Returns:
                ---
                NONE

                Example call:
                ---
                pixel_dist(30,40)
        '''

        focal_length = (self.img_width / 2) / math.tan(self.hfov_rad / 2)
        # print("focal length : ", focal_length)

        # self.Zm = self.drone_position[2] - self.building_alt - 0.2

        drone_pos = list(self.drone_position)

        # roll pitch and yaw equalization
        m_roll = [[1, 0, 0],
                  [0, math.cos(self.drone_orientation_euler[0]), -math.sin(self.drone_orientation_euler[0])],
                  [0, math.sin(self.drone_orientation_euler[0]), math.cos(self.drone_orientation_euler[0])]]

        m_pitch = [[math.cos(self.drone_orientation_euler[1]), 0, math.sin(self.drone_orientation_euler[1])],
                   [0, 1, 0],
                   [-math.sin(self.drone_orientation_euler[1]), 0, math.cos(self.drone_orientation_euler[1])]]

        m_yaw = [[math.cos(-self.drone_orientation_euler[2]), -math.sin(-self.drone_orientation_euler[2]), 0],
                 [math.sin(-self.drone_orientation_euler[2]), math.cos(-self.drone_orientation_euler[2]), 0],
                 [0, 0, 1]]

        rot_mat = np.dot(m_yaw, m_pitch)
        rot_mat = np.dot(rot_mat, m_roll)

        zmr = self.Zm/math.cos(self.drone_orientation_euler[0])
        zmp = self.Zm/math.cos(self.drone_orientation_euler[1])

        zm = max(zmr, zmp)

        ip_mat = [centre_x_pixel*zm - self.P[3], centre_y_pixel*zm - self.P[7], zm]
        self.K_inv = np.linalg.inv(self.K)
        op_mat = np.dot(self.K_inv, ip_mat)

        op_mat = np.dot(rot_mat, op_mat)
        # print("initial op_mat : ", op_mat)
        op_mat[2] -= 0.2
        # o = -self.updated_yaw*math.pi/180
        o = 0
        x_err = op_mat[0]*math.cos(o) - op_mat[1]*math.sin(o)
        y_err = op_mat[0]*math.sin(o) + op_mat[1]*math.cos(o)

        # print("op_mat : ", op_mat)
        # print("xerror : ", x_err)
        # print("y_err : ", y_err)
        # print(" o : ", o)
        # print(" updated _ yaw :", self.updated_yaw)

        if self.second_r == 0:
            self.found_x = x_err + drone_pos[0]*self.meter_conv[0]
            self.found_y = y_err + drone_pos[1]*self.meter_conv[1]
            self.found1.data = str(self.found_x) + " " + str(self.found_y)
        elif self.second_r == 1:
            self.found_x = x_err + drone_pos[0]*self.meter_conv[0]
            self.found_y = y_err + drone_pos[1]*self.meter_conv[1]
            self.found1.data = str(self.found_x) + " " + str(self.found_y)+" LOL"

        elif self.second_r == -1:
            self.found_x = x_err + drone_pos[0]*self.meter_conv[0]
            self.found_y = y_err + drone_pos[1]*self.meter_conv[1]
            self.found1.data = str(self.found_x) + " " + str(self.found_y)+" upward_marker_check_done"



        # if self.second_r == 0:
        #     self.found_x = ((centre_x_pixel - 200) * self.Zm / focal_length) + drone_pos[0]*self.meter_conv[0]
        #     self.found_y = ((centre_y_pixel -15 - 200) * self.Zm / focal_length) + drone_pos[1]*self.meter_conv[1]
        #     self.found1.data = str(self.found_x) + " " + str(self.found_y)
        # elif self.second_r == 1:
        #     self.found_x = ((centre_x_pixel  - 200) * self.Zm / focal_length) + drone_pos[0]*self.meter_conv[0]
        #     self.found_y = ((centre_y_pixel -15 - 200) * self.Zm / focal_length) + drone_pos[1]*self.meter_conv[1]
        #     self.found1.data = str(self.found_x) + " " + str(self.found_y)+" LOL"


    def cascade(self, img):
        '''
                Purpose:
                ---
                main function which is used for finding the presence of marker in the image and taking decisions if not.
                it obtains the bounding box of marker using cascade classifier.
                it also publishes the found coordinates of marker to position control(find_marker function)

                Input Arguments:
                ---
               `img` :  [Numpy]
                    image obtained from the camera feed of the drone


                Returns:
                ---
                NONE

                Example call:
                ---
                cascade(img1)
        '''

        logo_cascade = cv2.CascadeClassifier(self.cascade_xml_loc)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        self.logo = logo_cascade.detectMultiScale(gray, scaleFactor=1.05,minNeighbors=4)

        if len(self.logo) == 0 and self.do == 1:
            # self.one_t = 0
            self.found1.data = "upward_marker_check"


        elif len(self.logo) != 0 and self.do != 0:
            logo_curr = []
            red_index = 0
            red_check_flag = False

            for i in range(len(self.logo)):
                #[x, y, w, h] = self.logo[i]
                val = self.coord_dec(self.logo[i], "code_red")
                if val == 1:
                    red_index = i
                    red_check_flag = True

            logo_curr = self.logo[red_index]
            if red_check_flag:
                [x, y, w, h] = logo_curr
                cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv2.rectangle(self.img_1, (x, y), (x + w, y + h), (255, 255, 0), 2)

            if self.do == 1 and red_check_flag:
                #upward marker check done
                self.second_r = -1
                [x, y, w, h] = list(logo_curr)
                self.coord_dec(logo_curr, "nil")

            elif self.do == 2:
                self.second_r = 0
                [x, y, w, h] = list(logo_curr)
                self.coord_dec(logo_curr,"nil")


            elif self.do == 3:
                self.second_r = 1
                # if self.one_t == 0:
                #     self.one_t = 1
                self.coord_dec(logo_curr,"nil")
            elif self.do == 4:
                self.found1.data = "land_on_marker"

        if self.marker_detection == 1:
            self.found1_pub.publish(self.found1)  # publishing found coordinates to pos
        # print("found :",self.found1.data)
        self.found1.data = ""



    # second detection (finding center of X mark)
    def coord_dec(self,logo,red):
        '''
                Purpose:
                ---
                this contains x finding algo which checks whether the bounding box is actually a marker( because of false positive issue) and
                also finds the location of the x centre on the bounding box of marker obtained from cascade
                if it is a marker.

                Input Arguments:
                ---
                `logo` :  [list]
                    contains x,y(pixel coords in image),height and width of the bounding box(marker)

                `red` :  [String ]
                    "code_red" - to check whether it is actually a marker or not( because of false positive issue)
                    "nil" - do only x detection

                Returns:
                ---
                1 or 0 (only for "code_red") 0 - its not a marker,1 - marker
                Example call:
                ---
                coord_dec([x,y,h,w],"code_red")
        '''

        [x, y, w, h] = list(logo)
        ROI_mark = self.img[y:y + h, x:x + w]

        hsv = cv2.cvtColor(ROI_mark, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        ret, thresh = cv2.threshold(mask, 127, 255, 0)
        image,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            area = [cv2.contourArea(i) for i in contours]
            max_id = area.index(max(area))
            # print(max_id, area[max_id])

            #cv2.drawContours(img_result, [contours[max_id]], 0, (255, 0, 0), 3)
            (x1, y1), radius = cv2.minEnclosingCircle(contours[max_id])

            if red == "code_red":
                color = hsv[int(y1),int(x1)]
                lower_red = [0, 120, 70]
                upper_red = [10, 255, 255]
                for i in range(-1,2):
                    for j in range(-1,2):
                        if (hsv[int(y1)+j,int(x1)+i][0] >= lower_red[0] and hsv[int(y1)+j,int(x1)+i][0] <= upper_red[0]) and (hsv[int(y1)+j,int(x1)+i][1] >= lower_red[1] and hsv[int(y1)+j,int(x1)+i][1] <= upper_red[1]) and (hsv[int(y1)+j,int(x1)+i][2] >= lower_red[2] and hsv[int(y1)+j,int(x1)+i][2] <= upper_red[2]):
                            return 1
                        else:
                            return 0

            #center = (int(x1), int(y1))
            center = (int(x1)+x, int(y1)+y)
            radius = int(radius)
            cv2.circle(mask, (int(x1), int(y1)), radius, (0, 255, 0), 2)
            cv2.circle(mask, (int(x1), int(y1)), 1, (0, 255, 0), 2)

            cv2.circle(self.img_1, (int(x1) + x, int(y1) + y), radius, (255, 0, 0), 2)
            cv2.circle(self.img_1, (int(x1) + x, int(y1) + y), 1, (255, 0, 0), 2)

            centre_x_pixel, centre_y_pixel = center
            self.pixel_dist(centre_x_pixel, centre_y_pixel)
            # self.img_1 = mask


    def image_callback(self, data):
        '''
                Purpose:
                ---
                for getting image from drone camera

                Input Arguments:
                ---
                `data` :  [  Image]
                    image from cAMera



                Returns:
                ---
                NONE

                Example call:
                ---
                called by subscriber of "/edrone/camera/image_raw" topic
        '''
        try:
            self.img = self.bridge.imgmsg_to_cv2(data, "bgr8")  # Converting the image to OpenCV standard image
            self.img_1 = self.bridge.imgmsg_to_cv2(data, "bgr8")  # Converting the image to OpenCV standard image
            cv2.line(self.img_1,(0,215),(400,215),(0,255,0),2,cv2.LINE_AA)
            cv2.line(self.img_1, (200, 0), (200, 400), (0, 255, 0), 2, cv2.LINE_AA)
            self.cascade(self.img)


        except CvBridgeError as e:
            # print(e)
            return


if __name__ == '__main__':
    image_proc_obj = image_proc()
    r = rospy.Rate(5)
    while (True):

        cv2.imshow('vid_Cap', image_proc_obj.img_1)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

        image_proc_obj.publish_val()
        #r.sleep()

    cv2.destroyAllWindows()
