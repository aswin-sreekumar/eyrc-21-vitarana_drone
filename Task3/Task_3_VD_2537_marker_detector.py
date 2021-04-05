#!/usr/bin/env python


'''
This is a boiler plate script that contains an example on how to subscribe a rostopic containing camera frames
and store it into an OpenCV image to use it further for image processing tasks.
Use this code snippet in your code or you can also continue adding your code in the same file
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

import rospy
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


class image_proc():

    # Initialise everything
    def __init__(self):
        rospy.init_node('barcode_test')  # Initialise rosnode
        self.image_sub = rospy.Subscriber("/edrone/camera/image_raw", Image,
                                          self.image_callback)  # Subscribing to the camera topic
        self.img = np.empty([])  # This will contain your image frame from camera
        self.bridge = CvBridge()
        self.img_1 = np.ones((8, 8, 3), np.uint8)
        self.ROI = np.ones((8, 8, 3), np.uint8)

        self.hfov_rad = 1.3962634
        self.img_width = 400
        self.Zm = 0.31
        self.drone_position = [19, 72, 0.31]  # Current drone position in GPS coordinates
        self.meter_conv = [110693.227, 105078.267, 1]  # Factor for degrees to meters conversion(lat,long,alt to x,y,z)

        self.do = 0
        self.done1 = 0
        self.second_r = 0
        self.one_t = 0

        self.allow = 1


        rospy.Subscriber('/edrone/gps', NavSatFix, self.edrone_position)
        rospy.Subscriber('/edrone/range_finder_bottom', LaserScan, self.range_bottom)
        rospy.Subscriber('/find_marker_reply', Float32, self.find_marker_reply_fun)

        rospy.Subscriber('/err_x_m_pos', Float32, self.x_error)
        rospy.Subscriber('/err_y_m_pos', Float32, self.y_error)
        rospy.Subscriber('/curr_marker_id_pos', Float32, self.marker_id)
        rospy.Subscriber('/allow', Float32, self.update_allow)

        # Publishers
        self.found1_pub = rospy.Publisher('/find_marker',String,queue_size = 1)
        self.found1 = String()
        self.found1.data = ""

        # self.err_x_m_pub = rospy.Publisher('/edrone/err_x_m', Float32, queue_size=1)
        # self.err_y_m_pub = rospy.Publisher('/edrone/err_y_m', Float32, queue_size=1)
        # self.curr_marker_id_pub = rospy.Publisher('/edrone/curr_marker_id', Float32, queue_size=1)
        self.marker_data_pub = rospy.Publisher('/edrone/marker_data', MarkerData, queue_size=1)

        # self.err_x_m = Float32()
        # self.err_y_m = Float32()
        # self.curr_marker_id = Float32()
        # self.NaN = Float32()
        self.marker_data_val = MarkerData()

        # self.err_x_m.data = 0
        # self.err_y_m.data = 0
        # self.curr_marker_id.data = 2
        # self.NaN.data = float('NaN')

        self.marker_data_val.marker_id = int(0)
        self.marker_data_val.err_x_m = float('NaN')
        self.marker_data_val.err_y_m = float('NaN')



    def publish_val(self):

        if self.allow != 1:
            self.marker_data_val.err_x_m = float('NaN')
            self.marker_data_val.err_y_m = float('NaN')

        self.marker_data_pub.publish(self.marker_data_val)

    def update_allow(self, data):
        self.allow = data.data

    def x_error(self, data):
        self.marker_data_val.err_x_m = data.data

    def y_error(self, data):
        self.marker_data_val.err_y_m = data.data

    def marker_id(self, data):
        self.marker_data_val.marker_id = int(data.data)

    def find_marker_reply_fun(self,reply):
        self.do = reply.data


    def range_bottom(self,bottom):


        self.Zm = bottom.ranges[0]
        # print(self.Zm)

    # callback function for drone position
    def edrone_position(self, gps):
        self.drone_position[0] = gps.latitude
        self.drone_position[1] = gps.longitude
        self.drone_position[2] = gps.altitude

    def pixel_dist(self,centre_x_pixel,centre_y_pixel):
        focal_length = (self.img_width / 2) / math.tan(self.hfov_rad / 2)
        # self.Zm = self.drone_position[2] - 22.2
        if self.second_r == 0:
            self.found_x = (centre_x_pixel - 200) * self.Zm / focal_length
            self.found_y = (centre_y_pixel-15 - 200) * self.Zm / focal_length
            self.found1.data = str(self.found_x) + " " + str(self.found_y)
        elif self.second_r == 1:
            self.found_x = (centre_x_pixel - 200) * self.Zm / focal_length
            self.found_y = (centre_y_pixel-15 - 200) * self.Zm / focal_length
            self.found1.data = str(self.found_x) + " " + str(self.found_y)+" LOL"
        # print(self.found_x, self.found_y)
        # to start motion



    def cascade(self, img):

        logo_cascade = cv2.CascadeClassifier('/home/jk56/catkin_ws/src/vitarana_drone/scripts/cascade.xml')
        #img = cv2.imread('/home/greesh/PycharmProjects/chumma/intro_cascade_classifiers_training_and_usage/test_3.png')  # Source image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        logo = logo_cascade.detectMultiScale(gray, scaleFactor=1.05,minNeighbors=1)
        # print(len(logo))


        if len(logo) == 0 and self.do == 1:
            # self.one_t = 0
            self.found1.data = "urs1"
            # print("start")

        elif len(logo) != 0:

            (x, y, w, h) = logo[0]
            cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.rectangle(self.img_1, (x, y), (x + w, y + h), (255, 255, 0), 2)

            if self.do == 1:
                self.found1.data = "urs1_done"
            elif self.do == 2:
                self.second_r = 0
                [x, y, w, h] = list(logo[0])

                centre_x_pixel,centre_y_pixel = (x+(w/2),y+(h/2))
                self.pixel_dist(centre_x_pixel,centre_y_pixel)

                # focal_length = (self.img_width / 2) / math.tan(self.hfov_rad / 2)
                # # self.Zm = self.drone_position[2] - 22.2
                # self.found_x = (centre_x_pixel - 200) * self.Zm / focal_length
                # self.found_y = (centre_y_pixel - 200) * self.Zm / focal_length
                # print(self.found_x,self.found_y)
                # #to start motion
                #
                # self.found1.data = str(self.found_x) + " " + str(self.found_y)
                self.done1 = 1

            # elif self.do == 3:
            #      self.found1.data = "urs2_down"

            elif self.do == 3:
                self.second_r = 1
                # if self.one_t == 0:
                #     self.one_t = 1
                self.coord_dec(logo)
            elif self.do == 4:
                self.found1.data = "urs2_down_2.0"


        self.found1_pub.publish(self.found1)  # publishing found coordinates to pos
        # print("found :",self.found1.data)
        self.found1.data = ""




    def coord_dec(self,logo):
        [x, y, w, h] = list(logo[0])
        ROI_mark = self.img[y:y + h, x:x + w]
        # method_1

        # ROI_mark_gray = cv2.cvtColor(self.img[y:y+h, x:x+w],cv2.COLOR_BGR2GRAY)
        # adapt = cv2.adaptiveThreshold(ROI_mark_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        #
        # # if self.do == 4:
        # circles = cv2.HoughCircles(adapt, cv2.HOUGH_GRADIENT, 1, 20,param1=50, param2=30, minRadius=0, maxRadius=0)
        #
        # circles = np.uint16(np.around(circles))
        # print(circles)
        #
        # for i in circles[0, :]:
        #     # draw the outer circle
        #     cv2.circle(ROI_mark, (i[0]+x, i[1]+y), i[2], (0, 255, 0), 2)
        #     # draw the center of the circle
        #     cv2.circle(ROI_mark, (i[0]+x, i[1]+y), 2, (255, 0,0), 3)
        # centre_x_pixel, centre_y_pixel = i[0]+x, i[1]+y

        # method_2

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
            #center = (int(x1), int(y1))
            center = (int(x1)+x, int(y1)+y)
            radius = int(radius)
            cv2.circle(mask, (int(x1), int(y1)), radius, (0, 255, 0), 2)
            cv2.circle(mask, (int(x1), int(y1)), 1, (0, 255, 0), 2)
            centre_x_pixel, centre_y_pixel = center
            self.pixel_dist(centre_x_pixel, centre_y_pixel)
            self.img_1 = mask













    def image_callback(self, data):
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
    r = rospy.Rate(10)
    while (True):

        cv2.imshow('vid_Cap', image_proc_obj.img_1)
        #cv2.imshow('vid_Cap', image_proc_obj.ROI)

        # if cv2.waitKey(1) & 0xFF == ord('z'):
        #     image_proc_obj.do = 1
        # elif cv2.waitKey(1) & 0xFF == ord('x'):
        #     image_proc_obj.do = 2
        # elif cv2.waitKey(1) & 0xFF == ord('c'):
        #     image_proc_obj.do = 3
        # elif cv2.waitKey(1) & 0xFF == ord('v'):
        #     image_proc_obj.do = 4
        # elif cv2.waitKey(1) & 0xFF == ord('b'):
        #     image_proc_obj.do = 5
        if cv2.waitKey(1) & 0xFF == ord('s'):
            #cv2.imwrite("/home/greesh/PycharmProjects/chumma/detected.png", image_proc_obj.img)
            break

        image_proc_obj.publish_val()
        r.sleep()

    cv2.destroyAllWindows()
# rospy.spin()

