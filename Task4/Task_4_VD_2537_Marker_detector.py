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
import csv
import copy


class image_proc():

    # Initialise everything
    def __init__(self):

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
        self.meter_conv = [110693.227, 105078.267, 1]  # Factor for degrees to meters conversion(lat,long,alt to x,y,z)
        self.do = 0    # denotes each phase of detection
        self.second_r = 0
        self.one_t = 0
        self.allow = 1
        self.building_alt = 0

        self.package_drop_coordinates = []
        self.total_package_number = 0
        # for obtaining and storing the building coordinates from csv file
        self.file_read_func()
        self.meter_conv = [110693.227, 105078.267, 1]  # Factor for degrees to meters conversion(lat,long,alt to x,y,z)
        self.logo = []

        self.marker_detection = 1

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


    def marker_detect_start(self, data):
        if data.data == 1:
            self.marker_detection = 1
        else:
            self.marker_detection = 0

    def build_alt_update(self):
        o = 1
        for i in range(self.total_package_number):
            o+=1
            if abs(self.drone_position[0]*self.meter_conv[0] - self.package_drop_coordinates[i][0]*self.meter_conv[0]) < 8 and abs(self.drone_position[1]*self.meter_conv[1] - self.package_drop_coordinates[i][1]*self.meter_conv[1]) < 8:
                self.building_alt = self.package_drop_coordinates[i][2]

    # Extract data from csv file
    def file_read_func(self):

        with open(self.manifest_csv_loc) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                self.package_drop_coordinates.append([float(row[1]), float(row[2]), float(row[3])])
                self.total_package_number += 1


    def publish_val(self):

        if self.allow == 1:
            self.err_x_m_pub.publish(self.err_x_m)
            self.err_y_m_pub.publish(self.err_y_m)
        self.curr_marker_id_pub.publish(self.curr_marker_id)

    def update_allow(self, data):
        self.allow = data.data

    def x_error(self, data):
        self.err_x_m = data.data

    def y_error(self, data):
        self.err_y_m = data.data

    def marker_id(self, data):
        self.curr_marker_id = data.data

    def find_marker_reply_fun(self,reply):
        self.do = reply.data


    def range_bottom(self,bottom):
        #pass
        self.Zm_bott = bottom.ranges[0]


    #callback function for drone position
    def edrone_position(self, gps):
        self.drone_position[0] = gps.latitude
        self.drone_position[1] = gps.longitude
        self.drone_position[2] = gps.altitude


    def pixel_dist(self,centre_x_pixel,centre_y_pixel):
        focal_length = (self.img_width / 2) / math.tan(self.hfov_rad / 2)

        self.Zm = self.drone_position[2] - self.building_alt - 0.2

        if self.second_r == 0:
            self.found_x = (centre_x_pixel +3 - 200) * self.Zm / focal_length
            self.found_y = (centre_y_pixel -3 - 200) * self.Zm / focal_length
            self.found1.data = str(self.found_x) + " " + str(self.found_y)
        elif self.second_r == 1:
            self.found_x = (centre_x_pixel +3 - 200) * self.Zm / focal_length
            self.found_y = (centre_y_pixel -3 - 200) * self.Zm / focal_length
            self.found1.data = str(self.found_x) + " " + str(self.found_y)+" LOL"


    def cascade(self, img):

        logo_cascade = cv2.CascadeClassifier(self.cascade_xml_loc)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        self.logo = logo_cascade.detectMultiScale(gray, scaleFactor=1.05,minNeighbors=1)

        if len(self.logo) == 0 and self.do == 1:
            # self.one_t = 0
            self.found1.data = "upward_marker_check"


        elif len(self.logo) != 0:
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
            if  red_check_flag:
                [x, y, w, h] = logo_curr
                cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv2.rectangle(self.img_1, (x, y), (x + w, y + h), (255, 255, 0), 2)

            if self.do == 1 and red_check_flag:
                self.found1.data = "upward_marker_check_done"
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
    r = rospy.Rate(5)
    while (True):

        cv2.imshow('vid_Cap', image_proc_obj.img_1)
        image_proc_obj.build_alt_update()

        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

        image_proc_obj.publish_val()
        #r.sleep()

    cv2.destroyAllWindows()


