#! /usr/bin/python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest, CommandTOL
from tf.transformations import quaternion_from_euler
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import math
import time

class RockMapping:
    def __init__(self):
        rospy.loginfo("entered1")
        self.drone_ready_to_fly = True
        self.current_pose = PoseStamped()
        self.des_pose = PoseStamped()
        self.rate = rospy.Rate(10)
        self.t_start = rospy.get_time()

        self.rock_coordinate = np.array([60.2, -12.5, 21])
        self.probe_coordinate = np.array([40.5, 3.8, 15])
        self.rover_coordinate = np.array([12.6, -65.0, -3.5])
        self.t_start = rospy.get_time()

        self.local_pose_publisher = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)
        self.local_pose_subscriber = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, callback=self.Current_pose)
        self.image_sub = rospy.Subscriber('/uav_camera_down/image_raw', Image, callback=self.Image_data)
        self.drone_state_sub = rospy.Subscriber("/mavros/state", State, callback=self.Current_drone_state)

        self.bridge = CvBridge()
        self.probe_location_img = [None, None]

        self.probe_visible = False
        self.data_mulled = False
        self.rock_mapped = False
        self.mission_complete = False
        self.reached_probe = False
        self.reached_rock = False
        self.reached_rover = False

        self.des_pose = self.copy_pose(self.current_pose)
        self.rate.sleep()


        self.Fly_to_Probe()

        if self.reached_rover and self.mission_complete:
            rospy.wait_for_service('/mavros/cmd/land')
            landService = rospy.ServiceProxy('/mavros/cmd/land', CommandTOL)
            landService(altitude=0, latitude=self.des_pose.pose.position.x, longitude=self.des_pose.pose.position.y,
                        min_pitch=0, yaw=0)
            rospy.loginfo("Finished landing...Mission Complete")


    def Current_drone_state(self,msg):
        rospy.loginfo(msg.mode)
        if (msg.mode == "OFFBOARD"):
            self.drone_ready_to_fly = True
            rospy.loginfo(self.drone_ready_to_fly)

    def Current_pose(self, msg):
        self.current_pose = msg

    def Image_data(self, img):
        self.current_image = self.bridge.imgmsg_to_cv2(img, "passthrough")
        hsv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
        red_lower = np.array([110, 100, 50], np.uint8)
        red_upper = np.array([130, 255, 255], np.uint8)
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        kernal = np.ones((5, 5), "uint8")

        # For red color
        red_mask = cv2.dilate(red_mask, kernal)
        res_red = cv2.bitwise_and(self.current_image, self.current_image,
                                  mask=red_mask)


        # Creating contour to track red color
        _, contours, _ = cv2.findContours(red_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                self.current_image = cv2.rectangle(self.current_image, (x, y),
                                           (x + w, y + h),
                                           (0, 0, 255), 2)

                cv2.putText(self.current_image, "Probe", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255))
                x_center = x + (w/2)
                y_center = y + (h/2)
                self.probe_location_img = [x_center, y_center]
                self.probe_visible = True
                cv2.circle(self.current_image, (x_center, y_center), 3, (0, 0, 255))
            else:
                self.probe_visible = False
        cv2.imshow("Image data", self.current_image)
        cv2.waitKey(3)

    def Fly_to_Probe(self):
        self.des_pose.pose.position.x = self.probe_coordinate[0]
        self.des_pose.pose.position.y = self.probe_coordinate[1]
        self.des_pose.pose.position.z = self.probe_coordinate[2]
        while not rospy.is_shutdown():
            x = (float(self.probe_coordinate[0]) - self.current_pose.pose.position.x) ** 2
            y = (float(self.probe_coordinate[1]) - self.current_pose.pose.position.y) ** 2
            distance =math.sqrt(x + y)
            if distance > 0.1 and not self.reached_probe:
                self.local_pose_publisher.publish(self.des_pose)
                self.rate.sleep()
            if self.probe_visible and (not self.data_mulled) and distance <= 0.1 and (not self.rock_mapped):
                self.reached_probe = True
                self.visual_servoing()
                break
        self.Fly_to_Rock()

    def Fly_to_Rock(self):
        self.des_pose.pose.position.x = self.rock_coordinate[0]
        self.des_pose.pose.position.y = self.rock_coordinate[1]
        self.des_pose.pose.position.z = self.rock_coordinate[2]
        while not rospy.is_shutdown():
            x = (float(self.rock_coordinate[0]) - self.current_pose.pose.position.x) ** 2
            y = (float(self.rock_coordinate[1]) - self.current_pose.pose.position.y) ** 2
            distance = math.sqrt(x + y)
            if distance > 0.1 and not self.reached_rock:
                self.local_pose_publisher.publish(self.des_pose)
                self.rate.sleep()
            else:
                self.reached_rock = True
                rospy.loginfo("Reached Rock")
                break
        self.Orbit_mapping()

    def Orbit_mapping(self):
        rospy.loginfo("Start Orbitting the Rock")
        self.t_start = rospy.get_time()
        while not rospy.is_shutdown() and not self.rock_mapped:
            t = rospy.get_time() - self.t_start
            if t > 25:
                self.rock_mapped = False
                break
            self.des_pose.pose.position.x = self.rock_coordinate[0] + (7 * math.cos(0.5 * t))
            self.des_pose.pose.position.y = self.rock_coordinate[1] + (7 * math.sin(0.5 * t))
            self.des_pose.pose.position.z = self.rock_coordinate[2]
            heading = math.atan2(self.current_pose.pose.position.y - self.rock_coordinate[1],
                                 self.current_pose.pose.position.x - self.rock_coordinate[0])
            orientation = quaternion_from_euler(0, 0, 135 + heading)
            self.des_pose.pose.orientation.x = orientation[0]
            self.des_pose.pose.orientation.y = orientation[1]
            self.des_pose.pose.orientation.z = orientation[2]
            self.des_pose.pose.orientation.w = orientation[3]
            self.local_pose_publisher.publish(self.des_pose)
            self.rate.sleep()
        self.Fly_to_Rover()

    def Fly_to_Rover(self):
        self.des_pose.pose.position.x = self.rover_coordinate[0]
        self.des_pose.pose.position.y = self.rover_coordinate[1]
        self.des_pose.pose.position.z = self.rover_coordinate[2]
        while not rospy.is_shutdown():
            x = (float(self.rover_coordinate[0]) - self.current_pose.pose.position.x) ** 2
            y = (float(self.rover_coordinate[1]) - self.current_pose.pose.position.y) ** 2
            distance = math.sqrt(x + y)
            if distance > 0.1 and not self.reached_rover:
                self.local_pose_publisher.publish(self.des_pose)
                self.rate.sleep()
            else:
                self.reached_rover = True
                self.mission_complete = True
                rospy.loginfo("Reached Rover")
                break

    def visual_servoing(self):
        print("Found the probe, now staying centered around it with visual servoing while mulling data...")
        y_not_center = True
        x_not_center = True
        center_counter = 0
        self.t_start = rospy.get_time()
        while (center_counter < 5000) and (not rospy.is_shutdown()) and (self.probe_visible):
            t = rospy.get_time() - self.t_start
            if t > 2:
                self.probe_visible = False
                break
            x_offset = self.probe_location_img[0] - 320
            y_offset = self.probe_location_img[1] - 240
            # print(x_offset, y_offset)
            if abs(x_offset) > 40:
                x_not_center = True
                if x_offset < 0:
                    self.des_pose.pose.position.x -= 0.000001
                else:
                    self.des_pose.pose.position.x += 0.000001
            else:
                x_not_center = False

            if abs(y_offset) > 40:
                y_not_center = True
                if y_offset < 0:
                    self.des_pose.pose.position.y += 0.000001
                else:
                    self.des_pose.pose.position.y -= 0.000001
            else:
                y_not_center = False
            self.local_pose_publisher.publish(self.des_pose)
            if (not x_not_center) and (not y_not_center):
                center_counter += 1
                # print(center_counter)

        if self.probe_visible:
            print("Finished mulling, leaving probe to go Rock")
            self.data_mulled = True



    def copy_pose(self, pose):
        pt = pose.pose.position
        quat = pose.pose.orientation
        copied_pose = PoseStamped()
        copied_pose.header.frame_id = pose.header.frame_id
        copied_pose.pose.position = Point(pt.x, pt.y, pt.z)
        copied_pose.pose.orientation = Quaternion(quat.x, quat.y, quat.z, quat.w)
        return copied_pose


if __name__ == "__main__":
    rospy.init_node("Rock_Mapping_position_control", anonymous=True)
    RockMapping()
    rospy.spin()