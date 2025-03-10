#!/usr/bin/env python3
import rospy
import threading
import sensor_msgs.msg
from communication_msgs.msg import motors
import time
import random
from geometry_msgs.msg import TransformStamped
from tf.broadcaster import TransformBroadcaster
import nav_msgs.msg
import numpy as np
from numpy import sin,cos
class PublisherThread(threading.Thread):
    def __init__(self, topic_name, message_type, frequency, data_func):
        super(PublisherThread, self).__init__()
        self.topic_name = topic_name
        self.message_type = message_type
        self.frequency = frequency
        self.data_func = data_func
        self.pub = None
        self.stop_event = threading.Event()

    def run(self):
        self.pub = rospy.Publisher(
            self.topic_name, self.message_type, queue_size=10)
        rate = rospy.Rate(self.frequency)

        while not self.stop_event.is_set() and not rospy.is_shutdown():
            msg = self.data_func()
            self.pub.publish(msg)
            rate.sleep()

    def stop(self):
        self.stop_event.set()

# 定义生成IMU消息和电机状态消息的函数


def generate_imu_msg():
    imu_msg = sensor_msgs.msg.Imu()
    # 初始化imu_msg字段（例如：时间戳、加速度、角速度等）
    # ...
    imu_msg.header.stamp = rospy.Time.now()
    return imu_msg


def generate_motor_msg():
    motor_msg = motors()
    # 初始化motor_msg字段（例如：时间戳、速度等）
    motor_msg.header.stamp = rospy.Time.now()
    motor_msg.first.velocity = random.random()*1000
    return motor_msg
def rot_matrix(theta):
    return np.array([[cos(theta),-sin(theta),0],[sin(theta),cos(theta),0],[0,0,1]])
def generate_odom_msg():
    odom_msg = nav_msgs.msg.Odometry()
    odom_msg.header.frame_id = 'odom'
    odom_msg.child_frame_id = 'base_link'

    odom_msg.pose.pose.position.x = 0
    odom_msg.pose.pose.position.y = 0
    odom_msg.pose.pose.orientation.w = 1.0  # 简化起见，这里假设四元数始终指向正前方

    # 将对角线上的协方差填入对应的6x6矩阵中，非对角线元素设为0
    theta = np.pi/2
    R = np.array([[cos(theta), -sin(theta), 0],[sin(theta),cos(theta),0],[0,0,1]])
    pose_covariance = np.zeros([6,6])
    pose_covariance[:3,:3] = R@[[4,1,0],[1,9,0],[0,0,1]]@R.T
    print(pose_covariance)
    # 将numpy数组转换为列表并赋值到消息字段中
    odom_msg.pose.covariance = pose_covariance.flatten().tolist()
    odom_msg.header.stamp = rospy.Time.now()
    return odom_msg

if __name__ == '__main__':
    rospy.init_node('mock_data_publisher')

    imu_thread = PublisherThread(
        '/imu', sensor_msgs.msg.Imu, 3, generate_imu_msg)  # 假设IMU频率为50Hz
    motor_thread = PublisherThread(
        '/motors_state', motors, 50, generate_motor_msg)  # 假设电机状态频率为20Hz

    odom_thread = PublisherThread('/odom', nav_msgs.msg.Odometry, 10, generate_odom_msg)  # 假设Odometry频率为10Hz
    # imu_thread.start()
    #motor_thread.start()
    odom_thread.start()
    rospy.spin()  # 保持节点运行直到关闭

    # 在关闭节点前停止线程
    imu_thread.stop()
    motor_thread.stop()
    odom_thread.stop()
    motor_thread.join()
    odom_thread.join()
    
