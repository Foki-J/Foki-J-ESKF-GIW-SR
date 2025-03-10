#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Quaternion
import signal
import sys

# 分别存储各个话题的数据
data_dict = {
    '/odom': {},
    '/odom_imu': {},
    '/odom_GPS': {},
    '/odom_GPS_secondary': {},
    '/odom_old': {},  # 新增
    '/odom_lidar': {}  # 新增
}

def odom_callback(msg, topic_name):
    global data_dict
    
    # 获取时间戳
    timestamp = msg.header.stamp.to_sec()
    
    # 获取位置信息
    position = msg.pose.pose.position
    position_dict = {
        'x': position.x,
        'y': position.y,
        'z': position.z
    }
    
    # 获取姿态信息（四元数）
    orientation = msg.pose.pose.orientation
    orientation_dict = {
        'x': orientation.x,
        'y': orientation.y,
        'z': orientation.z,
        'w': orientation.w
    }
    
    # 将位置和姿态信息存储到相应的字典中
    data_dict[topic_name][timestamp] = {
        'position': position_dict,
        'orientation': orientation_dict
    }
    
    rospy.loginfo(f"Recorded {topic_name} at {timestamp}")

def save_data():
    global data_dict
    
    for topic_name in data_dict:
        if len(data_dict[topic_name]) > 100:
            filename = f"{topic_name.replace('/', '_')[1:]}.npy"
            np.save(filename, data_dict[topic_name])
            rospy.loginfo(f"Saved {topic_name} data to {filename}")

def signal_handler(sig, frame):
    rospy.loginfo("Interrupted by user. Saving data...")
    save_data()
    rospy.signal_shutdown("Interrupted by user")
    sys.exit(0)

def main():
    rospy.init_node('odom_recorder', anonymous=True)
    
    # 设置信号处理函数
    signal.signal(signal.SIGINT, signal_handler)
    
    # 订阅所有话题，包括新增的两个
    rospy.Subscriber('/odom', Odometry, lambda msg: odom_callback(msg, '/odom'))
    rospy.Subscriber('/odom_imu', Odometry, lambda msg: odom_callback(msg, '/odom_imu'))
    rospy.Subscriber('/odom_GPS', Odometry, lambda msg: odom_callback(msg, '/odom_GPS'))
    rospy.Subscriber('/odom_GPS_secondary', Odometry, lambda msg: odom_callback(msg, '/odom_GPS_secondary'))
    rospy.Subscriber('/odom_old', Odometry, lambda msg: odom_callback(msg, '/odom_old'))  # 新增
    rospy.Subscriber('/odom_lidar', Odometry, lambda msg: odom_callback(msg, '/odom_lidar'))  # 新增
    
    rospy.loginfo("Started recording from all specified topics. Press Ctrl+C to stop and save data.")
    
    # 持续运行直到用户中断
    rospy.spin()

if __name__ == '__main__':
    main()