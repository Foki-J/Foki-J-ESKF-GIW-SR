import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation
from matplotlib.image import imread
import pandas as pd
import seaborn as sns
def load_data(file_path):
    # 加载数据
    all_data = np.load(file_path, allow_pickle=True).item()
    return all_data
colors = {
        '/odom': 'b',
        '/odom_imu': 'orange',
        '/odom_old': 'r',
        '/odom_GPS_secondary': 'white'
    }
def plot_trajectories(data_dict, background_image_path=None, topics_order=None):
    fig, ax = plt.subplots()
    
    ratio = 896/861
    height = 120
    width = height*ratio
    origin = [-90, 85]
    x_lim = [origin[0], origin[0]+width]
    y_lim = [origin[1]-height, origin[1]]
    
    # fig.patch.set_facecolor('gray')
    # ax.set_facecolor('gray')
    if background_image_path:
        background_image = imread(background_image_path)
        ax.imshow(background_image, extent=x_lim+y_lim, aspect='auto')
    if topics_order is None:
        topics_order = list(data_dict.keys())  # 默认按数据中的顺序

    for topic in topics_order:
        if topic not in data_dict or topic == '/odom_GPS':
            continue  # 跳过不存在或特定的话题
        
        x_coords = []
        y_coords = []

        for timestamp, pose_data in sorted(data_dict[topic].items(), key=lambda item: item[0]):
            x_coords.append(pose_data['position']['x'])
            y_coords.append(pose_data['position']['y'])
        ax.plot(x_coords, y_coords,linewidth = 2,  color=colors.get(topic, 'k'), label=topic)
        # sizes = 1  # 散点的大小
        # if topic == '/odom_GPS_secondary':
        #     sizes = 10  # 可以为不同的轨迹设置不同的点大小
        #     ax.plot(x_coords, y_coords,linewidth = 2,  color=colors.get(topic, 'k'), label=topic)
        #     ax.scatter(x_coords, y_coords, s=sizes, color=colors.get(topic, 'k'), label=topic)
        # else:   
        #     ax.plot(x_coords, y_coords, s=sizes, color=colors.get(topic, 'k'), label=topic)  

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend().set_visible(False)
    plt.title('Trajectories of Different Topics')
    plt.show()
def highpass_filter(data, sampling_rate, cutoff_freq):
    nyquist_rate = sampling_rate / 2
    normal_cutoff = cutoff_freq / nyquist_rate
    b, a = butter(5, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data
def calculate_differential_distances(topic_data, max_time=1800):
    timestamps = sorted(topic_data.keys())
    distances = []  # 初始差分为空列表
    times = [(t - timestamps[0]) for t in timestamps]  # 时间从0开始，单位秒

    for i in range(1, len(timestamps)):
        if times[i] > max_time:
            break  # 如果超过max_time，则停止计算

        prev_pose = topic_data[timestamps[i-1]]['position']
        curr_pose = topic_data[timestamps[i]]['position']
        dx = curr_pose['x'] - prev_pose['x']
        dy = curr_pose['y'] - prev_pose['y']
        distance = np.sqrt(dx**2 + dy**2)
        distances.append(distance)
    return times[1:i], distances  # 返回times从第二个元素开始
def downsample(topic_data, factor):
    """
    对数据进行降采样。
    
    参数:
        topic_data (dict): 原始数据，键为时间戳，值为位置信息。
        factor (int): 降采样因子，每隔 `factor` 个样本取一个。
        
    返回:
        dict: 降采样后的数据。
    """
    sorted_keys = sorted(topic_data.keys())
    downsampled_data = {sorted_keys[i]: topic_data[sorted_keys[i]] for i in range(0, len(sorted_keys), factor)}
    return downsampled_data
def plot_differential_distances(data_dict, topics_order=None, max_time=1800):
    if topics_order is None:
        topics_order = list(data_dict.keys())  # 默认按数据中的顺序

    filtered_distances_list = []
    labels = []
    fig, ax = plt.subplots(figsize=(3, 4))
    for topic in topics_order:
        if topic not in data_dict or topic in ['/odom_GPS', '/odom_GPS_secondary']:
            continue  # 跳过不存在或特定的话题
        if topic=='/odom_imu':
            data_dict_topic = downsample(data_dict[topic], 2)
        else:
            data_dict_topic = data_dict[topic]
        times, distances = calculate_differential_distances(data_dict_topic, max_time=max_time)
        
        # 计算采样率（假设时间间隔是均匀的）
        sampling_rate = 1 / np.mean([times[i+1] - times[i] for i in range(len(times)-1)])
        
        # 高通滤波
        filtered_distances = highpass_filter(distances, sampling_rate, cutoff_freq=2)
        # filtered_distances = np.array(distances)
        filtered_distances[filtered_distances < 0] = 0  # 将负值设为0
        print("Average distance per second",topic, sum(filtered_distances)/len(times))
        print("Max distance per second",topic, np.max(filtered_distances), np.argmax(filtered_distances))
        print(len(filtered_distances))
        # 确保所有值都大于0以使用对数尺度
        filtered_distances += 1e-10

        # 计算CDF
        sorted_data = np.sort(filtered_distances)
        cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)

        # 绘制CDF，x轴使用对数尺度
        ax.semilogx(sorted_data, cdf, label=f'{topic} CDF', color=colors.get(topic, 'k'))
        ax.plot(times, distances, color=colors.get(topic, 'k'), label=topic) 
    ax.set_xlabel('Distances (Log Scale)')
    ax.set_ylabel('CDF')
    # ax.set_xlim([1e-6,1])
    # ax.set_ylim([0.0, 1.05])
    ax.set_title('Cumulative Distribution Function of Distances per Topic with Logarithmic X-Axis')
    # ax.legend()
    plt.show()
def calculate_error(data_dict):
    required_topics = ['/odom', '/odom_imu', '/odom_old']
    gps_data = data_dict['/odom_GPS']
    odom_gps_secondary_idx = 0
    topics_idx = {topic: 0 for topic in required_topics}
    odom_gps_secondary_timestamps = list(data_dict['/odom_GPS_secondary'].keys())
    topics_timestamps = {topic: list(data_dict[topic].keys()) for topic in required_topics}
    odom_gps_secondary_poses = list(data_dict['/odom_GPS_secondary'].values())
    topics_poses = {topic: list(data_dict[topic].values()) for topic in required_topics}
    position_errors = {topic: [] for topic in required_topics}
    orientation_errors = {topic: [] for topic in required_topics}
    for gps_timestamp, gps_pose in gps_data.items():
        #查找最近的/odom_GPS_secondary
        for i in range(odom_gps_secondary_idx, len(odom_gps_secondary_timestamps)-1):
            odom_gps_secondary_idx = i
            if odom_gps_secondary_timestamps[i] < gps_timestamp and odom_gps_secondary_timestamps[i+1] > gps_timestamp:
                prev_diff = abs(odom_gps_secondary_timestamps[i] - gps_timestamp)
                curr_diff = abs(odom_gps_secondary_timestamps[i+1] - gps_timestamp)
                odom_gps_secondary_pose = (odom_gps_secondary_poses[i]) if prev_diff <= curr_diff else (odom_gps_secondary_poses[i])
                odom_gps_secondary_timestamp = (odom_gps_secondary_timestamps[i]) if prev_diff <= curr_diff else (odom_gps_secondary_timestamps[i])
                
                break  
        gps_position = np.array([gps_pose['position']['x'],gps_pose['position']['y'],gps_pose['position']['z']])#右天线的位置
        gps_secondary_position = np.array([odom_gps_secondary_pose['position']['x'],odom_gps_secondary_pose['position']['y'],odom_gps_secondary_pose['position']['z']])#左天线的位置
        real_position = (gps_position+gps_secondary_position)/2#计算真实位置
        real_orientation = gps_secondary_position-gps_position
        real_orientation = real_orientation/np.linalg.norm(real_orientation)
        for topic in required_topics:
            #找出最近的时间戳，并且插值计算位置
            for i in range(topics_idx[topic], len(topics_timestamps[topic])-1):
                topics_idx[topic] = i
                if topics_timestamps[topic][i] < gps_timestamp and topics_timestamps[topic][i+1] > gps_timestamp:
                    prev_diff = abs(topics_timestamps[topic][i] - gps_timestamp)
                    curr_diff = abs(topics_timestamps[topic][i+1] - gps_timestamp)
                    last_position = np.array([topics_poses[topic][i]['position']['x'],topics_poses[topic][i]['position']['y'],topics_poses[topic][i]['position']['z']])
                    next_position = np.array([topics_poses[topic][i+1]['position']['x'],topics_poses[topic][i+1]['position']['y'],topics_poses[topic][i+1]['position']['z']])
                    temp_position = last_position + (next_position-last_position)*(gps_timestamp-topics_timestamps[topic][i])/(topics_timestamps[topic][i+1]-topics_timestamps[topic][i])
                    last_orientation = np.array([topics_poses[topic][i]['orientation']['x'],topics_poses[topic][i]['orientation']['y'],topics_poses[topic][i]['orientation']['z'],topics_poses[topic][i]['orientation']['w']])
                    next_orientation = np.array([topics_poses[topic][i+1]['orientation']['x'],topics_poses[topic][i+1]['orientation']['y'],topics_poses[topic][i+1]['orientation']['z'],topics_poses[topic][i+1]['orientation']['w']])
                    temp_orientation = (last_orientation) if prev_diff <= curr_diff else (last_orientation)
                    position_errors[topic].append(np.linalg.norm(real_position-temp_position))
                    R = Rotation.from_quat(temp_orientation).as_matrix()
                    lever = np.array([0,1,0]).T
                    orientation_errors[topic].append(np.arccos(((real_orientation)@(R@lever)).sum()))
                    break
    # 绘制误差曲线
    for topic in required_topics:
        print(topic, max(position_errors[topic]))
        # print(topic, topics_timestamps[topic][0])
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    for topic in required_topics:
        position_errors[topic] = position_errors[topic][:200]
        orientation_errors[topic] = orientation_errors[topic][:200]
        ax[0].plot(range(len(position_errors[topic])), position_errors[topic],color=colors.get(topic, 'k'), linewidth=1)
        ax[1].plot(range(len(orientation_errors[topic])), orientation_errors[topic], color=colors.get(topic, 'k'), linewidth=1)

    ax[0].set_xlabel('Time (seconds)')
    ax[0].set_ylabel('Position Error (meters)')
    ax[0].legend()
    ax[0].set_title('Position Error Over Time')

    ax[1].set_xlabel('Time (seconds)')
    ax[1].set_ylabel('Orientation Error (radians)')
    ax[1].legend()
    ax[1].set_title('Orientation Error Over Time')

    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    # 定义各个话题对应的文件路径
    file_paths = {
        '/odom': 'npy3/odom.npy',
        '/odom_imu': 'npy3/odom_imu.npy',
        '/odom_GPS': 'npy3/odom_GPS.npy',
        '/odom_GPS_secondary': 'npy3/odom_GPS_secondary.npy',
        '/odom_old': 'npy3/odom_old.npy'
    }

    # 加载各个话题的数据
    data_dict = {}
    for topic, file_path in file_paths.items():
        try:
            data_dict[topic] = load_data(file_path)
            print(f"Loaded {file_path} successfully.")
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")

    # 定义你想绘制的话题顺序
    custom_topics_order = ['/odom_old','/odom_imu', '/odom']

    # 使用自定义顺序绘制轨迹
    plot_trajectories(data_dict, background_image_path = "huahong.png", topics_order=custom_topics_order)
    # plot_trajectories(data_dict, topics_order=custom_topics_order)
    # # 使用自定义顺序绘制差分距离图
    plot_differential_distances(data_dict, topics_order=custom_topics_order, max_time=160)
    calculate_error(data_dict)