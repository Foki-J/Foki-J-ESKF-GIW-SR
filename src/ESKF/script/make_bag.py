import rosbag
from datetime import datetime

def combine_bags(input_bag_paths, output_bag_path):
    """
    将多个 .bag 文件合并成一个 .bag 文件。
    
    :param input_bag_paths: 输入 .bag 文件的路径列表
    :param output_bag_path: 输出 .bag 文件的路径
    """
    with rosbag.Bag(output_bag_path, 'w') as outbag:
        for input_bag_path in input_bag_paths:
            print(f"Processing {input_bag_path}...")
            for topic, msg, t in rosbag.Bag(input_bag_path).read_messages():
                outbag.write(topic, msg, t)
    print(f"已生成组合后的 bag 文件: {output_bag_path}")

def filter_bag_by_intervals(input_bag_path, output_bag_path, skip_topics, skip_intervals):
    """
    读取输入的 .bag 文件，并在指定的时间段内跳过某些话题。
    
    :param input_bag_path: 输入 .bag 文件的路径
    :param output_bag_path: 输出 .bag 文件的路径
    :param skip_topics: 需要跳过的话题列表
    :param skip_intervals: 需要跳过的时间段列表，格式为 [(start_time1, end_time1), (start_time2, end_time2), ...]
                           时间戳为浮点数，单位为秒
    """
    with rosbag.Bag(output_bag_path, 'w') as outbag:
        for topic, msg, t in rosbag.Bag(input_bag_path).read_messages():
            timestamp = t.to_sec()
            
            # 检查当前时间是否在需要跳过的时间段内
            skip_message = False
            for start, end in skip_intervals:
                if start <= timestamp <= end:
                    skip_message = True
                    break
            
            # 如果当前话题在跳过的话题列表中，并且在跳过的时间段内，则跳过该消息
            if topic in skip_topics and skip_message:
                continue
            
            # 写入其他消息到输出 bag 文件
            outbag.write(topic, msg, t)
    print(f"已生成过滤后的 bag 文件: {output_bag_path}")

if __name__ == '__main__':
    # 输入 bag 文件路径列表
    # input_bag_paths = ['/home/luoteng/bag/20250102_222328_lidar_topic_0.bag', '/home/luoteng/bag/20250102_222328_lidar_topic_1.bag', '/home/luoteng/bag/20250102_222328_lidar_topic_2.bag']
    input_bag_paths = ['/home/luoteng/bag/zfq_2025-02-24-20-09-33.bag']
    # 中间组合 bag 文件路径
    combined_bag_path = '/home/luoteng/bag/zfq_2025-02-24-20-09-33.bag'
    
    # 最终输出 bag 文件路径
    filtered_bag_path = '/home/luoteng/bag/zfq_2025-02-24-20-09-33_filter.bag'
    
    # 需要跳过的话题列表
    skip_topics = ['/fix']
    
    # 需要跳过的时间段列表，格式为 (开始时间, 结束时间)，单位为秒
    skip_intervals = [
        (1740399077, 1740399116),  # 在第 10 到 20 秒之间跳过
    ]
    
    # 第一步：将多个 bag 文件合并成一个
    # combine_bags(input_bag_paths, combined_bag_path)
    
    # 第二步：对组合后的 bag 文件进行过滤
    filter_bag_by_intervals(combined_bag_path, filtered_bag_path, skip_topics, skip_intervals)
    