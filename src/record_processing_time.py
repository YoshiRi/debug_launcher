#!/usr/bin/env python3
# # usage
# ros2 run debug_launcher record_processing_time.py --topics /topic1 /topic2
import csv
import os
import signal
import rclpy
from rclpy.node import Node
from tier4_debug_msgs.msg import Float64Stamped
import matplotlib.pyplot as plt
import argparse
import yaml

class Float64StampedLogger:
    def __init__(self, node: Node, topic_name: str):
        self.node = node
        self.topic_name = topic_name
        self.times = []
        self.values = []

        self.subscription = node.create_subscription(
            Float64Stamped,
            topic_name,
            self.listener_callback,
            10
        )

    def listener_callback(self, msg: Float64Stamped):
        timestamp = msg.stamp.sec + msg.stamp.nanosec * 1e-9
        value = msg.data
        self.times.append(timestamp)
        self.values.append(value)
        self.node.get_logger().info(f'Received on {self.topic_name}: time={timestamp}, value={value}')

    def save_to_file(self, csv_writer):
        for time, value in zip(self.times, self.values):
            csv_writer.writerow([self.topic_name, time, value])

    def shutdown(self, save_individual=True):
        if save_individual:
            csv_filename = f'{self.topic_name.replace("/", "_")}_output.csv'
            with open(csv_filename, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['topic', 'time', 'value'])
                self.save_to_file(csv_writer)
            # self.plot_data()

    def plot_data(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.times, self.values, marker='o')
        plt.title(f'{self.topic_name} Data Over Time')
        plt.xlabel('Time [s]')
        plt.ylabel('Value')
        plt.grid(True)
        plt.savefig(f'{self.topic_name.replace("/", "_")}_output_plot.png')  # Save plot as PNG
        plt.show()  # Show plot


class MultiTopicLoggerNode(Node):
    def __init__(self, topics: list):
        super().__init__('multi_topic_logger_node')
        self.loggers = [Float64StampedLogger(self, topic) for topic in topics]

    def shutdown(self, save_individual=True):
        csv_filename = 'all_topics_output.csv'
        with open(csv_filename, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['topic', 'time', 'value'])
            for logger in self.loggers:
                logger.save_to_file(csv_writer)
                logger.shutdown(save_individual)

    def __del__(self):
        self.shutdown()

def load_topics_from_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config['topics']


def main(args=None):
    parser = argparse.ArgumentParser(description="Multi-Topic Logger")
    parser.add_argument(
        '-t', '--topics', type=str, nargs='*', help='Specify the list of topics to subscribe to'
    )
    parser.add_argument(
        '-y', '--yaml', type=str, help='Specify a YAML file with the list of topics'
    )
    parser.add_argument(
        '--no-individual', action='store_true', help='Disable individual file saving'
    )
    parsed_args = parser.parse_args(args)

    if (parsed_args.yaml):
        topics = load_topics_from_yaml(parsed_args.yaml)
    elif (parsed_args.topics):
        topics = parsed_args.topics
    else:
        raise ValueError('You must specify topics via --topics or --yaml')

    rclpy.init(args=args)
    node = MultiTopicLoggerNode(topics)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        node.get_logger().info("Shutdown initiated by KeyboardInterrupt.")
        # Ensure node shutdown logic is executed
        node.shutdown(save_individual=not parsed_args.no_individual)
        # Safely shutdown ROS context
        rclpy.shutdown()



if __name__ == '__main__':
    main()
