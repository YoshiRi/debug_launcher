#!/usr/bin/env python3
"""
Autowareのデバッグ時に、圧縮画像を生の画像に変換するノードです。
image_transportを使っても下記のように書けますが、QoSがBEST_EFFORTにならないので結局書き直す羽目になりました。

ros2 run image_transport republish --ros-args --remap in:=/sensing/camera/camera0/image_rect_color/compressed --ros-args --remap out:=/sensing/camera/camera0/image_rect_color
"""

import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image


class ImageRepublisher(Node):
    def __init__(self, input_topics, output_topics):
        super().__init__("image_republisher")
        self.bridge = CvBridge()

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT  # または BEST_EFFORT

        self.subs = []
        self.pubs = []

        # make pair of subscriber and publisher
        for in_topic, out_topic in zip(input_topics, output_topics):
            publisher = self.create_publisher(Image, out_topic, qos)
            self.pubs.append(publisher)

            subscriber = self.create_subscription(
                CompressedImage,
                in_topic,
                lambda msg, pub=publisher: self.listener_callback(
                    msg, pub
                ),  # publisherオブジェクトを渡す
                qos,
            )
            self.subs.append(subscriber)

    def listener_callback(self, msg, publisher):
        try:
            # 圧縮画像をOpenCV形式に変換
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
            # OpenCV形式の画像を生のROSメッセージに変換
            raw_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            raw_msg.header = msg.header
            # 生の画像メッセージをパブリッシュ
            publisher.publish(raw_msg)
        except CvBridgeError as e:
            self.get_logger().error("CvBridge Error: {}".format(e))


def main(args=None):
    rclpy.init(args=args)

    # input_topics and output_topics
    input_topics = [
        "/sensing/camera/camera0/image_rect_color/compressed",
        "/sensing/camera/camera1/image_rect_color/compressed",
        "/sensing/camera/camera2/image_rect_color/compressed",
        "/sensing/camera/camera3/image_rect_color/compressed",
        "/sensing/camera/camera4/image_rect_color/compressed",
        "/sensing/camera/camera5/image_rect_color/compressed",
    ]
    output_topics = [
        "/sensing/camera/camera0/image_rect_color",
        "/sensing/camera/camera1/image_rect_color",
        "/sensing/camera/camera2/image_rect_color",
        "/sensing/camera/camera3/image_rect_color",
        "/sensing/camera/camera4/image_rect_color",
        "/sensing/camera/camera5/image_rect_color",
    ]

    image_republisher = ImageRepublisher(input_topics, output_topics)

    rclpy.spin(image_republisher)

    image_republisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
