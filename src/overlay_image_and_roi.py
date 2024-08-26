#!/usr/bin/env python3
"""
Autowareのデバッグ時にYOLOのROIと画像を重ねて表示するためのノード
"""
import argparse
import datetime
import threading
from datetime import datetime as dt
from functools import partial
from typing import Dict, List, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from tier4_perception_msgs.msg import DetectedObjectsWithFeature

NUM_IMAGE = 6
LABEL_COLOR_MAP = {
    0: (255, 0, 100),  # UNKNOWN - Bright Pink
    1: (255, 255, 0),  # CAR - Yellow
    2: (80, 127, 255),  # TRUCK - Light Blue
    3: (0, 0, 255),  # BUS - Blue
    4: (0, 140, 255),  # TRAILER - Azure
    5: (120, 20, 255),  # MOTORCYCLE - Violet
    6: (120, 120, 200),  # BICYCLE - Light Slate Gray
    7: (0, 226, 0),  # PEDESTRIAN - Green
}


def draw_title_image(img: np.array, title: str):
    img = cv2.putText(
        img,
        title,
        (10, 100),
        cv2.FONT_HERSHEY_PLAIN,
        6,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )


def is_valid_bbox(bbox):
    # calc area
    area = bbox.width * bbox.height
    if area <= 0:
        return False
    return True


def fit_bbox_to_image(img_shape, bbox) -> bool:
    img_height, img_width, _ = img_shape
    x_offset_ = bbox.x_offset
    y_offset_ = bbox.y_offset
    width_ = bbox.width
    height_ = bbox.height
    # bbox outside of image
    if bbox.x_offset > img_width or bbox.y_offset > img_height:
        bbox.x_offset, bbox.y_offset, bbox.width, bbox.height = 0, 0, 0, 0
        return False

    # reshape bbox inside image
    if bbox.x_offset < 0:
        x_offset_ = 0
    if bbox.y_offset < 0:
        y_offset_ = 0
    if bbox.x_offset + bbox.width > img_width:
        width_ = img_width - x_offset_
    if bbox.y_offset + bbox.height > img_height:
        height_ = img_height - y_offset_
    bbox.x_offset = x_offset_
    bbox.y_offset = y_offset_
    bbox.width = width_
    bbox.height = height_
    return True


# 単一画像にタイトルを描画する関数
def draw_title_image(img: np.array, title: str) -> None:
    cv2.putText(
        img, title, (10, 100), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 0), 2, cv2.LINE_AA
    )


# バウンディングボックスが有効か判定する関数
def is_valid_bbox(bbox) -> bool:
    area = bbox.width * bbox.height
    return area > 0


# バウンディングボックスを画像の範囲に収める関数
def fit_bbox_to_image(img_shape: Tuple[int, int, int], bbox) -> bool:
    img_height, img_width, _ = img_shape
    x_offset, y_offset, width, height = (
        bbox.x_offset,
        bbox.y_offset,
        bbox.width,
        bbox.height,
    )

    # bbox outside of image
    if x_offset > img_width or y_offset > img_height:
        bbox.x_offset, bbox.y_offset, bbox.width, bbox.height = 0, 0, 0, 0
        return False

    # reshape bbox inside image
    bbox.x_offset = max(x_offset, 0)
    bbox.y_offset = max(y_offset, 0)
    if x_offset + width > img_width:
        bbox.width = img_width - bbox.x_offset
    if y_offset + height > img_height:
        bbox.height = img_height - bbox.y_offset

    return True


# 画像上にROIを描画する関数
def process_image_and_rois(
    image: np.array,
    rois: List,
    color_map: Dict[int, Tuple[int, int, int]],
    show_title: bool,
    title: str = "",
) -> np.array:
    img_shape = image.shape
    for roi in rois:
        bbox = roi.feature.roi
        cls = roi.object.classification[0].label
        color = color_map.get(cls, (255, 255, 255))  # Use white as default color
        if fit_bbox_to_image(img_shape, bbox) and is_valid_bbox(bbox):
            cv2.rectangle(
                image,
                (bbox.x_offset, bbox.y_offset),
                (bbox.x_offset + bbox.width, bbox.y_offset + bbox.height),
                color=color,
                thickness=3,
            )
    if show_title:
        draw_title_image(image, title)
    return image


class ImgViewer(Node):
    def __init__(
        self,
        in_image_topics: List[str],
        in_rois_topics: List[str],
        rename_keyward: str,
        resize_rate: float,
        save_video: bool,
        show_timestamp: bool,
        is_raw_image: bool,
        product: str,
        show_title: bool,
        publish_topic: bool = False,
    ):
        super().__init__("rois_viewer")

        # load from ros parameter
        self.declare_parameters(
            namespace="",
            parameters=[
                ("image_topics", in_image_topics),
                ("rois_topics", in_rois_topics),
            ],
        )
        self.image_topics = self.get_parameter("image_topics").value
        self.rois_topics = self.get_parameter("rois_topics").value
        assert len(self.image_topics) == len(
            self.rois_topics
        ), "image_topics and rois_topics must have the same length"
        # logger
        self.get_logger().info(f"image_topics: {self.image_topics}")
        self.get_logger().info(f"rois_topics: {self.rois_topics}")

        self.resize_rate = resize_rate
        self.show_timestamp = show_timestamp
        self.is_raw_image = is_raw_image
        self.product = product
        self.show_title = show_title
        self.color_map: Dict[str, Tuple[int, int, int]] = LABEL_COLOR_MAP

        self.lock = threading.Lock()
        image_msg = Image if is_raw_image else CompressedImage
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.img_publishers = {i: None for i in range(len(self.image_topics))}

        num_image = len(self.image_topics)
        for i in range(num_image):
            topic = self.image_topics[i]
            rois_topic = self.rois_topics[i]
            self.create_subscription(
                image_msg,
                topic,
                partial(self.image_callback, image_id=i),
                qos_profile,
            )
            self.create_subscription(
                DetectedObjectsWithFeature,
                rois_topic,
                partial(self.rois_callback, image_id=i),
                qos_profile,
            )
            if publish_topic:
                self.img_publishers[i] = self.create_publisher(
                    Image, f"{rename_keyward}{i}/image_with_roi", qos_profile
                )

        # # init publisher
        # self.img_publishers = {i: self.create_publisher(image_msg, f"{rename_keyward}{i}/image_with_roi", qos_profile) for i in range(num_image)}

        self.images = {i: None for i in range(num_image)}
        self.rois = {i: None for i in range(num_image)}
        self.image_stamps = {i: None for i in range(num_image)}
        self.rois_stamps = {i: None for i in range(num_image)}
        self.image_buff = {i: [] for i in range(num_image)}
        self.image_stamp_buff = {i: [] for i in range(num_image)}
        self.rois_buff = {i: [] for i in range(num_image)}
        self.rois_stamp_buff = {i: [] for i in range(num_image)}

        self.fourcc = None
        self.video_writer = None
        self.save_video = save_video

    def __del__(self):
        if self.fourcc is not None:
            self.video_writer.release()

    def image_callback(self, msg, image_id):
        with self.lock:
            np_arr = np.frombuffer(msg.data, np.uint8)
            if self.is_raw_image:
                image = np_arr.reshape((msg.height, msg.width, -1))
            else:
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            image_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

            self.image_buff[image_id].append(image)
            self.image_stamp_buff[image_id].append(image_stamp)

            if self.save_video and self.video_writer is None:
                size = (
                    int(image.shape[1] * self.resize_rate * 2),
                    int(image.shape[0] * self.resize_rate * 1),
                )
                self.fourcc = cv2.VideoWriter_fourcc(*"DIVX")
                self.video_writer = cv2.VideoWriter(
                    "2img_video_{}.avi".format(dt.now()),
                    self.fourcc,
                    10,
                    size,
                )

    def rois_callback(self, msg, image_id):
        with self.lock:
            rois = msg.feature_objects
            rois_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

            self.rois_buff[image_id].append(rois)
            self.rois_stamp_buff[image_id].append(rois_stamp)

            if len(self.rois_stamp_buff[image_id]) < 2:
                return

            if image_id == 1 and self.is_synced(self.rois_stamp_buff[image_id][-2]):
                # print(f"draw image. timestamp: {self.rois_stamps[image_id]}")
                self.show_image()

    def lookup_index(self, stamp_buff, stamp):
        num_image = len(self.image_topics)
        min_offset_index = [None for i in range(num_image)]
        for i in range(num_image):
            min_offset = 0.1
            for index, item in enumerate(stamp_buff[i]):
                offset = stamp - item
                if offset >= 0 and offset < 0.1 and np.abs(offset) < min_offset:
                    min_offset = np.abs(offset)
                    min_offset_index[i] = index
        return min_offset_index

    def is_synced(self, stamp):
        num_image = len(self.image_topics)
        roi_indices = self.lookup_index(self.rois_stamp_buff, stamp)
        img_indices = self.lookup_index(self.image_stamp_buff, stamp)
        num_image = len(roi_indices)
        for i in range(num_image):
            if roi_indices[i] is None or img_indices[i] is None:
                return False

        for i in range(num_image):
            # Set data
            self.images[i] = self.image_buff[i][img_indices[i]]
            self.image_stamps[i] = self.image_stamp_buff[i][img_indices[i]]
            self.rois[i] = self.rois_buff[i][roi_indices[i]]
            self.rois_stamps[i] = self.rois_stamp_buff[i][roi_indices[i]]

            # Remove old data from buffers
            self.image_buff[i] = self.image_buff[i][img_indices[i] + 1 :]
            self.image_stamp_buff[i] = self.image_stamp_buff[i][img_indices[i] + 1 :]
            self.rois_buff[i] = self.rois_buff[i][roi_indices[i] + 1 :]
            self.rois_stamp_buff[i] = self.rois_stamp_buff[i][roi_indices[i] + 1 :]
        return True

    def show_image(self):
        num_image = len(self.image_topics)
        # titles = ["yolox", "bytetrack"]
        if not all([img is not None for img in self.images.values()]):
            return

        # process images to show
        for i in range(num_image):
            process_image_and_rois(
                self.images[i], self.rois[i], self.color_map, self.show_title, ""
            )
            if self.img_publishers[i] is not None:
                msg = Image()
                stamp_sec: float = self.rois_stamps[i]
                msg.header.stamp = rclpy.time.Time(
                    seconds=int(stamp_sec),
                    nanoseconds=int((stamp_sec - int(stamp_sec)) * 1e9),
                ).to_msg()
                msg.height = self.images[i].shape[0]
                msg.width = self.images[i].shape[1]
                msg.encoding = "bgr8"
                msg.is_bigendian = 0
                msg.step = 3 * msg.width
                msg.data = self.images[i].tobytes()
                self.img_publishers[i].publish(msg)
                # msg.format = "jpeg"
                # msg.data = cv2.imencode(".jpg", self.images[i])[1].tobytes()
                # self.img_publishers[i].publish(msg)

        img = cv2.hconcat([self.images[0]])

        # if self.product == "xx1":
        #     img_f = cv2.hconcat([self.images[2], self.images[0], self.images[4]])
        #     img_r = cv2.hconcat([self.images[3], self.images[1], self.images[5]])
        # elif self.product == "x2":
        #     img_f = cv2.hconcat([self.images[5], self.images[0], self.images[1]])
        #     img_r = cv2.hconcat([self.images[4], self.images[3], self.images[2]])
        # else:
        #     img_f = cv2.hconcat([self.images[0], self.images[1], self.images[2]])
        #     img_r = cv2.hconcat([self.images[3], self.images[4], self.images[5]])

        # img = cv2.vconcat([img_f, img_r])
        img_resize = cv2.resize(img, None, fx=self.resize_rate, fy=self.resize_rate)

        cv2.imshow("image", img_resize)
        cv2.waitKey(1)

        if self.save_video:
            self.video_writer.write(img_resize)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_image",
        type=str,
        default="/sensing/camera/camera0/image_rect_color/compressed",
    )
    parser.add_argument(
        "--rois_topic",
        type=str,
        default="/perception/object_recognition/detection/rois0",
    )
    parser.add_argument("--rename_keyward", "-k", type=str, default="camera")
    parser.add_argument("--resize_rate", type=float, default=0.9)
    parser.add_argument("--save_video", "-s", action="store_true")
    parser.add_argument("--raw_image", action="store_true")
    parser.add_argument("--show_timestamp", action="store_true")
    parser.add_argument("--show_title", action="store_true")
    parser.add_argument(
        "--product",
        type=str,
        help="choose sensor config",
        choices=["xx1", "x2"],
        default="xx1",
    )
    args, _ = parser.parse_known_args()
    return args


def main(args=None):
    args = parse_args()
    rclpy.init()

    # print("in_image", args.in_image)
    # print("rois_topic", args.rois_topic)
    input_images = [
        "/sensing/camera/camera0/image_rect_color/compressed",
        "/sensing/camera/camera0/image_rect_color/compressed",
    ]
    input_rois = [
        "/perception/object_recognition/detection/rois0",
        "/perception/object_recognition/detection/tracked/rois0",
    ]

    node = ImgViewer(
        input_images,
        input_rois,
        args.rename_keyward,
        args.resize_rate,
        args.save_video,
        args.show_timestamp,
        args.raw_image,
        args.product,
        args.show_title,
    )
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
