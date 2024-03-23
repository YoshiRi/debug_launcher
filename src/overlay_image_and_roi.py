#!/usr/bin/env python3
"""
Autowareのデバッグ時にYOLOのROIと画像を重ねて表示するためのノード
"""
import argparse
import datetime
import threading
from datetime import datetime as dt
from functools import partial
from typing import Dict, Tuple, List

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from tier4_perception_msgs.msg import DetectedObjectsWithFeature

NUM_IMAGE = 2
LABEL_COLOR_MAP = {
    0: (255, 0, 100),  # UNKNOWN
    1: (255, 255, 0),  # CAR
    2: (80, 127, 255),  # TRUCK
    3: (0, 0, 255),  # BUS
    4: (0, 140, 255),  # TRAILER
    5: (120, 20, 255),  # MOTORCYCLE
    6: (120, 120, 200),  # BICYCLE
    7: (0, 226, 0),  # PEDESTRIAN
}

def draw_title_image(img:np.array, title:str):
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
    ):
        super().__init__("rois_viewer")

        self.resize_rate = resize_rate
        self.show_timestamp = show_timestamp
        self.is_raw_image = is_raw_image
        self.product = product
        self.show_title = show_title
        self.color_map: Dict[str, Tuple[int, int, int]] = LABEL_COLOR_MAP

        self.lock = threading.Lock()
        image_msg = Image if is_raw_image else CompressedImage
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        num_image = len(in_image_topics)
        for i in range(num_image):
            topic = in_image_topics[i]
            rois_topic = in_rois_topics[i]
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
        self.images = {i: None for i in range(NUM_IMAGE)}
        self.rois = {i: None for i in range(NUM_IMAGE)}
        self.image_stamps = {i: None for i in range(NUM_IMAGE)}
        self.rois_stamps = {i: None for i in range(NUM_IMAGE)}
        self.image_buff = {i: [] for i in range(NUM_IMAGE)}
        self.image_stamp_buff = {i: [] for i in range(NUM_IMAGE)}
        self.rois_buff = {i: [] for i in range(NUM_IMAGE)}
        self.rois_stamp_buff = {i: [] for i in range(NUM_IMAGE)}

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
        min_offset_index = [None for i in range(NUM_IMAGE)]
        for i in range(NUM_IMAGE):
            min_offset = 0.1
            for index, item in enumerate(stamp_buff[i]):
                offset = stamp - item
                if offset >= 0 and offset < 0.1 and np.abs(offset) < min_offset:
                    min_offset = np.abs(offset)
                    min_offset_index[i] = index
        return min_offset_index

    def is_synced(self, stamp):
        roi_indices = self.lookup_index(self.rois_stamp_buff, stamp)
        img_indices = self.lookup_index(self.image_stamp_buff, stamp)
        num_image = len(roi_indices)
        for i in range(NUM_IMAGE):
            if roi_indices[i] is None or img_indices[i] is None:
                return False

        for i in range(NUM_IMAGE):
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
        titles = ["yolox", "bytetrack"]
        if all([img is not None for img in self.images.values()]):
            if self.show_timestamp:
                for i in range(NUM_IMAGE):
                    image = self.images[i]
                    timestamp = self.image_stamps[i]
                    image = cv2.putText(
                        image,
                        str(timestamp),
                        (10, 100),
                        cv2.FONT_HERSHEY_PLAIN,
                        6,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    image = cv2.putText(
                        image,
                        str(dt.fromtimestamp(timestamp)),
                        (10, 200),
                        cv2.FONT_HERSHEY_PLAIN,
                        6,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    self.images[i] = image

            for i in range(NUM_IMAGE):
                img_shape = self.images[i].shape
                for roi in self.rois[i]:
                    bbox = roi.feature.roi
                    cls = roi.object.classification[0].label
                    color = self.color_map[cls]
                    fit_bbox_to_image(img_shape, bbox)
                    if not is_valid_bbox(bbox):
                        print(f"invalid bbox:", bbox, " skip drawing to image ", i)
                        continue
                    cv2.rectangle(
                        self.images[i],
                        (bbox.x_offset, bbox.y_offset),
                        (bbox.x_offset + bbox.width, bbox.y_offset + bbox.height),
                        color=color,
                        thickness=3,
                    )
                if self.show_title:
                    draw_title_image(self.images[i], titles[i])

            img = cv2.hconcat([self.images[0], self.images[1]])

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
        "-i",
        type=str,
        default="/sensing/camera/camera0/image_rect_color/compressed",
    )
    parser.add_argument(
        "--rois_topic",
        type=str,
        default="/perception/object_recognition/detection/rois0",
    )
    parser.add_argument("--rename_keyward", "-k", type=str, default="camera")
    parser.add_argument("--resize_rate", "-r", type=float, default=0.9)
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
    args = parser.parse_args()

    return args


def main(args=None):
    args = parse_args()
    rclpy.init()

    input_images = ['/sensing/camera/camera0/image_rect_color/compressed', '/sensing/camera/camera0/image_rect_color/compressed']
    input_rois = ['/perception/object_recognition/detection/rois0', '/perception/object_recognition/detection/tracked/rois0']

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
