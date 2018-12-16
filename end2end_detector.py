import argparse
import os

import arrow
import cv2
import time
from eyewitness.config import (IN_MEMORY, BBOX)
from eyewitness.image_id import ImageId
from eyewitness.image_utils import (ImageProducer, swap_channel_rgb_bgr, ImageHandler)
from eyewitness.result_handler.db_writer import BboxPeeweeDbWriter
from eyewitness.result_handler.line_detection_result_handler import LineAnnotationSender
from peewee import SqliteDatabase
from PIL import Image

from naive_detector import PytorchDetectorWrapper
from data import BaseTransform, VOC_300, VOC_512, COCO_300, COCO_512, COCO_mobile_300


parser = argparse.ArgumentParser()
'''
Command line options
'''
parser.add_argument('-s', '--size', default='300', help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC', help='VOC or COCO dataset')
parser.add_argument(
    '--version', default='RFB_vgg', help='RFB_vgg ,RFB_E_vgg RFB_mobile SSD_vgg version.'
)
parser.add_argument(
    '--basenet', default='weights/RFB300_80_5.pth', help='pretrained base model'
)
parser.add_argument(
    '--cuda', default=False, type=bool, help='Use cuda to train model'
)
parser.add_argument(
    '--db_path', type=str, default='::memory::',
    help='the path used to store detection result records'
)
parser.add_argument(
    '--interval_s', type=int, default=3, help='the interval of image generation'
)


class InMemoryImageProducer(ImageProducer):
    def __init__(self, video_path, interval_s):
        self.vid = cv2.VideoCapture(video_path)
        self.interval_s = interval_s
        if not self.vid.isOpened():
            raise IOError("Couldn't open webcam or video")

    def produce_method(self):
        return IN_MEMORY

    def produce_image(self):
        while True:
            # clean buffer hack: for Linux V4L capture backend with a internal fifo
            for iter_ in range(5):
                self.vid.grab()
            _, frame = self.vid.read()
            yield Image.fromarray(swap_channel_rgb_bgr(frame))
            time.sleep(self.interval_s)


def image_url_handler(drawn_image_path):
    """if site_domain not set in env, will pass a pickchu image"""
    site_domain = os.environ.get('site_domain')
    if site_domain is None:
        return 'https://upload.wikimedia.org/wikipedia/en/a/a6/Pok%C3%A9mon_Pikachu_art.png'
    else:
        return '%s/%s' % (site_domain, drawn_image_path)


def line_detection_result_filter(detection_result):
    """
    used to check if sent notification or not
    """
    return any(i.label == 'person' for i in detection_result.detected_objects)


if __name__ == '__main__':
    args = parser.parse_args()
    # image producer from webcam
    image_producer = InMemoryImageProducer(0, interval_s=args.interval_s)

    # initialize object_detector
    if args.dataset == 'VOC':
        cfg = (VOC_300, VOC_512)[args.size == '512']
    else:
        cfg = (COCO_300, COCO_512)[args.size == '512']
    object_detector = PytorchDetectorWrapper(
        cfg, args.size, args.dataset, args.version, args.basenet, cuda=args.cuda)

    # detection result handlers
    result_handlers = []

    # update image_info drawn_image_path, insert detection result
    database = SqliteDatabase(args.db_path)
    bbox_sqlite_handler = BboxPeeweeDbWriter(database)
    result_handlers.append(bbox_sqlite_handler)

    # setup your line channel token and audience
    channel_access_token = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
    if channel_access_token:
        line_annotation_sender = LineAnnotationSender(
            channel_access_token=channel_access_token,
            image_url_handler=image_url_handler,
            detection_result_filter=line_detection_result_filter,
            detection_method=BBOX,
            update_audience_period=10,
            database=database)
        result_handlers.append(line_annotation_sender)

    for image in image_producer.produce_image():
        image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
        bbox_sqlite_handler.register_image(image_id, {})
        detection_result = object_detector.detect(image, image_id)

        # draw and save image, update detection result
        drawn_image_path = "detected_image/%s_%s.%s" % (
            image_id.channel, image_id.timestamp, image_id.file_format)
        ImageHandler.draw_bbox(image, detection_result.detected_objects)
        ImageHandler.save(image, drawn_image_path)
        detection_result.image_dict['drawn_image_path'] = drawn_image_path

        for result_handler in result_handlers:
            result_handler.handle(detection_result)
