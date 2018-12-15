import argparse
import cv2
import torch
import numpy as np
from collections import OrderedDict
from typing import Union

import torch.backends.cudnn as cudnn
from eyewitness.detection_utils import DetectionResult
from eyewitness.image_id import ImageId
from eyewitness.object_detector import ObjectDetector
from eyewitness.image_utils import ImageHandler
from torch.autograd import Variable
from PIL import Image

import models
from utils.timer import Timer
from utils.nms.py_cpu_nms import py_cpu_nms
from data import BaseTransform, VOC_300, VOC_512, COCO_300, COCO_512, COCO_mobile_300
from data import VOC_CLASSES as labelmap
from layers.functions import Detect, PriorBox



parser = argparse.ArgumentParser(description='Receptive Field Block Net Training')
parser.add_argument('-s', '--size', default='300', help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC', help='VOC or COCO dataset')
parser.add_argument(
    '--version', default='RFB_vgg', help='RFB_vgg ,RFB_E_vgg RFB_mobile SSD_vgg version.')
parser.add_argument(
    '--basenet', default='weights/RFB300_80_5.pth', help='pretrained base model')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda to train model')

BUILD_NET_METHOD = {
    'RFB_vgg': models.RFB_Net_vgg.build_net,
    'FSSD_vgg': models.FSSD_vgg.build_net,
}

class PytorchDetectorWrapper(ObjectDetector):
    def __init__(
            self, cfg, size, dataset, model_ver, weight_path,
            cuda=False, max_per_image=300, thresh=0.5):

        img_dim = (300, 512)[size == '512']
        num_classes = (21, 81)[dataset == 'COCO']
        self.net = self.create_net(img_dim, num_classes, model_ver, weight_path)

        priorbox = PriorBox(cfg)
        self.priors = Variable(priorbox.forward(), volatile=True)
        self.detection = Detect(num_classes, 0, cfg)

        rgb_means = ((104, 117, 123), (103.94, 116.78, 123.68))[model_ver == 'RFB_mobile']
        rgb_std = (1, 1, 1)
        transform = BaseTransform(self.net.size, rgb_means, rgb_std, (2, 0, 1))
        self.transform = transform
        self.num_classes = num_classes
        self.max_per_image = max_per_image
        self.cuda = cuda
        self.thresh = thresh
        if self.cuda:
            self.priors = self.priors.cuda()
            self.net = self.net.cuda()
        else:
            self.priors = self.priors.cpu()
            self.net = self.net.cpu()

    def load_weights(self, weight_path, location='cpu'):
        # load network
        print('Loading network weihgt from: %s' % (weight_path))
        state_dict = torch.load(weight_path, map_location=location)
        # create new OrderedDict that does not contain `module.`

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            if k.startswith('base.'):
                name = k.replace('base.', 'base.layers.')
            new_state_dict[name] = v
        return new_state_dict
        
    def create_net(self, img_dim, num_classes, model_ver, weight_path):
        assert model_ver in BUILD_NET_METHOD
        build_net = BUILD_NET_METHOD[model_ver]
        net = build_net(img_dim, num_classes)
        new_state_dict = self.load_weights(weight_path)
        net.load_state_dict(new_state_dict)
        net.eval()
        print('Finished loading model!')
        print(net)
        return net

    def predict(self, img):
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]]).cpu().numpy()
        _t = {'im_detect': Timer(), 'misc': Timer()}
        assert img.shape[2] == 3
        x = Variable(self.transform(img).unsqueeze(0), volatile=True).cpu()
        if self.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        out = self.net(x, test=True)  # forward pass
        boxes, scores = self.detection.forward(out, self.priors)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores = scores[0]

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image
        boxes *= scale
        _t['misc'].tic()
        all_boxes = [[] for _ in range(self.num_classes)]

        for j in range(1, self.num_classes):
            inds = np.where(scores[:, j] > self.thresh)[0]
            if len(inds) == 0:
                all_boxes[j] = np.zeros([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            print(scores[:, j])
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            # keep = nms(c_bboxes,c_scores)

            keep = py_cpu_nms(c_dets, 0.45)
            keep = keep[:50]
            c_dets = c_dets[keep, :]
            all_boxes[j] = c_dets
        if self.max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][:, -1] for j in range(1, self.num_classes)])
            if len(image_scores) > self.max_per_image:
                image_thresh = np.sort(image_scores)[-self.max_per_image]
                for j in range(1, self.num_classes):
                    keep = np.where(all_boxes[j][:, -1] >= image_thresh)[0]
                    all_boxes[j] = all_boxes[j][keep, :]

        nms_time = _t['misc'].toc()
        print('net time: ', detect_time)
        print('post time: ', nms_time)
        return all_boxes

    def detect(self, image: Image, image_id: Union[str, ImageId]) -> DetectionResult:
        image_array = np.array(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        detect_bboxes = self.predict(image_array)

        detected_objects = []
        for class_id, class_collection in enumerate(detect_bboxes):
            if len(class_collection) > 0:
                for i in range(class_collection.shape[0]):
                    label = labelmap[class_id]
                    x1, y1, x2, y2, score = class_collection[i]
                    if score > 0.6:
                        detected_objects.append([x1, y1, x2, y2, label, score, ''])

        image_dict = {
            'image_id': image_id,
            'detected_objects': detected_objects,
        }
        detection_result = DetectionResult(image_dict)
        return detection_result
        

if __name__ == '__main__':
    args = parser.parse_args()
    # get cfg 
    if args.dataset == 'VOC':
        cfg = (VOC_300, VOC_512)[args.size == '512']
    else:
        cfg = (COCO_300, COCO_512)[args.size == '512']

    object_detector = PytorchDetectorWrapper(
        cfg, args.size, args.dataset, args.version, args.basenet, cuda=args.cuda)

    image = Image.open('demo/test_image.jpg')
    image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
    detection_result = object_detector.detect(image, image_id)
    ImageHandler.draw_bbox(image, detection_result.detected_objects)
    ImageHandler.save(image, "detected_image/drawn_image.jpg")
