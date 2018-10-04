import argparse

import torch
import cv2
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image

from demo.live import ObjectDetector
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
args = parser.parse_args()



if __name__ == '__main__':

    # get cfg 
    if args.dataset == 'VOC':
        cfg = (VOC_300, VOC_512)[args.size == '512']
    else:
        cfg = (COCO_300, COCO_512)[args.size == '512']

    # get model generator
    if args.version == 'RFB_vgg':
        from models.RFB_Net_vgg import build_net
    elif args.version == 'RFB_mobile':
        from models.RFB_Net_mobile import build_net
        cfg = COCO_mobile_300
    elif args.version == 'SSD_vgg':
        from models.SSD_vgg import build_net
    elif args.version == 'FSSD_vgg':
        from models.FSSD_vgg import build_net
    elif args.version == 'FRFBSSD_vgg':
        from models.FRFBSSD_vgg import build_net
    else:
        print('Unkown version!')


    priorbox = PriorBox(cfg)
    priors = Variable(priorbox.forward(), volatile=True)

    img_dim = (300, 512)[args.size == '512']
    num_classes = (21, 81)[args.dataset == 'COCO']
    net = build_net(img_dim, num_classes)

    # load resume network
    resume_net_path = args.basenet
    print('Loading resume network: %s' % (resume_net_path))
    state_dict = torch.load(resume_net_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

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
    net.load_state_dict(new_state_dict)
    net.eval()
    print('Finished loading model!')
    print(net)
    # load data
    if args.cuda:
        print("use cuda")
        net = net.cuda()
        priors = priors.cuda()
        cudnn.benchmark = True
    else:
        priors = priors.cpu()
        net = net.cpu()

    detector = Detect(num_classes, 0, cfg)
    rgb_means = ((104, 117, 123), (103.94, 116.78, 123.68))[args.version == 'RFB_mobile']
    rgb_std = (1, 1, 1)
    transform = BaseTransform(net.size, rgb_means, rgb_std, (2, 0, 1))
    object_detector = ObjectDetector(net, detector, transform, cuda=args.cuda, priors=priors)
    image = np.array(Image.open('demo/test_image.jpg'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    detect_bboxes = object_detector.predict(image)
    for class_id, class_collection in enumerate(detect_bboxes):
        if len(class_collection) > 0:
            for i in range(class_collection.shape[0]):
                if class_collection[i, -1] > 0.6:
                    pt = class_collection[i]
                    cv2.rectangle(image, (int(pt[0]), int(pt[1])), (int(pt[2]),
                                                                    int(pt[3])), COLORS[i % 3], 2)
                    cv2.putText(image, labelmap[class_id], (int(pt[0]), int(pt[1])), FONT,
                                0.5, (255, 255, 255), 2)
    cv2.imwrite('5566.jpg', image)



