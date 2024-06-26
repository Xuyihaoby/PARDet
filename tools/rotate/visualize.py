from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmcv import Config
import numpy as np
import cv2 as cv
import random
import glob
import os
import os.path as osp
import argparse
import tqdm
import xml.etree.ElementTree as ET

path = '/data/Aerial/DOTA1_0/origin/train/images'
labels = '/data/Aerial/DOTA1_0/origin/train/labelTxt'



DIOR={'airplane':(220, 20, 60),
      'airport': (119, 11, 32),
      'baseballfield':(0, 0, 142),
      'basketballcourt':(0, 0, 230),
      'bridge':(106, 0, 228),
      'chimney':(0, 60, 100),
      'Expressway-Service-area':(0, 80, 100),
      'Expressway-toll-station': (0, 0, 70),
      'dam':(0, 0, 192),
      'golffield': (250, 170, 30),
      'groundtrackfield':(100, 170, 30),
      'harbor':(220, 220, 0),
      'overpass':(175, 116, 175),
      'ship':(250, 0, 30),
      'stadium':(165, 42, 42),
      'storagetank':(255, 77, 255),
      'tenniscourt':(0, 226, 252),
      'trainstation':(182, 182, 255),
      'vehicle':(0, 82, 0),
      'windmill':(120, 166, 157)}

def visualGT(cfg, num, dstpath, sp=None):
    dstpath = osp.join(dstpath, 'GT')
    os.makedirs(dstpath, exist_ok=True)
    img_path = cfg.data.train.img_prefix
    if cfg.data.train.type in ['SSDD']:
        _img_path = osp.join(img_path, 'JPEGImages')
    if sp == 'None' and cfg.data.train.type in ['DOTADatasetV1']:
        # _img_path = path
        _img_path = img_path
        imglist = glob.glob(_img_path + '/*')
        num = len(imglist)
        selectnum = min(num, len(imglist))
        _tovis = random.sample(imglist, selectnum)
        for _singlevis in tqdm.tqdm(_tovis):
            img = cv.imread(_singlevis)
            ann_path = _singlevis.replace('images', 'labelTxt').replace('png', 'txt')
            assert osp.exists(ann_path), 'ann_file must be a file'
            with open(ann_path, 'r') as fread:
                lines = fread.readlines()
                if len(lines) == 0:
                    continue
                nplines = []
                # read lines
                for line in lines:
                    line = line.split()
                    if len(line) < 4:
                        continue
                    npline = np.array(line[:8], dtype=np.float32).astype(np.int32)
                    nplines.append(npline[np.newaxis])
                nplines = np.concatenate(nplines, 0).reshape(-1, 4, 2)
                cv.polylines(img, nplines, isClosed=True, color=(255, 125, 125), thickness=3)
                cv.imwrite(osp.join(dstpath, osp.basename(_singlevis)), img)
    elif sp == 'None' and cfg.data.train.type in ['DIOR']:
        ann_path = cfg.data.train.ann_file
        with open(ann_path, 'r') as f:
            imglist = f.readlines()
        selectnum = min(num, len(imglist))
        _tovis = random.sample(imglist, selectnum)
        for _singlevis in tqdm.tqdm(_tovis):
            img_path = osp.join(cfg.data.train.img_prefix, _singlevis.strip()+'.jpg')
            img = cv.imread(img_path)
            ann_path = osp.join(cfg.data.train.data_root, 'Annotations/Oriented Bounding Boxes',
                                f'{_singlevis.strip()}.xml')

            tree = ET.parse(ann_path)
            root = tree.getroot()
            polygons = []
            _labels = []
            for obj in root.findall('object'):
                label = obj.find('name').text
                bbox = []
                for key in ['x_left_top', 'y_left_top', 'x_right_top', 'y_right_top',
                            'x_right_bottom', 'y_right_bottom', 'x_left_bottom',
                            'y_left_bottom']:
                    bbox.append(obj.find('robndbox').find(key).text)
                # Coordinates may be float type
                poly = list(map(float, bbox))
                poly = np.array(poly, dtype=np.int32)
                nplines = poly.reshape(-1, 4, 2)
                cv.polylines(img, nplines, isClosed=True, color=DIOR[label], thickness=3)
            cv.imwrite(osp.join(dstpath, _singlevis.strip() + '.jpg'), img)
    else:
        pass




def visualINF(cfg, num, dstpath, version='v1'):
    dstpath = osp.join(dstpath, 'INF')
    os.makedirs(dstpath, exist_ok=True)
    device = 'cuda:0'
    # init a detector
    model = init_detector(cfg, checkpoint_file, device=device)
    img_path = cfg.data.test.img_prefix
    if cfg.data.test.type in ['SSDD']:
        with open(osp.join(img_path, 'test.txt'), 'r') as f:
            lines = f.readlines()
        imglist = [glob.glob(osp.join(img_path, 'JPEGImages', lines[i].strip()+'*'))[0] for i in range(len(lines))]
        print(len(imglist))
    if cfg.data.test.type in ['HRSC']:
        ann_file = cfg.data.test.ann_file
        with open(ann_file, 'r') as f:
            lines = f.readlines()
        imglist = [osp.join(img_path, 'AllImages',
                                      lines[i].strip() + '.bmp')for i in range(len(lines))]
    else:
        imglist = glob.glob(img_path + '/*')
    selectnum = min(num, len(imglist))
    _tovis = random.sample(imglist, selectnum)
    for _singlevis in tqdm.tqdm(_tovis):
        base_name = os.path.basename(_singlevis)
        dst_name = os.path.join(dstpath, base_name)
        # inference the demo image
        result = inference_detector(model, _singlevis)
        show_result_pyplot(model, _singlevis, result, dst_name, score_thr=0.3, version=version)

def visualSP(cfg, dstpath, name):
    dstpath = osp.join(dstpath, 'SP')
    os.makedirs(dstpath, exist_ok=True)
    device = 'cuda:0'
    # init a detector
    model = init_detector(cfg, checkpoint_file, device=device)
    img_path = cfg.data.test.img_prefix
    img_name = osp.join(img_path, name)
    dst_name = os.path.join(dstpath, name)
    result = inference_detector(model, img_name)
    show_result_pyplot(model, img_name, result, dst_name, 0.3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='test config file path')
    # parser.add_argument('pth', help='the pth need to load')
    parser.add_argument('--mode', default='INF', help='the way you need to visualize the result(GT/INF/SP)')
    parser.add_argument('--dst', default='./checkpoints/visual')
    parser.add_argument('--num', type=int, default=20, help='the number of images you want to visual')
    parser.add_argument('--gtpath', default='None')
    parser.add_argument('--img', default='None')
    parser.add_argument('--version', default='v1')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    os.makedirs(args.dst, exist_ok=True)

    checkpoint_file = osp.join(cfg.work_dir, 'latest.pth')

    if args.mode == 'GT':
        visualGT(cfg, args.num, args.dst, args.img)

    elif args.mode == 'INF':
        visualINF(cfg, args.num, args.dst, args.version)

    elif args.mode == 'SP':
        visualSP(cfg, args.dst, args.img)


