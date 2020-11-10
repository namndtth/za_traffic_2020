import os
import gluoncv as gcv
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt

from pycocotools.coco import COCO


class TrafficSigns(mx.gluon.data.dataset.Dataset):
    def __init__(self, root, min_object_area=0):
        self.root = root
        self.min_object_area = min_object_area
        self.json_id_to_contiguous = None
        self.contiguous_id_to_json = None
        self.classes = None
        self.items, self.labels = self.load_json()

    def __getitem__(self, idx):
        img_path = self.items[idx]
        label = self.labels[idx]
        img = mx.image.imread(img_path, 1)
        return img, np.array(label).copy()

    def __len__(self):
        return len(self.items)

    def load_json(self):
        items = []
        labels = []
        anno = os.path.join(self.root, 'train_traffic_sign_dataset.json')
        traffic_signs = COCO(anno)
        self.classes = [c['name'] for c in traffic_signs.loadCats(traffic_signs.getCatIds())]
        self.json_id_to_contiguous = {
            v: k for k, v in enumerate(traffic_signs.getCatIds())
        }

        image_ids = sorted(traffic_signs.getImgIds())

        for entry in traffic_signs.loadImgs(image_ids):
            abs_path = self.parse_image_path(entry)
            if not os.path.exists(abs_path):
                raise IOError('Images: {} not exists.'.format(abs_path))
            label = self.check_load_bbox(traffic_signs, entry)

            if not label:
                continue
            items.append(abs_path)
            labels.append(label)

        print('len item', len(items))
        return items, labels

    def parse_image_path(self, entry):
        filename = entry["file_name"]
        abs_path = os.path.join(self.root, 'images', filename)
        return abs_path

    def check_load_bbox(self, traffic_signs, entry):
        entry_id = entry["id"]
        width = entry["width"]
        height = entry["height"]

        ann_ids = traffic_signs.getAnnIds(imgIds=entry_id, iscrowd=None)
        objs = traffic_signs.loadAnns(ann_ids)

        valid_objs = []

        for obj in objs:
            if obj["area"] < self.min_object_area:
                continue
            from gluoncv.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(obj['bbox']), width, height)

            if obj['area'] > 0 and xmax > xmin and ymax > ymin:
                contiguous_cid = self.json_id_to_contiguous[obj['category_id']]
                valid_objs.append([xmin, ymin, xmax, ymax, contiguous_cid])

        return valid_objs


