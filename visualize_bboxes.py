import numpy as np
import gluoncv as gcv
import matplotlib.pyplot as plt

from traffic_signs import TrafficSigns

if __name__ == '__main__':
    root = 'za_traffic_2020/traffic_train'
    idx = np.random.randint(0, 100)
    ts = TrafficSigns(root)
    train_image, train_label = ts[idx]
    bboxes = train_label[:, :4]
    class_ids = train_label[:, 4:5]
    print('Image size (height, width, RGB):', train_image.shape)
    print('Num of objects:', bboxes.shape[0])
    print('Bounding boxes (num_boxes, x_min, y_min, x_max, y_max):\n', bboxes)
    print('Class IDs (num_boxes, ):\n', class_ids)

    gcv.utils.viz.plot_bbox(train_image, bboxes, scores=None, labels=class_ids, class_names=ts.classes)
    plt.show()
