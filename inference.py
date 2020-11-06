import mxnet as mx
import gluoncv as gcv
import cv2
import os
from traffic_signs import TrafficSigns
import matplotlib.pyplot as plt
import shutil
import json
import time
if __name__ == '__main__':
    ctx = [mx.cpu()]
    classes = ['Cấm ngược chiều', 'Cấm dừng và đỗ', 'Cấm rẽ', 'Giới hạn tốc độ', 'Cấm còn lại', 'Nguy hiểm',
               'Hiệu lệnh']

    net = gcv.model_zoo.get_model('yolo3_darknet53_custom',
                                  classes=classes, pretrained_base=False, ctx=ctx)

    net.load_parameters('models/traffic_sign_best.params', ctx=ctx)

    test_folder = 'za_traffic_2020/traffic_public_test/images'
    image_files = os.listdir(test_folder)
    image_files = sorted(image_files)
    shutil.rmtree('inference_results')
    os.mkdir('inference_results')

    results = []
    start = time.time()
    for filename in image_files:
        print(filename)
        x, img = gcv.data.transforms.presets.yolo.load_test(os.path.join(test_folder, filename), short=608)
        class_IDs, scores, bounding_boxes = net(x)

        orginal_img = cv2.imread(os.path.join(test_folder, filename))
        resized = gcv.data.transforms.bbox.resize(bounding_boxes, (img.shape[0], img.shape[1]),
                                                  (orginal_img.shape[0], orginal_img.shape[1]))

    #     xywh = gcv.utils.bbox.bbox_xyxy_to_xywh(resized.asnumpy()[0])
    #
    #     for i in range(len(xywh)):
    #         if scores[0][i] >= 0.3:
    #             print(scores[0][i].asscalar())
    #             detected = {
    #                 "image_id": int(filename.split('.')[0]),
    #                 "category_id": int(class_IDs[0][i].asscalar()),
    #                 "bbox": xywh[i].tolist(),
    #                 "score": scores[0][i].asscalar()
    #             }
    #             results.append(detected)
    #     break
    # with open("submission.json", "w") as file:
    #     json.dump(results, file)
    #
    # print(time.time() - start)
        plotted_img = gcv.utils.viz.cv_plot_bbox(orginal_img, resized[0], scores[0],
                                                 class_IDs[0], class_names=net.classes, thresh=0.3)

        cv2.imwrite(os.path.join('inference_results', filename), plotted_img)
