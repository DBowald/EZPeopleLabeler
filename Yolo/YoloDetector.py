from models import *
from utils.utils import *

import torch
from torchvision import transforms
import numpy as np
import cv2
from queue import Empty
import time
import random

class YoloDetector():
    def __init__(self):
        self.config_path = 'Yolo/config/yolov3.cfg'
        self.weights_path = 'Yolo/weights/yolov3.weights'
        self.img_size = 416
        self.min_conf_thresh = 0.85
        self.nms_thresh = 0.3

        self.model = Darknet(self.config_path, img_size=self.img_size)
        self.model.load_darknet_weights(self.weights_path)

        self.input_started = False

        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()


    def yolo_detect_thread(self, raw_frame_queue):
        while True:
            start_time = time.time()
            try:
                if(not self.input_started):
                    img_t = raw_frame_queue.get().cuda()
                    self.input_started = True
                else:
                    img_t = raw_frame_queue.get(timeout=0.5).cuda()
            except Empty:
                break

            inf_start_time = time.time()
            with torch.no_grad():
                detections = self.model(img_t)
                detections = non_max_suppression(detections, self.min_conf_thresh, self.nms_thresh)[0]
                inf_end_time = time.time()

            to_pil_transform = transforms.Compose([transforms.ToPILImage()])
            pil_img = to_pil_transform(img_t.cpu().squeeze())
            cv_frame = np.asarray(pil_img)
            cv_frame = cv2.cvtColor(cv_frame, cv2.COLOR_RGB2BGR)

            end_time = time.time()
            print("Total proc took %f seconds" % (end_time - start_time))
            print("\t * Inference was %f seconds" % (inf_end_time - inf_start_time))


    def yolo_bbox_thread(self, raw_frame_queue):
        yolo_classes = load_classes('Yolo/data/coco.names')
        yolo_class_colors = dict()
        for each_class in yolo_classes:
            B = random.randint(0, 255)
            G = random.randint(0, 255)
            R = random.randint(0, 255)

            yolo_class_colors[each_class] = (B, G, R)

        while True:
            start_time = time.time()
            try:
                if(not self.input_started):
                    img_t = raw_frame_queue.get().cuda()
                    self.input_started = True
                else:
                    img_t = raw_frame_queue.get(timeout=0.5).cuda()
            except Empty:
                break

            inf_start_time = time.time()
            with torch.no_grad():
                detections = self.model(img_t)
                detections = non_max_suppression(detections, self.min_conf_thresh, self.nms_thresh)[0]
                inf_end_time = time.time()

            to_pil_transform = transforms.Compose([transforms.ToPILImage()])
            pil_img = to_pil_transform(img_t.cpu().squeeze())
            cv_frame = np.asarray(pil_img)
            cv_frame = cv2.cvtColor(cv_frame, cv2.COLOR_RGB2BGR)

            if detections is not None:
                for x1, y1, x2, y2, object_conf, class_score, class_pred in detections:
                    class_pred_string = yolo_classes[int(class_pred)]
                    color = yolo_class_colors[class_pred_string]
                    cv2.rectangle(cv_frame, (x1, y1), (x2, y2), color)
                    cv2.rectangle(cv_frame, (x1,y1), (x1 + 60, y1 - 25), color, -1)
                    cv2.putText(cv_frame, class_pred_string,
                        (x1, y1-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255,255,255))
                    cv2.putText(cv_frame, "%d%% conf" % (int(float(object_conf) * 100)),
                        (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255,255,255))

            cv2.imshow('YOLO_DETECTIONS', cv_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            end_time = time.time()
            print("Total proc took %f seconds" % (end_time - start_time))
            print("\t * Inference was %f seconds" % (inf_end_time - inf_start_time))

        cv2.destroyAllWindows()