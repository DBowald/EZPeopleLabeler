from multiprocessing import Process, Queue
import time
import numpy as np
import cv2
from queue import Empty
from PIL import Image
from torchvision import transforms

from Yolo import YoloDetector

from functools import partial


def main(args):
    raw_frame_queue = Queue(120)

    vsp_proc = Process(target=video_stream_producer, args=(raw_frame_queue,))
    yolo_proc = Process(target=start_yolo_thread, args=(raw_frame_queue,))

    vsp_proc.start()
    #yd = YoloDetector.YoloDetector()
    #yd.yolo_detect_thread(raw_frame_queue)
    yolo_proc.start()
    vsp_proc.join()
    yolo_proc.join()

def start_yolo_thread(raw_frame_queue):
    yd = YoloDetector.YoloDetector()
    yd.yolo_bbox_thread(raw_frame_queue)

def video_stream_producer(RawFrameQueue, video_path="chaplin.mp4"):
    cap = cv2.VideoCapture(video_path)
    #cap.set(cv2.CAP_PROP_FPS, 10)
    print(cap.get(cv2.CAP_PROP_FPS))
    while True:
        ret, cv_frame = cap.read()
        if ret is False:
            break
        pil_img = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(pil_img)

        img_transform = transforms.Compose([
            transforms.Resize((416, 416)),
            transforms.ToTensor()
        ])
        img_t = img_transform(pil_img)
        img_t = img_t.unsqueeze(0)

        RawFrameQueue.put(img_t)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    args = None
    main(args)