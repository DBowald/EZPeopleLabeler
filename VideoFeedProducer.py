from multiprocessing import Process, Queue
import time
import numpy as np
import cv2
from functools import partial


def main(args):
    raw_frame_queue = Queue()

    vsp_proc = Process(target=video_stream_producer, args=(raw_frame_queue,))
    people_labeler = Process(target=f2, args=(raw_frame_queue,))

    vsp_proc.start()
    people_labeler.start()
    vsp_proc.join()
    people_labeler.join()


def video_stream_producer(RawFrameQueue, video_path='chaplin.mp4'):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        RawFrameQueue.put(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

def f2(q):
    while(True):
        img = q.get()
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(img, kernel, iterations=1)
        cv2.imshow('eroded', erosion)
        if cv2.waitKey(1000//60) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = None
    main(args)