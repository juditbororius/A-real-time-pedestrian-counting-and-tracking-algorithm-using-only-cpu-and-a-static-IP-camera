from imageai.Detection import ObjectDetection
import os
import cv2
import imutils
import numpy as np
import dlib
import time
import datetime
import matplotlib.pyplot as plt

W = 960
H = 720

videos = os.listdir('../Videos')
detector = ObjectDetection()

detector.setModelTypeAsRetinaNet()
detector.setModelPath("resnet50_coco_best_v2.1.0.h5")
detector.loadModel()
custom = detector.CustomObjects(person=True)

for video in videos:
    video_path = '../Videos/{}'.format(video)
    f = open('Results/{}_output.txt'.format(video_path.split('/')[-1].split('.')[0]), 'w')
    cap = cv2.VideoCapture(video_path)

    Icount = 0
    dim = (W, H)

    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = int(totalFrames/fps)
    duration = str(datetime.timedelta(seconds = duration))
    print(duration)
    out = cv2.VideoWriter('Results/{}_output.avi'.format(video_path.split('/')[-1].split('.')[0]),
                          cv2.VideoWriter_fourcc(*'DIVX'), fps, (W, H))

    # loop over frames from the video stream
    start_time = time.time()

    while True:
        success, img = cap.read()
        img = cv2.resize(img, dim)
        image_show = img

        # Getting corners around the person
        output_img, detections = detector.detectCustomObjectsFromImage(custom_objects=custom,
                                                                       input_type='array',
                                                                       input_image=img,
                                                                       output_type='array',
                                                                       minimum_percentage_probability=50)

        for detection in detections:
            (x1, y1, x2, y2) = [int(v) for v in detection['box_points']]
            cv2.rectangle(image_show, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('image', image_show)
        out.write(image_show)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    end_time = time.time()
    new_duration = int(end_time - start_time)
    new_fps = int(totalFrames / new_duration)
    new_duration = str(datetime.timedelta(seconds=new_duration))
    f.write(
        'The video {} has:\n - FPS: {}\n -Duration: {}\nThe execution of the Detector MobileNet SSD has been done in {}, so the FPS of this detector is: {}'.format(
            video, fps, duration, new_duration, new_fps))
    out.release()
    cap.release()
    cv2.destroyAllWindows()

