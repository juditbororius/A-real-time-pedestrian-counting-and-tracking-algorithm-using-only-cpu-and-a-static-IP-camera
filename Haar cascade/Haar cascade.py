import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time
import datetime

videos = os.listdir('../Videos')
W = 960
H = 720

for video in videos:
    # Source data : Video File
    video_path = '../Videos/{}'.format(video)
    f = open('Results/{}_output.txt'.format(video_path.split('/')[-1].split('.')[0]), 'w')

    # Read the source video file
    vs = cv2.VideoCapture(video_path)

    # pre trained classifiers
    pedestrian_classifier = 'models/haarcascade_fullbody.xml'

    # Classified Trackers
    pedestrian_tracker = cv2.CascadeClassifier(pedestrian_classifier)

    totalFrames = vs.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vs.get(cv2.CAP_PROP_FPS)
    duration = int(totalFrames/fps)
    duration = str(datetime.timedelta(seconds = duration))
    print(duration)
    out = cv2.VideoWriter('Results/{}_output.avi'.format(video_path.split('/')[-1].split('.')[0]),
                          cv2.VideoWriter_fourcc(*'DIVX'), fps, (W, H))

    # loop over frames from the video stream
    start_time = time.time()
    while True:
        # start reading video file frame by frame like an image
        (read_successful, frame) = vs.read()
        frame = cv2.resize(frame, (W, H))

        if read_successful:
            #convert to grey scale image
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            break

        # Detect Cars, Pedestrians, Bus and 2Wheelers
        pedestrians = pedestrian_tracker.detectMultiScale(gray_frame,1.1,9)

        # Draw square around the pedestrians
        for (x, y, w, h) in pedestrians:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Human', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        # display the imapge with the face spotted
        out.write(frame)
        cv2.imshow('Detect Pedestrians',frame)

        # capture key
        key = cv2.waitKey(1)

        # Stop incase Esc is pressed
        if key == 27:
            break

    # Release video capture object
    end_time = time.time()
    new_duration = int(end_time-start_time)
    new_fps = int(totalFrames/new_duration)
    new_duration = 	str(datetime.timedelta(seconds = new_duration))
    f.write('The video {} has:\n - FPS: {}\n -Duration: {}\nThe execution of the Detector MobileNet SSD has been done in {}, so the FPS of this detector is: {}'.format(video, fps, duration, new_duration, new_fps))
    out.release()
    vs.release()
    cv2.destroyAllWindows()