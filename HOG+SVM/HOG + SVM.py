import cv2
import imutils
import numpy as np
import os
import time
import datetime
W = 960
H = 720
videos = os.listdir('../Videos')


def detect(frame, out):
    bounding_box_coordinates, weights = HOGCV.detectMultiScale(frame, padding=(4, 4), scale=1.02) #winStride=(8, 8), padding=(32, 32), scale=1.05
    person = 1
    for i, (x, y, w, h) in enumerate(bounding_box_coordinates):
        x, y, w, h = (x, y, w, h)
        if weights[i]>0.7:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'person {}'.format(person), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            person += 1
        '''if weights[i] < 0.7 and weights[i] > 0.3:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 122, 255), 2)
            cv2.putText(frame, 'person {}'.format(person), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            person += 1'''

    cv2.putText(frame, 'Status: Detecting', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, 'Total Persons: {}'.format(person - 1), (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow('output', frame)
    out.write(frame)
    return out, frame

def humanDetector(video_path):
    print('[INFO] Opening Video from path.')
    vs = cv2.VideoCapture(video_path)
    totalFrames = vs.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vs.get(cv2.CAP_PROP_FPS)
    duration = int(totalFrames / fps)
    duration = str(datetime.timedelta(seconds=duration))
    print(duration)
    out = cv2.VideoWriter('Results/{}_output.avi'.format(video_path.split('/')[-1].split('.')[0]),
                          cv2.VideoWriter_fourcc(*'DIVX'), fps, (W, H))

    #out = cv2.VideoWriter('Results/{}_output.avi'.format(video_path.split('/')[-1].split('.')[0]),-1, fps, (W, H))

    # loop over frames from the video stream
    start_time = time.time()
    check, frame = vs.read()
    if check == False:
        print('Video not Found. Please Enter a Valid Path (Full path of Video Should be provided).')
        return
    print('Detecting people...')
    while vs.isOpened():
        check, frame = vs.read()
        frame = cv2.resize(frame, (W, H))
        if check:
            #frame = cv2.resize(frame, (960,720))
            out, frame = detect(frame, out)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            break
    end_time = time.time()
    new_duration = int(end_time - start_time)
    new_fps = int(totalFrames / new_duration)
    new_duration = str(datetime.timedelta(seconds=new_duration))
    f.write(
        'The video {} has:\n - FPS: {}\n -Duration: {}\nThe execution of the Detector MobileNet SSD has been done in {}, so the FPS of this detector is: {}'.format(
            video, fps, duration, new_duration, new_fps))
    out.release()
    vs.release()
    cv2.destroyAllWindows()


for video in videos:
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    path = '../Videos/{}'.format(video)
    f = open('Results/{}_output.txt'.format(path.split('/')[-1].split('.')[0]), 'w')
    humanDetector(path)