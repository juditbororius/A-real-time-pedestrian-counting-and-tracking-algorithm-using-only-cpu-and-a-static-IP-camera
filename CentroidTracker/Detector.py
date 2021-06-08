import datetime
import os

import numpy as np
import cv2
import pandas as pd
from scipy.spatial import ConvexHull
from collections import deque
from pyglet.window import key
from Functions import *

width = 960
height = 720

df = pd.read_csv('Variables/Kkmoon botiga.txt', delimiter=';', index_col='variables')
mintresh = eval(df.loc['minthresh'][0])
if mintresh[0] != None:
    xminthresh = mintresh[0]
else:
    xminthresh = 0
if mintresh[1] != None:
    yminthresh = mintresh[1]
else:
    yminthresh = 0
maxthresh = eval(df.loc['maxthresh'][0])
if maxthresh[0] != None:
    xmaxthresh = maxthresh[0]
else:
    xmaxthresh = width
if maxthresh[1] != None:
    ymaxthresh = maxthresh[1]
else:
    ymaxthresh = height

if eval(df.loc['lists_out'][0]) != None:
    lists_out = []
    for list_out in eval(df.loc['lists_out'][0]):
        lists_out.append(list_out)
else:
    lists_out = None

minarea = eval(df.loc['minarea'][0])
maxarea = eval(df.loc['maxarea'][0])

video_path = '../Videos/Kkmoon botiga.mp4'
#video_path = '../../COUNTING/Videos/23-02-2021.mp4'

#video_path = 'rtsp://192.168.1.90/1'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
ufps = 3
#fps = fps*ufps
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
#out = cv2.VideoWriter('Results/{}_output.avi'.format(video_path.split('/')[-1].split('.')[0]),cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))

method = df.loc['method'][0]
queue = deque()
prova = []
prova_rois = []
Icount = 0


while True:
    #reading each frame of the video
    ret, img = cap.read()
    if ret == True:
        #resizign all the frame to the same
        img = cv2.resize(img, (width,height))
        #preparaing the three frames queue in order to to the mean or the median of these three images
        queue.append(img)
        #enter to the whole detector preocess if the queue has exactly 3 frames
        if len(queue) == int(ufps):
            #copying images for further visualizations and mofications
            real_img = img.copy()
            roi_img = img.copy()
            #selecting which method to combine the three frames (mean is default)
            if method == 'mean':
                img = np.mean(queue, axis=0).astype(dtype=np.uint8)
            if method == 'median':
                img = np.median(queue, axis=0).astype(dtype=np.uint8)
            #These two following lines are to set the variables by hand
            #imgplot = plt.imshow(img)
            #plt.show()
            #applying MOG2 and filters to the resulting mask to improve the detector
            #rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fgmask = fgbg.apply(img)
            kernel = np.ones((2,2), np.uint8)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            #fgmask = cv2.erode(fgmask, kernel, iterations=1)
            #fgmask = cv2.dilate(fgmask, kernel, iterations=1)
            #cv2.imwrite('mask.jpg', fgmask)
            #creating the contours of the objects
            (contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            centroids = []
            areas = []
            rois = []
            tmp_r = []
            extra = fgmask.copy()
            #filtering which contours are pedestrians and which not
            for index, contour in enumerate(contours):
                (x, y, w, h) = cv2.boundingRect(contour)
                (startX, startY, endX, endY) = (x, y, x+w, y+h)
                hull = cv2.convexHull(contour)
                area = cv2.contourArea(hull)
                if area<maxarea and area>minarea and w>yminthresh and h>yminthresh and w<xmaxthresh  and h<ymaxthresh:
                    hull_ = hull[:, 0, :]
                    hull_ = ConvexHull(hull_)
                    centroid = (np.mean(hull_.points[hull_.vertices,0]), np.mean(hull_.points[hull_.vertices,1]))
                    #skipping the objects that are inside the zones where the pedestrians are not able to be detected
                    if any([isinside(list, centroid) for list in lists_out]):
                        continue
                    #preparing the mean of pixels calculation selecting only the bounding box of all the frame
                    roi = roi_img[startY:endY, startX:endX]
                    tmp_r.append(roi)
                    #compiting the average of the bounding box pixels
                    avg = np.average(roi, axis=0)
                    avg = np.average(avg, axis=0)
                    #prepating the lists to call the update fucntion of centroid tracker
                    avg = list(avg)
                    rois.append(avg) #son numeros
                    centroids.append(centroid)
                    areas.append(area)
                    prova.append(area)
            temp_pd = pd.DataFrame({'type': ['rois', 'centroids', 'areas', 'Icount', 'datetime'], 'list':[rois, centroids, areas, Icount, datetime.datetime.now()]})
            temp_pd.to_csv('Temporary/temp.csv')
            #verifying that these three lists have the same length
            assert len(centroids) == len(areas) == len(rois)
            queue.clear()
            prova_rois.append(tmp_r)
            Icount += 1
    else:
        break
foo = open('Temporary/temp.csv', 'r')
foo.close()
os.remove('Temporary/temp.csv')
cap.release()
trackingfile.close()
countingfile.close()