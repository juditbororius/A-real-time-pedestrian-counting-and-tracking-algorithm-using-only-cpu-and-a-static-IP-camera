import datetime
import os

from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2
from dataclasses import dataclass
from pyglet.window import key
from scipy.spatial import ConvexHull
import time
from statistics import mean
from collections import deque
from scipy.linalg import norm
import matplotlib.pyplot as plt
import pandas as pd

distances = []

class CentroidTracker():
    def __init__(self, maxDisappeared=50, maxDisappearedAtZones=2):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.areas = OrderedDict()
        self.rois = OrderedDict()
        self.disappeared = OrderedDict()
        self.position_memory = OrderedDict()
        self.area_memory = OrderedDict()
        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared
        self.totalIn = [0] * len(zones)
        self.totalOut = [0] * len(zones)

    def register(self, centroid, area, roi):
        # when registering an object we use the next available object
        # ID to store the centroid
        tmp = 0
        for i, zona in enumerate(zones):
            if isinside(creation_in_out_spaces(zona), (centroid[0], centroid[1])):
                self.totalIn[i] += 1
                tmp += 1
        if tmp > 0: print('There is one NEW appeared in in the middle of the frame --> ID: ', self.nextObjectID)
        self.objects[self.nextObjectID] = centroid
        self.areas[self.nextObjectID] = area
        self.rois[self.nextObjectID] = roi
        self.position_memory[self.nextObjectID] = [list(centroid)]
        self.area_memory[self.nextObjectID] = [area]
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        tmp = 0
        for i, zona in enumerate(zones):
            if isinside(creation_in_out_spaces(zona), (objects[objectID][0], objects[objectID][1])):
                self.totalOut[i] += 1
                tmp += 1
        if tmp > 0: print('This detection has desappeared outside the ROIs ---> ID: ', objectID)
        del self.objects[objectID]
        del self.areas[objectID]
        del self.rois[objectID]
        del self.area_memory[objectID]
        del self.disappeared[objectID]

    def update(self, inputCentroids, inputAreas, inputRois):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(inputCentroids) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if (self.disappeared[objectID] > self.maxDisappeared):
                    self.deregister(objectID)
                '''if any([isinside(zona, self.objects[objectID]) for zona in lists_out]):
                    if self.disappeared[objectID] > maxDisappearedAtZones:
                        self.deregister(objectID)'''
            # return early as there are no centroids or tracking info
            # to update
            return self.objects, self.position_memory, self.rois
            # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], inputAreas[i], inputRois[i])
        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            D[D > 60] = D.max()
            # D[D > 80] = D.max()+1
            '''if D.shape[0] == 1 and D.shape[1] == 1 and D[0, 0] > 60:
                self.register(inputCentroids[0], inputAreas[0], inputRois[0])
                return self.objects, self.position_memory, self.rois'''
            objectAreas = [mean(value) for value in self.area_memory.values()]
            objectRois = list(self.rois.values())
            A = []
            for objectArea in objectAreas:
                sub = [abs(area - objectArea) for area in inputAreas]
                A.append(sub)
            R = []
            for objectRoi in objectRois:
                sub = [abs(sum(roi - objectRoi)) for roi in inputRois]
                R.append(sub)
            A = np.array(A)
            R = np.array(R)
            assert A.shape == D.shape == R.shape
            D = np.interp(D, (D.min(), D.max()), (0, len(D)))
            A = np.interp(A, (A.min(), A.max()), (0, len(A) * 2))
            R = np.interp(R, (R.min(), R.max()), (0, len(R)))
            comb = np.add(A, D)
            comb = np.add(comb, R)
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = comb.min(axis=1).argsort()
            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = comb.argmin(axis=1)[rows]
            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue
                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.position_memory[objectID].append(list(inputCentroids[col]))
                self.areas[objectID] = inputAreas[col]
                self.area_memory[objectID].append(inputAreas[col])
                self.disappeared[objectID] = 0
                # indicate that we have examined each of the row and
                # column indexes, respectively
                distances.append(D[row][col])
                usedRows.add(row)
                usedCols.add(col)
            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
                    '''if any([isinside(zona, self.objects[objectID]) for zona in lists_out]):
                        if self.disappeared[objectID] > maxDisappearedAtZones:
                            self.deregister(objectID)'''

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], inputAreas[col], inputRois[col])
        # return the set of trackable objects
        return self.objects, self.position_memory, self.rois


@dataclass
class Point:
    x: int
    y: int


@dataclass
class zone:
    start: Point
    end: Point
    extense: int
    its_h: bool

def creation_in_out_spaces(zone):
    #de momento los horizontales son igual de amplios por dentro y por fuera
    start = [zone.start.x, zone.start.y]
    end = [zone.end.x, zone.end.y]
    extense = zone.extense

    if zone.its_h:
        zone_space = [[start[0] + extense, start[1]],
                         [start[0], start[1]],
                         [end[0], end[1]],
                         [end[0] + extense, end[1]]]

    #esta hecho para que la tienda mire hacia abajo y no hacia arriba
    else:
        zone_space = [[start[0], start[1] - extense],
                         [end[0], end[1] - extense],
                         [end[0], end[1]+20],
                         [start[0], start[1]+20]]

    return zone_space

def isinside(l, p):
    poly = np.array(l,dtype=np.int32)
    poly_new = poly.reshape((-1,1,2))
    result1 = cv2.pointPolygonTest(poly_new, p, False)
    #cv2.polylines(medianFrame,[poly_new],isClosed=True,color=(0,255,0),thickness=10)
    #plt.imshow(medianFrame)
    if result1 == 1.0:
        return True
    else:
        return False

def creation_in_out_spaces(zone):
    #de momento los horizontales son igual de amplios por dentro y por fuera
    start = [zone.start.x, zone.start.y]
    end = [zone.end.x, zone.end.y]
    extense = zone.extense

    if zone.its_h:
        zone_space = [[start[0] + extense, start[1]],
                         [start[0], start[1]],
                         [end[0], end[1]],
                         [end[0] + extense, end[1]]]

    #esta hecho para que la tienda mire hacia abajo y no hacia arriba
    else:
        zone_space = [[start[0], start[1] - extense],
                         [end[0], end[1] - extense],
                         [end[0], end[1]+20],
                         [start[0], start[1]+20]]

    return zone_space


video_path = '../Videos/Kkmoon oficina.mp4'
video_path = '../Videos/Kkmoon botiga.mp4'

width = 960
height = 720

df = pd.read_csv('Variables/{}.txt'.format(video_path.split('/')[-1].split('.')[0]), delimiter=';', index_col='variables')
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

if eval(df.loc['maxarea'][0]) != None:
    maxarea = eval(df.loc['maxarea'][0])
else:
    maxarea = width*height
if eval(df.loc['minarea'][0]) != None:
    minarea = eval(df.loc['minarea'][0])
else:
    minarea = 0

if eval(df.loc['lists_out'][0]) != None:
    lists_out = []
    for list_out in eval(df.loc['lists_out'][0]):
        lists_out.append(list_out)
else:
    lists_out = None

zones = eval(df.loc['zones'][0])

#video_path = 'rtsp://192.168.1.90/1'
video_path = '../../COUNTING/Videos/23-02-2021.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
ufps = 3
#fps = fps*ufps
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
ct = CentroidTracker()
out = cv2.VideoWriter('Results/{}_output.avi'.format(video_path.split('/')[-1].split('.')[0]),cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))
if 'tracking.csv' in os.listdir('Results'):
    os.remove('Results/tracking.csv')
trackingfile = open("Results/tracking.csv", "a")
trackingfile.write('frame; ID; time; position\n')
if 'counting.csv' in os.listdir('Results'):
    os.remove('Results/counting.csv')
countingfile = open("Results/counting.csv", "a")
countingfile.write('frame; time{}; occ_tienda; occupation\n'.format(''.join(['; in_{}; out_{}'.format(i, i) for i in range(len(zones))])))
method = df.loc['method'][0]
queue = deque()
prova = []
prova_rois = []
Icount = 0
previn = 0
prevout = 0
prevocc = 0
prevocc_tienda = 0
prevout_tienda = 0
previn_tienda = 0
while cap.isOpened:
    try:
        ret, img = cap.read()
        if ret == True:
            img = cv2.resize(img, (width,height))
            queue.append(img)
            if len(queue) == int(ufps):
                real_img = img.copy()
                roi_img = img.copy()
                if method == 'mean':
                    img = np.mean(queue, axis=0).astype(dtype=np.uint8)
                if method == 'median':
                    img = np.median(queue, axis=0).astype(dtype=np.uint8)
                #imgplot = plt.imshow(img)
                #plt.show()
                #rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                fgmask = fgbg.apply(img)
                kernel = np.ones((2,2), np.uint8)
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
                #fgmask = cv2.erode(fgmask, kernel, iterations=1)
                #fgmask = cv2.dilate(fgmask, kernel, iterations=1)
                #cv2.imwrite('mask.jpg', fgmask)
                (contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                centroids = []
                areas = []
                rois = []
                tmp_r = []
                extra = fgmask.copy()
                for index, contour in enumerate(contours):
                    (x, y, w, h) = cv2.boundingRect(contour)
                    (startX, startY, endX, endY) = (x, y, x+w, y+h)
                    hull = cv2.convexHull(contour)
                    area = cv2.contourArea(hull)

                    if area<maxarea and area>minarea and w>yminthresh and h>yminthresh and w<xmaxthresh  and h<ymaxthresh:
                        hull_ = hull[:, 0, :]
                        hull_ = ConvexHull(hull_)
                        centroid = (np.mean(hull_.points[hull_.vertices,0]), np.mean(hull_.points[hull_.vertices,1]))
                        if any([isinside(list, centroid) for list in lists_out]):
                            continue
                        roi = roi_img[startY:endY, startX:endX]
                        tmp_r.append(roi)
                        avg = np.average(roi, axis=0)
                        avg = np.average(avg, axis=0)
                        rois.append(avg) #son numeros
                        centroids.append(centroid)
                        cv2.polylines(real_img,[hull],True,(0,255,255), 2)
                        cv2.polylines(extra,[hull],True,(0,255,255), 2)
                        areas.append(area)
                        prova.append(area)
                assert len(centroids) == len(areas) == len(rois)
                objects, memory, outputRois = ct.update(centroids, areas, rois)
                if Icount == 0:
                    prevocc = len(objects)
                for (objectID, centroid) in objects.items():
                    # draw both the ID of the object and the centroid of the
                    # object on the output frame
                    trackingfile.write('{}; {}; {}; {}\n'.format(Icount, objectID, datetime.datetime.now(), centroid))
                    text = "ID {}".format(objectID)
                    cv2.putText(real_img, text, (int(centroid[0]) - 10, int(centroid[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(real_img, (int(centroid[0]), int(centroid[1])), 4, (0, 255, 0), -1)

                txt = 'Total Outs: {}'

                #IZQUIERDA
                cv2.rectangle(real_img, (0, height-(2*40)), (len(txt)*13, height), (255,255,255), thickness=-1)
                cv2.putText(real_img, 'Total Ins: {}'.format(ct.totalIn[0]), (20, int(height)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(real_img, 'Total Outs: {}'.format(ct.totalOut[0]), (20, int(height)-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                #DERECHA
                cv2.rectangle(real_img, (width-len(txt)*13, height-(2*40)), (width, height), (255,255,255), thickness=-1)
                cv2.putText(real_img, 'Total Ins: {}'.format(ct.totalIn[1]), (width-len(txt)*11, int(height)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(real_img, 'Total Outs: {}'.format(ct.totalOut[1]), (width-len(txt)*11, int(height)-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                #TIENDA
                cv2.rectangle(real_img, (zones[2].start.x-20,  zones[2].start.y-zones[2].extense-70), (zones[2].start.x+len(txt)*12, zones[2].start.y-zones[2].extense-10), (255,255,255), thickness=-1)
                cv2.putText(real_img, 'Total Outs: {}'.format(ct.totalIn[2]), (zones[2].start.x, zones[2].start.y-zones[2].extense-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(real_img, 'Total Ins: {}'.format(ct.totalOut[2]), (zones[2].start.x, zones[2].start.y-zones[2].extense-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                occupation = prevocc - (sum(ct.totalOut)-prevout) + (sum(ct.totalIn)-previn)
                occ_tienda = prevocc_tienda - (ct.totalOut[2]-prevout_tienda) + (ct.totalIn[2]-previn_tienda)
                if occ_tienda < 0:
                    occ_tienda = 0
                previn_tienda = ct.totalIn[2]
                prevout_tienda = ct.totalOut[2]
                prevocc = occ_tienda
                if occupation < 0:
                    occupation = 0
                prevout = sum(ct.totalOut)
                previn = sum(ct.totalIn)
                prevocc = occupation
                #print(ct.totalIn, ct.totalOut)
                #print('; '.join(['; '.join([str(i), str(o)]) for i, o in zip(ct.totalIn, ct.totalOut)]))
                countingfile.write('{}; {}; {}; {}; {}\n'.format(Icount, datetime.datetime.now(), '; '.join(['; '.join([str(i), str(o)]) for i, o in zip(ct.totalIn, ct.totalOut)]),occ_tienda, occupation))

                cv2.imshow('frame', real_img)
                cv2.imshow('mask', extra)
                out.write(real_img)
                queue.clear()
                prova_rois.append(tmp_r)
                Icount += 1
                if cv2.waitKey(1) & 0xFF == key.SPACE:
                    break
        else:
            break

    except:
        print(datetime.datetime.now())
        break
cap.release()
out.release()
cv2.destroyAllWindows()
trackingfile.close()
countingfile.close()

