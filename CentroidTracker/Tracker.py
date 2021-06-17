from CentroidTrackerClass import CentroidTracker
from Functions import *
from datetime import datetime
from pyglet.window import key
import os
import pandas as pd
from collections import OrderedDict
import ast
import time

time.sleep(1)

width = 960
height = 720

df = pd.read_csv('Variables/Kkmoon botiga.txt', delimiter=';', index_col='variables')
zones = eval(df.loc['zones'][0])

if 'tracking.csv' in os.listdir('Results'):
    os.remove('Results/tracking.csv')
trackingfile = open("Results/tracking.csv", "a")
print('tracking file is opened')
trackingfile.write('frame; ID; time; position\n')
if 'counting.csv' in os.listdir('Results'):
    os.remove('Results/counting.csv')
countingfile = open("Results/counting.csv", "a")

countingfile.write('frame; time{}; occ_tienda; occupation\n'.format(''.join(['; in_{}; out_{}'.format(i, i) for i in range(len(zones))])))
distances = []

previn = 0
prevout = 0
prevocc = 0
prevocc_tienda = 0
prevout_tienda = 0
previn_tienda = 0
Icount = 0
ct = CentroidTracker()

lastmodified_set = datetime.now()
print('counting file is opened')
while 'temp.csv' in os.listdir('Temporary'):
    lastmodified_file = os.path.getmtime('Temporary/temp.csv')
    if lastmodified_file != lastmodified_set:
        try:
            temp_df = pd.read_csv('Temporary/temp.csv', index_col=0)
            rois = eval(temp_df.loc[0][1])
            centroids = eval(temp_df.loc[1][1])
            areas = eval(temp_df.loc[2][1])
            Icount = int(temp_df.loc[3][1])
            dtime = datetime.strptime((temp_df.loc[4][1]), '%Y-%m-%d %H:%M:%S.%f')
            #print(type(rois), type(centroids), type(areas), type(Icount), type(dtime))
            #print(temp_df)
            objects, memory, outputRois = ct.update(centroids, areas, rois)
            if Icount == 0:
                prevocc = len(objects)
            for (objectID, centroid) in objects.items():
                # draw both the ID of the object and the centroid of the
                # object on the output frame
                trackingfile.write('{}; {}; {}; {}\n'.format(Icount, objectID, dtime, centroid))
            occupation = prevocc - (sum(ct.totalOut) - prevout) + (sum(ct.totalIn) - previn)
            occ_tienda = prevocc_tienda - (ct.totalOut[2] - prevout_tienda) + (ct.totalIn[2] - previn_tienda)

            '''if occ_tienda < 0:
                occ_tienda = 0
            previn_tienda = ct.totalIn[2]
            prevout_tienda = ct.totalOut[2]
            prevocc = occ_tienda'''
            if occupation < 0:
                occupation = 0
            prevout = sum(ct.totalOut)
            previn = sum(ct.totalIn)
            prevocc = occupation
            # print(ct.totalIn, ct.totalOut)
            # print('; '.join(['; '.join([str(i), str(o)]) for i, o in zip(ct.totalIn, ct.totalOut)]))
            countingfile.write('{}; {}; {}; {}; {}\n'.format(Icount, dtime, '; '.join(
                ['; '.join([str(i), str(o)]) for i, o in zip(ct.totalIn, ct.totalOut)]), occ_tienda, occupation))
            if 0xFF == key.SPACE:
                break
            lastmodified_set = lastmodified_file
        except:
            5

trackingfile.close()
countingfile.close()