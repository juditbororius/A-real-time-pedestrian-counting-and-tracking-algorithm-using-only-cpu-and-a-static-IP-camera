import os
from statistics import mean
import cv2
import shutil
import numpy as np
import pandas as pd
from collections import deque, Counter
from dataclasses import dataclass
import time
from math import sqrt
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import Variables as var
from numba import jit

@dataclass
class Point:
    x: int
    y: int

@dataclass
class BoundingBox:
    ident: int
    topleft: Point
    width: int
    height: int
    bottomright: Point
    center: Point
    v: Point
    mindist: float
    counted: bool = False
    middle_count: bool = False
    derecha_count: bool = False
    izquierda_count: bool = False

@dataclass
class tempclass:
    ident: int
    mindist: float
    prevcenter: Point
    tempident: int
    new: bool

@dataclass
class shop:
    start: Point
    end: Point
    extense: int
    espace: int
    its_h: bool

def width_and_height(cap):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps= cap.get(cv2.CAP_PROP_FPS)
    return w,h,Nframes, fps

def search_and_reset_dir(path_to_dir):
    if os.path.isdir(path_to_dir) == True:
        shutil.rmtree(path_to_dir)
        os.mkdir(path_to_dir)
    else: os.mkdir(path_to_dir)

@jit(parallel = True, forceobj=True, fastmath = True)
def creating_countours(queue):
    
    #deciding the method
    if var.method == 'mean':
        medianFrame = np.mean(queue, axis=0).astype(dtype=np.uint8)
    if var.method == 'median':
        medianFrame = np.median(queue, axis=0).astype(dtype=np.uint8)
    medianFrame = cv2.resize(medianFrame, var.dim)
    
    #deciding the model
    if var.model == 'MOG2':
        
        #mask applied to the frame
        fgmask = var.fgbg.apply(medianFrame)
        bgmask = cv2.bitwise_not(fgmask)
        
        #resize the mask
        fgmask = cv2.resize(fgmask, var.dim)
        bgmask = cv2.resize(bgmask, var.dim)

        #apply dfilters to remove the noise
        kernel = np.ones((4,4), np.uint8)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        fgmask = cv2.dilate(fgmask, kernel, iterations=1)
        
        #depending on the version of opencv the contours are done in a different way
        if var.version == '2' or float(var.version)>3.0:
            (contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if var.version == '3':
            (im2, contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if  var.model == 'HOG':
        contours, _ = var.fgbg.detectMultiScale(medianFrame, winStride=(8,8), padding=(32,32), scale=1.05) # describing the parameters of HOG and returning them as a Human found function in 'found'

    return medianFrame, contours

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

@jit(forceobj=True)
def contour_calculations(contours, ident, medianFrame, realFrame, dataframe, Icount, memory):
    #curr always will have the bounding boxes of the current (median)frames 
    #and prev will have the bounding boxes of the previous (median)frame
    curr = []
    centers = []
    #create a black frame to draw the contours there
    fg_bounding = 0*np.ones_like(medianFrame)

    #counting variable
    count_people = 0

    tz_ES = pytz.timezone('Europe/Madrid')
    datetime_ES = datetime.now(tz_ES)
    #time = datetime_ES.strftime("%H:%M:%S")
    date = datetime_ES.strftime("%d/%m/%Y")

    for ci, c in enumerate(contours):

        #for each bounding box we are generating the x, y coordinates of the top left and the width and height
        if var.model == 'MOG2':
            (x, y, w, h) = cv2.boundingRect(c)
        if var.model == 'HOG':
            x, y, w, h = c

        #measures restrictions for the bounding boxes: the area, the height and width, etc
        # S'hauria d'afegir alguna restricció de area, tot i que fent les restriccions de altura i amplada és com si hi haguessin d'àres no?
        if (w<var.minthreshwidth or h<var.minthreshheight) or (w>var.maxthreshwidth or h>var.maxthreshheight):
            continue

        M = cv2.moments(c)
        center = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))

        #zonas donde hay movimiento pero no nos interesan
        if isinside(var.l_ofpoints, center)==True or isinside(var.l_ofpoints2, center)==True:
            continue

        #generate a list of bounding boxes objects
        if Icount <= 1:
            bb = BoundingBox(ident, Point(x,y), w, h, Point(x+w, y+h), Point(center[0], center[1]), Point(0,0), 0)
            ident += 1
        else:
            bb = BoundingBox(ident, Point(x,y), w, h, Point(x+w, y+h), Point(center[0], center[1]), Point(0, 0), 0)
            tmplist = []
            bb,ident = id_connections(bb, memory, ident, var.min_bet_frames)
            bb = counting_ins_outs(var.veritas, bb)

            t1 = time.time()
            bb, var.total_derecha, var.total_izquierda = middle_counts(bb, None, None, var.total_derecha, var.total_izquierda) #no le pongo condiciones de posicion, solo de direccion
            print('middle_counts: '+str(time.time()-t1))
            t1 = time.time()
            bb, var.out_derecha, var.in_derecha = derecha_counts(bb, (3/4)*var.W, None, var.out_derecha, var.in_derecha)
            print('derecha_counts: '+str(time.time()-t1))
            bb, var.in_izquierda, var.out_izquierda = izquierda_counts(bb, var.W/4, None, var.in_izquierda, var.out_izquierda)

        dataframe = dataframe.append({'Frame': Icount, 'ID': bb.ident,
                                      'date': date, 'time':time,
                                      'topleft': (bb.topleft.x, bb.topleft.y),
                                      'width':bb.width, 'height':bb.height,
                                      'mindist': bb.mindist,
                                      'center':(bb.center.x, bb.center.y),
                                      'velocity':(bb.v.x, bb.v.y), 'maxID':ident-1, 'counted':bb.counted, 'izquierda/derecha counting':bb.middle_count},
                                     ignore_index = True, sort = False)


        curr.append(bb)

        #draw the bounding boxes in the black frame
        cv2.rectangle(fg_bounding, (x, y), (x + w, y + h), list(np.random.random(size=3) * 256), -1)
        #to correct a little bit the bounding box
        pad_w, pad_h = int(0.15*w), int(0.05*h)



        #cv2.rectangle(realFrame, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 0, 255), 1)
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)

        #generating and drawing the centroid
        cv2.circle(realFrame, center, 3, (0, 0, 255), -1) #hauria de ser el centre (cx y cy pero no sortia molt bé
        cv2.putText(fg_bounding, 'ID{}'.format(bb.ident), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(realFrame, 'ID{}'.format(bb.ident), center, 16, 0.5, (0, 0, 255), 2)
        #cv2.putText(fg_bounding, '(%d, %d)'%(x+w, y+h), (x+w, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        count_people += 1
        centers.append(center)
    return ident, curr, centers, dataframe, count_people, fg_bounding

def id_connections(bb, memory, ident, min_tmp):
    minident = -1
    prevcenter = 0
    theorylist = []
    for numprev in range(len(memory)):  # llista
        tmpident = ident
        prev = memory[numprev]
        for i in prev:
            pos_prevista = Point(i.center.x + (numprev + 1) * i.v.x, i.center.y + (numprev + 1) * i.v.y)
            dist = sqrt((bb.center.x - pos_prevista.x) ** 2 + (bb.center.y - pos_prevista.y) ** 2)
            dist = round(dist, 5)
            if dist < min_tmp:
                min_tmp = dist
                minident = i.ident
                prevcenter = i.center
        if minident == -1:
            temp = tempclass(tmpident, 0, Point(0, 0), tmpident, True)
            temp.tempident += 1
        else:
            temp = tempclass(minident, min_tmp, prevcenter, tmpident, False)
        theorylist.append(temp)
    ids = [i.ident for i in theorylist]
    count_defid = Counter(ids)
    if all(x == 1 for x in count_defid.values()):
        defid = theorylist[-1]
    else:
        defid = theorylist[len(ids) - 1 - ids[::-1].index(count_defid.most_common(1)[0][0])]

    if defid.new:
        bb.ident = defid.ident
        bb.mindist = defid.mindist
        bb.counted = False
        bb.middle_count = False
        bb.derecha_count = False
        bb.izquierda_count = False
    else:
        bb.ident = defid.ident
        bb.mindist = defid.mindist
        bb.v = Point(bb.center.x-defid.prevcenter.x, bb.center.y-defid.prevcenter.y)

        counteds = []
        middle_counteds = []
        derecha_counteds = []
        izquierda_counteds = []
        for mem in memory:
            for bounding in mem:
                if bounding.ident == defid.ident:
                    counteds.append(bounding.counted)
                    middle_counteds.append(bounding.middle_count)
                    derecha_counteds.append(bounding.derecha_count)
                    izquierda_counteds.append(bounding.izquierda_count)
        if any(counteds)==True: bb.counted = True
        else: bb.counted = False
        if any(middle_counteds)==True: bb.middle_count = True
        else: bb.middle_count = False
        if any(derecha_counteds)==True: bb.derecha_count = True
        else: bb.derecha_count = False
        if any(izquierda_counteds)==True: bb.izquierda_count = True
        else: bb.izquierda_count = False

    ident = defid.tempident
    return bb, ident

def creation_in_out_spaces(shop):
    #de momento los horizontales son igual de amplios por dentro y por fuera
    start = [shop.start.x, shop.start.y]
    end = [shop.end.x, shop.end.y]
    espace = shop.espace
    extense = shop.extense

    if shop.its_h:
        shop_in_space = [[start[0] - extense, start[1]],
                         [end[0], end[1]],
                         [end[0], end[1]],
                         [start[0] - extense, start[1]]]

        shop_out_space = [[start[0], start[1]],
                       [end[0] + extense, end[1]],
                       [end[0] + extense, end[1]],
                       [start[0], start[1] + extense]]

    #esta hecho para que la tienda mire hacia abajo y no hacia arriba
    else:
        shop_in_space = [[start[0], start[1] - extense],
                         [end[0], end[1] - extense],
                         [end[0], end[1]],
                         [start[0], start[1]]]

        shop_out_space = [[start[0] - espace, start[1] - espace],
                       [end[0] + espace, end[1] - espace],
                       [end[0] + espace, end[1] + extense - espace],
                       [start[0] - espace, start[1] + extense - espace]]

    return shop_in_space, shop_out_space

def direction_calculation(curr_bb, its_h):
    temp = var.new[var.new['ID'] == curr_bb.ident]
    if its_h:
        all_centers_x = [i[0] for i in temp['center']]
        try:
            mean_x = mean(all_centers_x)
        except:
            mean_x = curr_bb.center.x
        direction = curr_bb.center.x - mean_x
    else:
        all_centers_y = [i[1] for i in temp['center']]
        try:
            mean_y = mean(all_centers_y)
        except:
            mean_y = curr_bb.center.y
        direction = curr_bb.center.y - mean_y
    return direction

def counting_ins_outs(shop, curr_bb):
    shop_in_space, shop_out_space = creation_in_out_spaces(shop)

    if isinside(shop_in_space, (curr_bb.center.x, curr_bb.center.y)) or isinside(shop_out_space, (curr_bb.center.x, curr_bb.center.y)):
        if curr_bb.counted == False:
            direction = direction_calculation(curr_bb, shop.its_h)

            if abs(direction) > 15:
                if direction > 0 and isinside(shop_out_space, (curr_bb.center.x, curr_bb.center.y)):  # negative direction
                    var.total_out += 1
                    curr_bb.counted = True
                if direction < 0 and isinside(shop_in_space, (curr_bb.center.x, curr_bb.center.y)):  # positive direction
                    var.total_in += 1
                    curr_bb.counted = True
        else:
            curr_bb.counted = True
    return curr_bb

def middle_counts(curr_bb, W, H, total_derecha, total_izquierda):
    if curr_bb.middle_count == False:
        direction = direction_calculation(curr_bb, True)
        if abs(direction)>15:
            #como las personas van en horizontal se pone la W si fuesen verticalmente serisa H
            if direction < 0:
                total_izquierda += 1
                curr_bb.middle_count = True

            if direction > 0:
                total_derecha += 1
                curr_bb.middle_count = True
    else:
        curr_bb.middle_count = True
    return curr_bb, total_derecha, total_izquierda

def derecha_counts(curr_bb, W, H, total_derecha, total_izquierda):
    if curr_bb.derecha_count == False:
        direction = direction_calculation(curr_bb, True)
        if abs(direction)>5:
            #como las personas van en horizontal se pone la W si fuesen verticalmente serisa H
            if W is not None:
                if curr_bb.center.x>(W) and direction<0:
                    total_izquierda += 1
                    curr_bb.derecha_count = True

            if W is not None:
                if curr_bb.center.x>(W) and direction>0:
                    total_derecha += 1
                    curr_bb.derecha_count = True

    else:
        curr_bb.derecha_count = True
    return curr_bb, total_derecha, total_izquierda

def izquierda_counts(curr_bb, W, H, total_derecha, total_izquierda):
    if curr_bb.izquierda_count == False:
        direction = direction_calculation(curr_bb, True)
        if abs(direction)>5:
            #como las personas van en horizontal se pone la W si fuesen verticalmente serisa H
            if W is not None:
                if curr_bb.center.x<(W) and direction<0:
                    total_izquierda += 1
                    curr_bb.izquierda_count = True

            if W is not None:
                if curr_bb.center.x<(W) and direction>0:
                    total_derecha += 1
                    curr_bb.izquierda_count = True

    else:
        curr_bb.izquierda_count = True
    return curr_bb, total_derecha, total_izquierda