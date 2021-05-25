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
from IPython.display import display
import psutil


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
    counts: list

@dataclass
class tempclass:
    ident: int
    mindist: float
    prevcenter: Point
    tempident: int
    new: bool

@dataclass
class zone:
    start: Point
    end: Point
    extense: int
    its_h: bool

def width_and_height(cap):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps= cap.get(cv2.CAP_PROP_FPS)
    return w,h,Nframes, fps

def search_and_reset_dir_file(path_to_dir_file, dir_file):
    if dir_file == 'directory':
        if os.path.isdir(path_to_dir_file) == True:
            shutil.rmtree(path_to_dir_file)
            os.mkdir(path_to_dir_file)
        else:
            os.mkdir(path_to_dir_file)
    elif dir_file == 'file':
        if os.path.exists(path_to_dir_file) == True:
            os.remove(path_to_dir_file)
            f = open("Execution time.txt", "a")
        else: f = open("Execution time.txt", "a")
        return f
    else:
        print('The path or the input is not correct!')

#@jit(parallel = True, forceobj=True, fastmath = True)
def creating_countours(queue):
    #deciding the method
    for i in range(len(queue)):
        queue[i] = cv2.resize(queue[i], var.dim)
    if var.method == 'mean':
        medianFrame = np.mean(queue, axis=0).astype(dtype=np.uint8)
    if var.method == 'median':
        medianFrame = np.median(queue, axis=0).astype(dtype=np.uint8)
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
        contours, _ = var.fgbg.detectMultiScale(medianFrame, winStride=(8,8), padding=(32,32), scale=1.05) # describing the parameters of HOG and returning them as a Human found function in 'foun
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
    time_col = datetime_ES.strftime("%H:%M:%S")
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
        if any([isinside(list, center) for list in var.lists_out])==True:
            continue

        #generate a list of bounding boxes objects
        if Icount <= 1:
            bb = BoundingBox(ident, Point(x,y), w, h, Point(x+w, y+h), Point(center[0], center[1]), Point(0,0), [0] * len(var.zones))
            ident += 1
        else:
            bb = BoundingBox(ident, Point(x,y), w, h, Point(x+w, y+h), Point(center[0], center[1]), Point(0, 0), [0] * len(var.zones))

            bb,ident = id_connections(bb, memory, ident, var.min_bet_frames)

            bb = counting_ins_outs(var.veritas, bb)

            bb, var.total_derecha, var.total_izquierda = middle_counts(bb, var.W/2, None, var.total_derecha, var.total_izquierda) #no le pongo condiciones de posicion, solo de direccion
            bb, var.out_derecha, var.in_derecha = derecha_counts(bb, (6/7)*var.W, None, var.out_derecha, var.in_derecha)
            bb, var.in_izquierda, var.out_izquierda = izquierda_counts(bb, var.W/7, None, var.in_izquierda, var.out_izquierda)

        dataframe = dataframe.append({'Frame': Icount, 'ID': bb.ident,
                                      'date': date, 'time':time_col,
                                      'topleft': (bb.topleft.x, bb.topleft.y),
                                      'width':bb.width, 'height':bb.height,
                                      'mindist': bb.mindist,
                                      'center':(bb.center.x, bb.center.y),
                                      'velocity':(bb.v.x, bb.v.y), 'maxID':ident-1, 'in_counted':bb.in_counted, 'out_counted':bb.out_counted,
                                      'izquierda/derecha counting':bb.middle_count},
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
        cv2.putText(fg_bounding, ' ID{}'.format(bb.ident), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(realFrame, ' ID{}'.format(bb.ident), center, 16, 0.5, (0, 0, 255), 2)
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
    else:
        bb.ident = defid.ident
        bb.mindist = defid.mindist
        bb.v = Point(bb.center.x-defid.prevcenter.x, bb.center.y-defid.prevcenter.y)

        for i, zone in enumerate(var.zones)
        in_counteds = []
        out_counteds = []
        middle_counteds = []
        derecha_counteds = []
        izquierda_counteds = []
        for mem in memory:
            for bounding in mem:
                if bounding.ident == defid.ident:
                    in_counteds.append(bounding.in_counted)
                    out_counteds.append(bounding.out_counted)
                    middle_counteds.append(bounding.middle_count)
                    derecha_counteds.append(bounding.derecha_count)
                    izquierda_counteds.append(bounding.izquierda_count)
        if any(in_counteds)==True: bb.in_counted = True
        else: bb.in_counted = False
        if any(out_counteds)==True: bb.out_counted = True
        else: bb.out_counted = False
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
    extense = shop.extense

    if shop.its_h:
        shop_in_space = [[start[0] - extense, start[1]],
                         [end[0], end[1]],
                         [end[0], end[1]],
                         [start[0] - extense, start[1]]]

        shop_out_space = [[start[0] + extense, start[1]],
                         [end[0], end[1]],
                         [end[0], end[1]],
                         [start[0] + extense, start[1]]]

    #esta hecho para que la tienda mire hacia abajo y no hacia arriba
    else:
        shop_in_space = [[start[0], start[1] - extense],
                         [end[0], end[1] - extense],
                         [end[0], end[1]+20],
                         [start[0], start[1]+20]]

        shop_out_space = [[start[0], start[1] + extense],
                         [end[0], end[1] + extense],
                         [end[0], end[1]-20],
                         [start[0], start[1]-20]]

    return shop_in_space, shop_out_space

def direction_calculation(curr_bb):
    temp = var.new[var.new['ID'] == curr_bb.ident]

    all_centers_x = [i[0] for i in temp['center']]
    all_centers_y = [i[1] for i in temp['center']]
    try:
        mean_x = mean(all_centers_x)
        mean_y = mean(all_centers_y)
    except:
        mean_x = curr_bb.center.x
        mean_y = curr_bb.center.y
    direction_y = curr_bb.center.y - mean_y
    direction_x = curr_bb.center.x - mean_x

    return direction_y, direction_x

def counting_ins_outs(shop, curr_bb):
    shop_in_space, shop_out_space = creation_in_out_spaces(shop)

    if isinside(shop_in_space, (curr_bb.center.x, curr_bb.center.y)) or isinside(shop_out_space, (curr_bb.center.x, curr_bb.center.y)):
        if curr_bb.out_counted == False:
            direction_y, direction_x = direction_calculation(curr_bb)
            if abs(direction_y) > 0 and abs(direction_y) < 10:
                if direction_y > 0 and isinside(shop_out_space, (curr_bb.center.x, curr_bb.center.y)) and abs(direction_x)<5:  # negative direction_y
                    var.total_out += 1
                    curr_bb.out_counted = True
                    print('{} OUT direction_y {} and direction_x {}'.format(curr_bb.ident, direction_y, direction_x))
                    if any(var.new[var.new['ID']==curr_bb.ident].in_counted)==True:
                        print('This OUT has been counted as IN previously')

                    time.sleep(5)
        if curr_bb.in_counted == False:
            direction_y, direction_x = direction_calculation(curr_bb)
            if abs(direction_y) > 0 and abs(direction_y) < 10:
                if direction_y < 0 and isinside(shop_in_space, (curr_bb.center.x, curr_bb.center.y)) and abs(direction_x)<5:  # positive direction_y
                    var.total_in += 1
                    curr_bb.in_counted = True
                    print('{} IN direction_y {} and direction_x {}'.format(curr_bb.ident, direction_y, direction_x))
                    if any(var.new[var.new['ID']==curr_bb.ident].out_counted)==True:
                        print('This IN has been counted as OUT previously')
                    time.sleep(5)
        else:
            curr_bb.counted = True
    return curr_bb

def middle_counts(curr_bb, W, H, total_derecha, total_izquierda):
    if curr_bb.middle_count == False:
        _, direction = direction_calculation(curr_bb)
        if abs(direction)>15:
            #como las personas van en horizontal se pone la W si fuesen verticalmente serisa H
            if direction < 0 and curr_bb.center.x<W:
                total_izquierda += 1
                curr_bb.middle_count = True

            if direction > 0 and curr_bb.center.x>W:
                total_derecha += 1
                curr_bb.middle_count = True
    else:
        curr_bb.middle_count = True
    return curr_bb, total_derecha, total_izquierda

def derecha_counts(curr_bb, W, H, total_derecha, total_izquierda):
    if curr_bb.derecha_count == False:
        _, direction = direction_calculation(curr_bb)
        if abs(direction)>5:
            #como las persoas van en horizontal se pone la W si fuesen verticalmente serisa H
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
        _, direction = direction_calculation(curr_bb)
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