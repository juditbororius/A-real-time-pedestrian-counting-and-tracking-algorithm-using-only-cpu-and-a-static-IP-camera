import numpy as np
import cv2
from dataclasses import dataclass


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

