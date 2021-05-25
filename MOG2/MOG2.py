import time

import cv2
import numpy as np
from pyglet.window import key
import pandas as pd
import matplotlib.pyplot as plt

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

W = 960
H = 720

video_path = '../Videos/Kkmoon botiga.mp4'
cap = cv2.VideoCapture(video_path)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=False)

df = pd.read_csv('Variables/{}.txt'.format(video_path.split('/')[-1].split('.')[0]), delimiter=';', index_col='variables')
mintresh = eval(df.loc['minthresh'][0])
xminthresh = mintresh[0]
yminthresh = mintresh[1]
maxthresh = eval(df.loc['maxthresh'][0])
xmaxthresh = maxthresh[0]
ymaxthresh = maxthresh[1]
lists_out = []
for list_out in eval(df.loc['lists_out'][0]):
    lists_out.append(list_out)
kernel_input = tuple(eval(df.loc['kernel'][0]))
time_start = time.time()
while True:
    ret, img = cap.read()
    if ret:
        img = cv2.resize(img, (960,720))
        fgmask_orig = fgbg.apply(img)
        kernel = np.ones(kernel_input, np.uint8)
        fgmask = cv2.morphologyEx(fgmask_orig, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        kernel1 = np.ones((5,5), np.uint8)
        fgmask1 = cv2.morphologyEx(fgmask_orig, cv2.MORPH_OPEN, kernel1)
        fgmask1 = cv2.morphologyEx(fgmask1, cv2.MORPH_CLOSE, kernel1)
        kernel2 = np.ones((1,1), np.uint8)
        fgmask2 = cv2.morphologyEx(fgmask_orig, cv2.MORPH_OPEN, kernel2)
        fgmask2 = cv2.morphologyEx(fgmask2, cv2.MORPH_CLOSE, kernel2)
        if time.time() - time_start > 1:
            cv2.imwrite('original_image.png', img)
            cv2.imwrite('original_mask.png', fgmask_orig)
            cv2.imwrite('mask_with_morph_33.png', fgmask)
            cv2.imwrite('mask_with_morph_55.png', fgmask1)
            cv2.imwrite('mask_with_morph_11.png', fgmask2)
            break
        (contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for index, contour in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            if w > xminthresh and h > yminthresh and w<xmaxthresh and h<ymaxthresh:
                centroid = (int(x+w/2), int(y+h/2))
                if any([isinside(list, centroid) for list in lists_out]):
                    continue
                cv2.rectangle(img,  (x, y, w, h), (0, 255, 0), 2)
                cv2.circle(img, centroid, 1, (255, 255, 255), 1)
        #imgplot = plt.imshow(img)
        #plt.show()
        cv2.imshow('frame', img)
        cv2.imshow('mask', fgmask)

        if cv2.waitKey(1) & 0xFF == key.SPACE:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()