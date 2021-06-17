import cv2
import matplotlib.pyplot as plt

video_path = 0
#video_path = 'rtsp://192.168.1.90/1'

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, img = cap.read()
    if ret:
        img = cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)
        imgplot = plt.imshow(img)
        plt.show()
        break
    else:
        print('Video not found')
        break
    break

cap.release()
