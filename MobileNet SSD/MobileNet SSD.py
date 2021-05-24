import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time
import datetime

W = 960
H = 720

videos = os.listdir('../Videos')

for video in videos:
	# initialize the list of class labels MobileNet SSD was trained to
	# detect
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]
	# load our serialized model from disk
	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

	print("[INFO] opening video file...")
	path = '../Videos/{}'.format(video)
	print(path.split('/')[-1].split('.')[0])
	f = open('Results/{}_output.txt'.format(path.split('/')[-1].split('.')[0]), 'w')
	vs = cv2.VideoCapture(path)

	totalFrames = vs.get(cv2.CAP_PROP_FRAME_COUNT)
	fps = vs.get(cv2.CAP_PROP_FPS)
	duration = int(totalFrames/fps)
	duration = str(datetime.timedelta(seconds = duration))
	print(duration)
	out = cv2.VideoWriter('Results/{}_output.avi'.format(path.split('/')[-1].split('.')[0]), cv2.VideoWriter_fourcc(*'DIVX'), 15, (W, H))
	# loop over frames from the video stream
	start_time = time.time()
	while vs.isOpened():
		ret, frame = vs.read()
		# if we are viewing a video and we did not grab a frame then we
		# have reached the end of the video
		if ret == True:
			if frame is None:
				break
			frame = cv2.resize(frame, (W, H))
			blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
			net.setInput(blob)
			detections = net.forward()
			# loop over the detections
			for i in np.arange(0, detections.shape[2]):
				confidence = detections[0, 0, i, 2]
				# filter out weak detections by requiring a minimum
				# confidence
				if confidence > 0.3:
					# extract the index of the class label from the
					# detections list
					idx = int(detections[0, 0, i, 1])
					# if the class label is not a person, ignore it
					if CLASSES[idx] != "person":
						continue
					# compute the (x, y)-coordinates of the bounding box
					# for the object
					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startX, startY, endX, endY) = box.astype("int")
					cv2.rectangle(img=frame, rec=(startX, startY, endX-startX, endY-startY), color=(0, 255, 0), thickness=2)
			cv2.imshow('Tracking', frame)
			out.write(frame)

			if cv2.waitKey(1) & 0xff == ord('q'):
				break
		else:
			break

	end_time = time.time()
	new_duration = int(end_time-start_time)
	new_fps = int(totalFrames/new_duration)
	new_duration = 	str(datetime.timedelta(seconds = new_duration))
	f.write('The video {} has:\n - FPS: {}\n -Duration: {}\nThe execution of the Detector MobileNet SSD has been done in {}, so the FPS of this detector is: {}'.format(video, fps, duration, new_duration, new_fps))
	out.release()
	vs.release()
	cv2.destroyAllWindows()