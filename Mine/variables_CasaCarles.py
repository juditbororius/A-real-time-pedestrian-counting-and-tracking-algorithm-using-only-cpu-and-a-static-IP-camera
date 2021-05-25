from functions import *

version = cv2.__version__.split('.')[0]

#reading and writing video file
W = 960
H = 720
dim = (W, H)

video_path = '../Videos/23-02-2021.mp4'
cap = cv2.VideoCapture(video_path)
width, height, Nframes, fps = width_and_height(cap)
print('The dimensions of the video are (width, height) = (%d, %d), it has %d frames and fps = %d \n'
      %(width, height, Nframes, fps))

directory = 'Frames2'
_ = search_and_reset_dir_file(directory, 'directory')

file_exec_time = 'Execution time.txt'
execution_time_file = search_and_reset_dir_file(file_exec_time, 'file')


out = cv2.VideoWriter('DemoOutputs/outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, dim)
model = 'MOG2'

if model == 'MOG2':
    #check opencv version
    if version == '2' :
        fgbg = cv2.BackgroundSubtractorMOG2()
    if float(version) >= float('3'):
        fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
if model == 'HOG':
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

Icount = 0
contour_areas = []
shadow_mask = np.zeros((height, width, 3))
D_hue = np.zeros((height, width))

ident = 0
minthreshwidth = 30
maxthreshwidth = 80
minthreshheight = 50
maxthreshheight = 200
min_bet_frames = 80  #distancia mínima en el que un ID se conecte, la distancia maxima que una persona podria hacer


l_ofpoints = [[0, 0], [0, 300], [W, 350], [W, 0]]

l_ofpoints2 = [[0, 500], [0, H], [W, H], [W, 600]]

lists_out = [l_ofpoints, l_ofpoints2]

#input('Write how many frames do you want to make the median. \n')
new = pd.DataFrame()
memory = deque()
q = deque()
prev = []

ufps = 3
memory_frames = 8
method = 'mean'

#horizontal shop variables
#ZONAS
zona1 = zone(Point(W/9, 300), Point(W/9, 600), -W/9, True)
zona2 = zone(Point((8/9)*W, 295), Point((8/9)*W, 650), W/9, True)
veritas = zone(Point(780, 450), Point(860, 450), 60, False)
zones = [zona1, zona2, veritas]

#counts zonas
total_in = [0]*len(zones)
total_out = [0]*len(zones)

ids_counted = [[]]*len(zones)

plt.rcParams['animation.html'] = 'jshtml'
fig = plt.figure()
ax = fig.add_subplot(111)
#fig.show()
