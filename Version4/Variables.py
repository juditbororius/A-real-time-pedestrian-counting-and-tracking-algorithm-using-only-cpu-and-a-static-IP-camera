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

out = cv2.VideoWriter('DemoOutputs/outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, dim)
directory = 'Frames'
search_and_reset_dir(directory)

model = 'MOG2'

if model == 'MOG2':
    #check opencv version
    if version == '2' :
        fgbg = cv2.BackgroundSubtractorMOG2()
    if float(version) >= float('3'):
        fgbg = cv2.createBackgroundSubtractorMOG2()
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

l_ofpoints = [[0, 0], [0, 300], [960, 350], [960, 0]]
l_ofpoints2 = [[0, 500], [0, 720], [960, 720], [960, 600]]

#input('Write how many frames do you want to make the median. \n')
new = pd.DataFrame()
q = deque()
memory = deque()
prev = []
frame_array = []

ufps = 15
memory_frames = 3
method = 'mean'

#TIENDA
total_in = 0
total_out = 0

#CONTEO EN EL MEDIO
total_izquierda = 0
total_derecha = 0

#CONTEO EN LA DERECHA
in_derecha = 0
out_derecha = 0

#CONTEO EN LA IZQUIERDA
in_izquierda = 0
out_izquierda = 0

#horizontal shop variables
veritas = shop(Point(740, 500), Point(900, 500), 120, 70, False)