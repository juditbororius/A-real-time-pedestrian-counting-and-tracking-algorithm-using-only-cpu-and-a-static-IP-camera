from scipy.spatial import distance as dist
from collections import OrderedDict
import pandas as pd
from Functions import *
from statistics import mean

width = 960
height = 720

df = pd.read_csv('Variables/Kkmoon botiga.txt', delimiter=';', index_col='variables')
zones = eval(df.loc['zones'][0])

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
            if isinside(creation_in_out_spaces(zona), (self.objects[objectID][0], self.objects[objectID][1])):
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
                sub = [abs(sum(np.array(roi) - objectRoi)) for roi in inputRois]
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