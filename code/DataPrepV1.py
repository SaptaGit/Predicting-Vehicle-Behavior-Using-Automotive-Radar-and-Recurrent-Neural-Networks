import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import skimage
import math
from scipy.io import savemat
import csv
import shutil
from math import log, exp, tan, atan, pi, ceil, cos, sin
from pyproj import Proj, transform
from scipy import dot
import imutils
#from keras.models import Sequential
from keras.models import Sequential
from keras.utils import Sequence
from keras.layers import LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dropout, MaxPooling1D
import matplotlib.pyplot as plt
from keras import optimizers
from keras.utils import multi_gpu_model
from sklearn.preprocessing import normalize
import random


# Specify the test trajectory csv file
testTrajFilePath = '/home/saptarshi/PythonCode/Junction/data/junc.csv'

# Set the different Occupancy Grid map and scene dimensions

# Create the visible window
# cv2.namedWindow('test', cv2.WINDOW_NORMAL)


# Model parametrs 
historyTemporal = 20
futureTemporal = 30
validationVehicleCount = 2
validationFileName = 'temporal30Long.txt'


# Convert Lat lon to pixel
EARTH_RADIUS = 6378137
EQUATOR_CIRCUMFERENCE = 2 * pi * EARTH_RADIUS
INITIAL_RESOLUTION = EQUATOR_CIRCUMFERENCE / 256.0
ORIGIN_SHIFT = EQUATOR_CIRCUMFERENCE / 2.0

def latlontopixels(lat, lon, zoom):
    mx = (lon * ORIGIN_SHIFT) / 180.0
    my = log(tan((90 + lat) * pi/360.0))/(pi/180.0)
    my = (my * ORIGIN_SHIFT) /180.0
    res = INITIAL_RESOLUTION / (2**zoom)
    px = (mx + ORIGIN_SHIFT) / res
    py = (my + ORIGIN_SHIFT) / res
    return px, py



# Create the two dictionaries one based on FrameID and other based on VehicleID
def CreateVehicleAndFrameDict(loadFileName):

    print('Creating Vehicle and Frame based dictionary')

    loadFile = open(loadFileName, 'r')
    loadReader = csv.reader(loadFile)
    loadDataset = []
    for loadRow in loadReader:
        # if (loadRow[0] == '738' or loadRow[0] == '1755'): # remove two extreme car for better resolution
        #     continue
        loadDataset.append(loadRow[0:24])

    loadDataset.pop(0)
    sortedList = sorted(loadDataset, key=lambda x: (float(x[0]), float(x[1])))
    datasetArray = np.array(sortedList, dtype=np.float)

    # Normalize each feature columns
    localXIndex = 4
    localYIndex = 5
    velocityIndex = 11
    laneIDIndex = 13
    directionIndex = 18
    movementIndex = 19

    normalizeIndexList = [localXIndex,velocityIndex,laneIDIndex,directionIndex,movementIndex,localYIndex]

    for eachIndex in normalizeIndexList:
        minVal = min(datasetArray[:,eachIndex])
        maxVal = max(datasetArray[:,eachIndex])
        datasetArray[:,eachIndex] = (datasetArray[:,eachIndex] - minVal)/(maxVal - minVal)

    #Create Dictionary for Mapper
    mapper = dict()

    # Create Dictionary with unique Frames
    uniquFrameIds = list(np.unique(datasetArray[:,1]))
    frameKeys = []
    for idx in range(0, len(uniquFrameIds)):
        frameKeys.append(str(uniquFrameIds[idx]))

    dictionaryByFrames = {key : list() for key in frameKeys}

    for jdx in range(0,len(datasetArray)):
        key = str(datasetArray[jdx,1])
        dictionaryByFrames[key].append(datasetArray[jdx])

    # Create Dictionary with unique Vehicles
    uniquVehicleIds = list(np.unique(datasetArray[:,0]))
    vehicleKeys = []
    for idx in range(0, len(uniquVehicleIds)):
        vehicleKeys.append(str(uniquVehicleIds[idx]))

    dictionaryByVehicles = {key : list() for key in vehicleKeys}

    for jdx in range(0,len(datasetArray)):
        key = str(datasetArray[jdx,0])
        if len(dictionaryByVehicles[key])==0:
            dictionaryByVehicles[key].append(datasetArray[jdx])
            continue
        lastFrame = dictionaryByVehicles[key][-1][1]
        lastTime = dictionaryByVehicles[key][-1][3]
        currentFrame = datasetArray[jdx][1]
        currentTime = datasetArray[jdx][3]
        if(abs(currentFrame-lastFrame)==1 and abs(currentTime-lastTime)==100):
            dictionaryByVehicles[key].append(datasetArray[jdx])
        else:
            if key in mapper:
                updatedKey = mapper[key]
                lastFrame = dictionaryByVehicles[updatedKey][-1][1]
                lastTime = dictionaryByVehicles[updatedKey][-1][3]
                currentFrame = datasetArray[jdx][1]
                currentTime = datasetArray[jdx][3]
                if(abs(currentFrame-lastFrame)==1 and abs(currentTime-lastTime)==100):                    
                    dictionaryByVehicles[updatedKey].append(datasetArray[jdx])
                else:
                    print('Wrong Assumption regarding the  presensce of one vehicle ID exists only twice...')
                    print('The problem occured for vehicle ID: ' + key + ' at frame: ' + str(currentFrame) + '...')
                    sys.exit()
            else:
                currentKeys = list(dictionaryByVehicles.keys())
                currentKeys.sort(key=float)
                newKey = str(float(currentKeys[-1]) + 1)
                mapper[key] = newKey
                dictionaryByVehicles[newKey] = list()
                dictionaryByVehicles[newKey].append(datasetArray[jdx])

    loadFile.close()

    return dictionaryByFrames,dictionaryByVehicles

# Plot all cars trajectory on the global GPS map
def TrainData(inputFileName):

    # Load the Vehicle and Frame based Dictionaries
    dictByFrames,dictByVehicles = CreateVehicleAndFrameDict(inputFileName)
    finalVehicleKeys = list(dictByVehicles.keys())
    finalVehicleKeys.sort(key=float)
    finalFrameKeys = list(dictByFrames.keys())
    finalFrameKeys.sort(key=float)

    # Randomly sample 300 validation vehicles and write to file for testing
    validationVehicles = random.sample(finalVehicleKeys,validationVehicleCount)
    with open(validationFileName, 'w') as f:
        for item in validationVehicles:
            f.write("%s\n" % item)

    targetSections = [2,3]
    targetIntersections = [2]

    straightCount = 0
    leftTurnCount = 0
    rightTurnCount = 0
    processedIds = []

    finalXTrain = []
    finalYTrain = []
    finalXVal = []
    finalYVal = []

    for currentVehicle in finalVehicleKeys:
        currentVehicleList = dictByVehicles[str(currentVehicle)]
        # Add the check for the side origins and side destination
        sideOrigin = currentVehicleList[0][14]
        sideDestination = currentVehicleList[0][15]
        if ((sideOrigin == 111 and sideDestination == 203) or (sideOrigin == 103 and sideDestination == 211) or (sideOrigin == 105 and sideDestination == 210) or (sideOrigin == 110 and sideDestination == 205) or (sideOrigin == 107 and sideDestination == 209) or (sideOrigin == 109 and sideDestination == 207)):
            continue

        #print('Processing Vehicle : ' + str(currentVehicle))
        currentVehicleLength = len(currentVehicleList)

        for idx in range(historyTemporal,currentVehicleLength-futureTemporal):

            localXData = []
            for jdx in range(idx-historyTemporal,idx):
                localX = currentVehicleList[jdx][4]
                localY = currentVehicleList[jdx][5]
                velocity = currentVehicleList[jdx][11]
                laneID = currentVehicleList[jdx][13]
                direction = currentVehicleList[jdx][18]
                movement = currentVehicleList[jdx][19]
                localXData.append([localX,velocity,laneID,direction,movement,localY])

            nextMovement = currentVehicleList[idx + futureTemporal][19]

            # Next movements are 0, 0.5, 1 due to normalization 1/0 - through (TH), 2/0.5 - left-turn (LT), 3/1 - right-turn.....

            if(nextMovement == 0):
                localYData = [1,0,0]
            elif(nextMovement == 0.5):
                localYData = [0,1,0]
            elif(nextMovement == 1):
                localYData = [0,0,1]
            else:
                print('Unknow movement data!!!')
                sys.exit()

            if(currentVehicle in validationVehicles):
                finalXVal.append(localXData)
                finalYVal.append(localYData)
            else:
                finalXTrain.append(localXData)
                finalYTrain.append(localYData)

    finalXTrainArray = np.array(finalXTrain)
    finalYTrainArray = np.array(finalYTrain)
    finalXValArray = np.array(finalXVal)
    finalYValArray = np.array(finalYVal)

    return finalXTrainArray,finalYTrainArray,finalXValArray,finalYValArray


if __name__ == '__main__':

    XTrain,YTrain,XVal,YVal = TrainData(testTrajFilePath)
    temporal = XTrain.shape[1]
    features = XTrain.shape[2]

    model = Sequential()
    model.add(LSTM(256, activation='relu', return_sequences=True,input_shape=(temporal,features)))
    model.add(LSTM(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    model.fit(XTrain, YTrain, batch_size=128, epochs=5, verbose=1, validation_data=(XVal,YVal))

    model.save('./models/temporal30Long.h5')

    print('All the cars are plotted in the scene.')

    sys.exit()







