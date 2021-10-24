import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
from keras.models import load_model



# Specify the test trajectory csv file
testTrajFilePath = '/home/saptarshi/PythonCode/Junction/data/Lankershim.csv'

# Speify the model path
modelPath = '/home/saptarshi/PythonCode/Junction/models/temporal30Long.h5'

# Speify the map path for the map based visualization and load the map
mapFileName = '/home/saptarshi/PythonCode/Junction/Maps/Lanekrshim.png'
mapImage = cv2.imread(mapFileName)

# Model parameters
# To visualize no the global map 1 -> visible 
visual = 0  

# History and Future temporal window length
historyTemporal = 20
futureTemporal = 30
features = 6

validationFileName = 'temporal30Long.txt'

# Class strings
straightStr = 'Straight'
leftTurnStr = 'Left Turn'
rightTurnStr = 'Right Turn'


# Create the visible window if vusal is true
if(visual == 1):
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)


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


# Convert the Global X/Y location to the absolute pixel location on the global map..

# Create the projection from State plane to lat/lon
inProj = Proj(init='epsg:2229', preserve_units = True)
outProj = Proj(init='epsg:4326')

# Get the corner points to calculate the relative movements
cornerLat = 34.143
cornerLon = -118.363
cornerPixelX, cornerPixelY = latlontopixels(cornerLat, cornerLon, 21)

def GlobalPoseToMapPixel(XPoseVal, YPoseVal):
    lon,lat = transform(inProj,outProj,XPoseVal,YPoseVal)
    pX,pY = latlontopixels(lat, lon, 21) 
    dx = int(cornerPixelX - pX )*-1 - 80
    dy = int(cornerPixelY - pY)
    return dx,dy

# Convert the normalized movement values to string 1/0 - through (TH), 2/0.5 - left-turn (LT), 3/1 - right-turn
def movementValueToString(movementValue):
    movementString = ''
    if(movementValue == 0):
        movementString = straightStr
    elif(movementValue == 0.5):
        movementString = leftTurnStr
    elif(movementValue == 1):
        movementString = rightTurnStr
    else:
        print('Unknown movement value!!!')
        sys.exit()
    return movementString

# Calculate the confusion matrix
def CalcConfusionMatrix(predList):
    straightArray = np.array([0,0,0])
    leftTurnArray = np.array([0,0,0])
    rightTurnArray = np.array([0,0,0])

    for eachPredValues in predList:
        gtValue,predValue = eachPredValues
        if gtValue == straightStr:
            if predValue == straightStr:
                straightArray[0] = straightArray[0] + 1
            elif predValue == leftTurnStr:
                straightArray[1] = straightArray[1] + 1
            elif predValue == rightTurnStr:
                straightArray[2] = straightArray[2] + 1
            else:
                print('Unknow movement predicted in the consufion matrix calculation')
                sys.exit()
        elif gtValue == leftTurnStr:
            if predValue == straightStr:
                leftTurnArray[0] = leftTurnArray[0] + 1
            elif predValue == leftTurnStr:
                leftTurnArray[1] = leftTurnArray[1] + 1
            elif predValue == rightTurnStr:
                leftTurnArray[2] = leftTurnArray[2] + 1
            else:
                print('Unknow movement predicted in the consufion matrix calculation')
                sys.exit()
        elif gtValue == rightTurnStr:
            if predValue == straightStr:
                rightTurnArray[0] = rightTurnArray[0] + 1
            elif predValue == leftTurnStr:
                rightTurnArray[1] = rightTurnArray[1] + 1
            elif predValue == rightTurnStr:
                rightTurnArray[2] = rightTurnArray[2] + 1
            else:
                print('Unknow movement predicted in the consufion matrix calculation')
                sys.exit()
        else:
            print('Unknow Groud truth movement in the consufion matrix calculation')
            sys.exit()

    # print('          Staright Left-Trun Right-Turn')
    # print('straight: '  + str(straightArray))
    # print('Left Turn: ' + str(leftTurnArray))
    # print('Right Turn: ' + str(rightTurnArray))
    finalConfusionMatrix = np.column_stack((straightArray,leftTurnArray,rightTurnArray))
    return finalConfusionMatrix

# Caluclate the class based Accuracy
def classBasedAcc(inputConfMat):
    straightAcc  = (inputConfMat[0,0]+inputConfMat[1,1]+inputConfMat[1,2]+inputConfMat[2,1]+inputConfMat[2,2])/(sum(sum(inputConfMat)))
    leftTurnAcc  = (inputConfMat[1,1]+inputConfMat[0,0]+inputConfMat[0,2]+inputConfMat[2,0]+inputConfMat[2,2])/(sum(sum(inputConfMat)))
    rightTurnAcc = (inputConfMat[2,2]+inputConfMat[0,0]+inputConfMat[0,1]+inputConfMat[1,0]+inputConfMat[1,1])/(sum(sum(inputConfMat)))
    return round(straightAcc,3),round(leftTurnAcc,3),round(rightTurnAcc,3)

# Calculate the Precision for each class
def classBasedPrecision(inputConfMat):
    columnBasedSum = sum(inputConfMat)
    straightPrec = inputConfMat[0,0]/columnBasedSum[0]
    leftTurnPrec = inputConfMat[1,1]/columnBasedSum[1]
    rightTurnPrec = inputConfMat[2,2]/columnBasedSum[2]
    return round(straightPrec,3),round(leftTurnPrec,3),round(rightTurnPrec,3)


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
def PredData(inputFileName):

    #load the model
    model = load_model(modelPath)

    # Initialize a list to collect the predicted and ground truth results for futher confusuin matrix calculation
    resultList = []  

    # Load the Vehicle and Frame based Dictionaries
    dictByFrames,dictByVehicles = CreateVehicleAndFrameDict(inputFileName)
    finalVehicleKeys = list(dictByVehicles.keys())
    finalVehicleKeys.sort(key=float)
    finalFrameKeys = list(dictByFrames.keys())
    finalFrameKeys.sort(key=float)

    targetSections = [2,3]
    targetIntersections = [2]

    straightCount = 0
    leftTurnCount = 0
    rightTurnCount = 0
    processedIds = []

    f = open(validationFileName, "r")
    ValidationVehicles = f.read().splitlines()

    for currentVehicle in finalVehicleKeys:
        # Perfrom prediction only for the vehicles in the validation list..
        if currentVehicle not in ValidationVehicles:
            continue

        currentVehicleList = dictByVehicles[str(currentVehicle)]

        # Add the check for the side origins and side destination
        sideOrigin = currentVehicleList[0][14]
        sideDestination = currentVehicleList[0][15]
        if ((sideOrigin == 111 and sideDestination == 203) or (sideOrigin == 103 and sideDestination == 211) or (sideOrigin == 105 and sideDestination == 210) or (sideOrigin == 110 and sideDestination == 205) or (sideOrigin == 107 and sideDestination == 209) or (sideOrigin == 109 and sideDestination == 207)):
            continue

        # Quick check for the moment:
        # origin = currentVehicleList[0][14]
        # destination = currentVehicleList[0][15]
        # if not ((origin == 101 or origin == 102) and destination == 203):
        #     continue

        print('Processing Vehicle : ' + str(currentVehicle))
        currentVehicleLength = len(currentVehicleList)

        for idx in range(historyTemporal,currentVehicleLength-futureTemporal):

            localXTrain = []
            for jdx in range(idx-historyTemporal,idx):
                # Get Global X/Y for plotting on map visualization
                globalX = currentVehicleList[jdx][6]
                globalY = currentVehicleList[jdx][7]
                
                # Get the mentioned features for prediction 
                localX = currentVehicleList[jdx][4]
                localY = currentVehicleList[jdx][5]
                velocity = currentVehicleList[jdx][11]
                laneID = currentVehicleList[jdx][13]
                direction = currentVehicleList[jdx][18]
                movement = currentVehicleList[jdx][19]
                localXTrain.append([localX,velocity,laneID,direction,movement,localY])

            # Get the ground truth next movement
            groundTurthNextMovement = currentVehicleList[idx + futureTemporal][19]

            # Prepere the input array and predict the next movement
            localXTrainArray = np.array(localXTrain).reshape(1,historyTemporal,features)
            #predictedMovement = model.predict(localXTrainArray)

            # Convert the predicted movement to normalized vales..... (0,0.5,1) 1/0 - through (TH), 2/0.5 - left-turn (LT), 3/1 - right-turn
            # predictedMovementValue = -1
            # if((predictedMovement == np.array([1,0,0])).all()):
            #     predictedMovementValue = 0
            # elif((predictedMovement == np.array([0,1,0])).all()):
            #     predictedMovementValue = 0.5
            # elif((predictedMovement == np.array([0,0,1])).all()):
            #     predictedMovementValue = 1
            # else:
            #     print('Unknow movement data predicted!!!')
            #     print(predictedMovement)
            #     sys.exit()

            # Convert the predicted movement to normalized vales..... (0,0.5,1) 1/0 - through (TH), 2/0.5 - left-turn (LT), 3/1 - right-turn (diff type)

            predictedMovement = model.predict_classes(localXTrainArray)

            predictedMovementValue = -1
            if(predictedMovement == 0):
                predictedMovementValue = 0
            elif(predictedMovement == 1):
                predictedMovementValue = 0.5
            elif(predictedMovement == 2):
                predictedMovementValue = 1
            else:
                print('Unknow movement data predicted!!!')
                print(predictedMovement)
                sys.exit()
            
            # Convert the predicted and ground truth values to string to write on visualize image
            groundTurthNextMovementString = movementValueToString(groundTurthNextMovement)
            predictedNextMovementString = movementValueToString(predictedMovementValue)

            # Append the results to list for confusion matrix calculation

            resultList.append((groundTurthNextMovementString,predictedNextMovementString))


            # Plot the global X/Y location on the global map and write the predicted and Ground truth values
            if(visual == 1):
                visualMap = mapImage.copy()
                color = (0,255,0)
                mapPixelX,mapPixelY = GlobalPoseToMapPixel(globalX, globalY)
                visualMap = cv2.circle(visualMap, (mapPixelX,mapPixelY), 12, color, -1)
                fontScale = 2
                thickness = 8
                font = cv2.FONT_HERSHEY_SIMPLEX 
                visualMap = cv2.putText(visualMap, groundTurthNextMovementString, (mapPixelX+40,mapPixelY+40), font, fontScale, color, thickness, cv2.LINE_AA)
                visualMap = cv2.putText(visualMap, predictedNextMovementString, (mapPixelX+80,mapPixelY+80), font, fontScale, color, thickness, cv2.LINE_AA)

                cv2.imshow('test', visualMap)
                cv2.waitKey(10)

                del visualMap

    # Calculate the confusion matrix/accuracy/precision
    confMatrix = CalcConfusionMatrix(resultList)
    straightAccuracy,leftTurnAccuracy,rightTurnAccuracy = classBasedAcc(confMatrix)
    straightPrecision,leftTurnPrecision,rightTurnPrecision = classBasedPrecision(confMatrix)

    print('Confusion Matrix .....')
    print(confMatrix)
    print('Class Based Accuracy .....')
    print(straightAccuracy,leftTurnAccuracy,rightTurnAccuracy)
    print('Class Based Precision .......')
    print(straightPrecision,leftTurnPrecision,rightTurnPrecision)
    print('All predicted Done!!!!!')


if __name__ == '__main__':

    mapFile = '/home/saptarshi/PythonCode/AdvanceLSTM/Maps/Lanekrshim.png'

    PredData(testTrajFilePath)
    # temporal = XTrain.shape[1]
    # features = XTrain.shape[2]

    # model = Sequential()
    # model.add(LSTM(256, activation='relu', return_sequences=True,input_shape=(temporal,features)))
    # model.add(LSTM(128, activation='relu'))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(3, activation='softmax'))

    # model.compile(loss='categorical_crossentropy', optimizer='adam')
    # model.summary()

    # model.fit(XTrain, YTrain, batch_size=128, epochs=5, verbose=1)

    # model.save('./models/basic.h5')

    print('All the cars are predicted in the scene.')

    sys.exit()







