#########################################################################################################################
# DO NOT CHANGE THE CODE, KEEP THE MODEL NAMES AND EVEYTHING EXCATLY SAME IN CASE WE WANT TO RTRAIN-RETEST CYCLE
# FEW MORE TIMES. THE MODEL NAMES AND EVERYTHING ELSE ARE IN THE SECOND LAST COMMENT BLOCK. IN CASE WANT TO RETRAIN-RETEST
# SIMPLY START THE CODE WITHOUT CHANGING ANYTHING EXCEPT THE RESULT OR LOSS FILE NAMES. KEEP THE MODEL NAMES SAME AS 
# VERSION "11" NOTHING ELSE AT ALL. THE FIRST TRAINING IS ALSREADY COMMENTED OUT. SO JUST START THE CODE WITH NEW LOSS AND ERROR FILE NAME. 
#########################################################################################################################
# More re-training loops.... 8
# batch 1024 step 7.0,20.0, 0.001, 0.002, 0.00001, 0.00001, ELU alpha 0.1, 4 surrouding, early 0.2, epochs 30-80 ->
# Result file -> ModifiedDecoderV8Run4.txt
# loss file name -> ModifiedDecoderV8LossRun4.txt'
# Train error went down as low as 3.8 but the val loss is almost 5.3 not good enough. Seems like an ideal case of overfitting. Will try with lower initial_lrate 
# for the next learinng rate. 
#########################################################################################################################
# Trying to fix why all ther GPUS are not used equally -> Fixed by specifying the GPU ID agaist each process
#########################################################################################################################
# batch 1024 step 7.0,20.0, 0.001, 0.001, 0.00001, 0.00001, ELU alpha 0.1, 4 surrouding, early 1.25, epochs 30-80 -> done..
# resultFileName = 'ModifiedDecoderV8Run32.txt'
# historyFileName = 'ModifiedDecoderV8LossRun32.txt'
# Both train and val is not going down.. maybe the second step initial_lrate should be 0.002 instead of 0.001
# also the jump during intermediate prediction was 15 instad of 10 to reduce time consumption, maybe this was the reason
##########################################################################################################################
# batch 1024 step 7.0,20.0, 0.001, 0.002, 0.00001, 0.00001, ELU alpha 0.1, 4 surrouding, early 1.25, epochs 30-80 -> imppresive, training error 2.0, val error...4.69 
# Check the resutl file to see the fall in training error and val error
# Changed the intermediate jump back to 10 from 15. 
# check each cycles loss.. clear case of overfitting......
# # # resultFileName = 'ModifiedDecoderV8Run33.txt'
# # # historyFileName = 'ModifiedDecoderV8LossRun33.txt'
# Check error and loss files to verify
# As batchNorm causing probelm keep retrain-test from the last stage -> Stopped..
# Done not change or Touch the code
# Keep the model name in case we have to keep retrain few more cycles..
# # # # encoderModelFilename = '/home/sap/Sap/Junction/ModV8Encoder11.h5'
# # # # decoderModelFilename = '/home/sap/Sap/Junction/ModV8Decoder11.h5'
# # # # mainModelFileName  = '/home/sap/Sap/Junction/ModV8MainModel11.h5'
# Keep the result and loss files till now in case..
# # # resultFileName = 'ModifiedDecoderV8Run50.txt'
# # # historyFileName = 'ModifiedDecoderV8LossRun50.txt'
###################################################################################################################
# batch 1024 step 7.0,20.0, 0.001, 0.002, 0.00001, 0.00001, ELU alpha 0.1, 4 surrouding, early 1.25, epochs 30-80 -> to be done not started due to batchNorm error. 
# From previous it is clear overfitting during the last rounds (loss from each round and error)..
# Added BatchNorm layer only 4-3-2 at each stage. just to see if it is fixing the obvious overfitting issue. 
# Let it run all the way end just to see.... BatchNorm may make things worse.. but worth try....
# Continued in the next version . ModifiedDecoderV9.py
##########################################################################################################################


import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import numpy as np
import pandas as pd
import sys
import math
import csv
from math import log, exp, tan, atan, pi, ceil, cos, sin
import random
import multiprocessing as mp
from multiprocessing import Process, Manager
import time
from time import sleep


# Specify the test trajectory csv file
# Path for local folder
# testTrajFilePath = '/home/saptarshi/PythonCode/Junction/data/juncSmall.csv'
# Path for Sen Server trajectory csv file
# testTrajFilePath = '/media/disk1/sap/Junction/data/Lankershim.csv'
# Path for Big Screen Server trajectory csv file
testTrajFilePath = '/home/sap/Sap/Junction/data/Lankershim.csv'
# Path for small Server trajectory csv file
# testTrajFilePath = '/home/sap/Junction/data/Lankershim.csv'

# Specify if process the data or read the processed data
#  'read' -> Read Data  'process' -> Process data
readStr = 'read'
processStr = 'process'
processOrRead = readStr

# Specify the folder name for the sample to read/write based on the above flag
# Path for local folder
# folderName = '/home/saptarshi/PythonCode/Junction/ModifyDecoderCheck'
# Path for Sen Server folder
# folderName ='/media/disk1/sap/Junction/ModifiedDecoder5Surrounding'
# Path for Big Screen Server folder    ######## old data -> ModifiedDecoder5Surrounding new data -> EligibilityCheckRemovedD1 ##############  EligibilityCheckRemoved5Surrouding
folderName = '/home/sap/Sap/Junction/EligibilityCheckRemovedD1'    # EligibilityCheckRemovedD1   smallTest
# Path for small Server folder
# folderName = '/home/sap/Junction/Surrounding6RelativeDecoderJuncDistEligibilityCheck'

# Sepcify the file paths for the encoder decoder model to save
# Path for local folder
# encoderModelFilename = '/home/saptarshi/PythonCode/Junction/Encoder.h5'
# decoderModelFilename = '/home/saptarshi/PythonCode/Junction/Decoder.h5'
# Path for Sen server folder
# encoderModelFilename = '/media/disk1/sap/Junction/Encoder.h5'
# decoderModelFilename = '/media/disk1/sap/Junction/Decoder.h5'
# Path for Big Screen Server folder
encoderModelFilename = '/home/sap/Sap/Junction/ModV8Encoder11.h5'
decoderModelFilename = '/home/sap/Sap/Junction/ModV8Decoder11.h5'
mainModelFileName  = '/home/sap/Sap/Junction/ModV8MainModel11.h5'


# Specify the result file to store each sample error last one (ModifiedDecoderV5Run15.txt, ModifiedDecoderV5LossRun1.txt)
resultFileName = 'ModifiedDecoderV8Run50.txt'
historyFileName = 'ModifiedDecoderV8LossRun50.txt'
f = open(resultFileName, 'x')
f.close()
h = open(historyFileName, 'x')
h.close()

# Specify the validation vehicle file name
validationFileName = folderName + '/' + 'validation.txt'

# Specify the training vehicle file name
trainingFileName = folderName + '/' + 'training.txt'

# Create the visible window
# cv2.namedWindow('test', cv2.WINDOW_NORMAL)

# Train and Validation process lists
manager = Manager()
trainProcessList = manager.list()
validationProcessList = manager.list()

# To keep count of number of sample processed
countList = manager.list()

# To count the manager list
errorManagerList = manager.list()
errorCountList = manager.list()

# # To keep the max relative X and Y for later use
# maxRelativeXY = manager.list()

# Model parametrs 
numberOfTrainingLoop = 10 
batchSize = 1024   #256  128
initialNumberEpochs = 30   #   30  #30
secondNumberEpochs = 80
historyTemporal = 30   #30
futureTemporal = 50   #50
surroudingCarCounts = 4    # 4
inputFeatureCount = 7  # 6 -> (poseX,poseY,velocity, LaneID, Movement, Direction, distance from junc)
globalInputFeatures = (surroudingCarCounts+1)*inputFeatureCount  
globalOutputFeatures = 6                          # 6 -> (poseX,poseY,velocity, Class0, Class1, Class2)
decoderFeatureCount = 7 # output + junc from dist (6+1=7)
globalDecoderFeatures = (surroudingCarCounts+1)*decoderFeatureCount 
leakyAlphaValue = 0.1      # 0.5
maximumAllowabelJuncDist = 250     #(250 Feet)
maximumSurroundingCarDist = 40     #(25 Feet)
predictionDistanceThreshold = 250  #(100 Feet )
ignoreFrameCount = 100
classOut = 3
poseOut = 2
velcoityOut = 1
n_units = 256

# Validation vehiles
totalVehileCount = 2000
validationVehicleCount = 300

# Min Max values for normalize or denormalize
minLocalY = 0
maxLocalY = 0
minLocalX = 0
maxLocalX = 0
minVel = 0
maxVel = 0
maxRealtiveX = -999
maxRealtiveY = -999

# Index of different features in the csv file
vechileIDIndex = 0
frameIDIndex = 1
totoalFrameIndex = 2
globalTimeIndex = 3
localXIndex = 4
localYIndex = 5
velocityIndex = 11
laneIDIndex = 13
originIndex = 14
destinationIndex = 15
intersectionIndex = 16
sectionIndex = 17 
directionIndex = 18
movementIndex = 19

# String Constants 
inputStr = 'Input'
decoderStr = 'Decoder'
trainStr = 'Train'
validationStr = 'Validation'

# Unit constants
feetToMeter = 0.3048

# csv feature count
csvFeatureCount = 24

# Make the frame dictionary global for use during prediction
dictByFrames = dict()
# Make the Vehicle dictionary global for use multi processing
dictByVehicles = dict()
# Make the mapper dict global
# Create Dictionary for Mapper
mapper = dict()
mapperDict = dict()

# other Create Dictionary for Mapper

# Define the junction location distances
juncLocDict = {
  "1.0": 65,
  "2.0": 430,
  "3.0": 1068,
  "4.0": 1560
}


# Custome Loss function
def euclidean_distance_loss(y_true, y_pred):
    from tensorflow.keras import backend as K
    """
    Euclidean distance loss
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


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

# Convert the Movement value to one hot encoded value as [1,0,0]
def MovementToClassForm(movementInfo):
    # Next movements are 0, 0.5, 1 due to normalization 1/0 - through (TH), 2/0.5 - left-turn (LT), 3/1 - right-turn.....
    returnClassInfo = []
    if(movementInfo == 0):
        returnClassInfo = [1,0,0]
    elif(movementInfo == 0.5):
        returnClassInfo = [0,1,0]
    elif(movementInfo == 1):
        returnClassInfo = [0,0,1]
    else:
        print('Unknow movement data!!!')
        sys.exit()
    return returnClassInfo

# Create the two dictionaries one based on FrameID and other based on VehicleID
def CreateVehicleAndFrameDict(loadFileName):

    print('Creating Vehicle and Frame based dictionary')

    loadFile = open(loadFileName, 'r')
    loadReader = csv.reader(loadFile)
    loadDataset = []
    for loadRow in loadReader:
        loadDataset.append(loadRow[0:csvFeatureCount])

    loadDataset.pop(0)
    sortedList = sorted(loadDataset, key=lambda x: (float(x[0]), float(x[1])))
    datasetArray = np.array(sortedList, dtype=np.float)

    # Close the load file
    loadFile.close()

    # Normalize each feature columns except localX and localY for relative movement
    normalizeIndexList = [velocityIndex,laneIDIndex,directionIndex,movementIndex]

    # Save the original min max value for further denormalization
    global minLocalX,maxLocalX,minLocalY,maxLocalY,minVel,maxVel

    minLocalX = min(datasetArray[:,localXIndex])
    maxLocalX = max(datasetArray[:,localXIndex])

    minLocalY = min(datasetArray[:,localYIndex])
    maxLocalY = max(datasetArray[:,localYIndex])

    minVel = min(datasetArray[:,velocityIndex])
    maxVel = max(datasetArray[:,velocityIndex])

    for eachIndex in normalizeIndexList:
        minVal = min(datasetArray[:,eachIndex])
        maxVal = max(datasetArray[:,eachIndex])
        datasetArray[:,eachIndex] = (datasetArray[:,eachIndex] - minVal)/(maxVal - minVal)

    # Create Dictionary with unique Times not Frames
    uniquFrameIds = list(np.unique(datasetArray[:,globalTimeIndex]))
    frameKeys = []
    for idx in range(0, len(uniquFrameIds)):
        frameKeys.append(str(uniquFrameIds[idx]))

    dictionaryByFrames = {key : list() for key in frameKeys}

    for jdx in range(0,len(datasetArray)):
        key = str(datasetArray[jdx,globalTimeIndex])
        dictionaryByFrames[key].append(datasetArray[jdx])

    # Create Dictionary with unique Vehicles
    uniquVehicleIds = list(np.unique(datasetArray[:,vechileIDIndex]))
    vehicleKeys = []
    for idx in range(0, len(uniquVehicleIds)):
        vehicleKeys.append(str(uniquVehicleIds[idx]))

    dictionaryByVehicles = {key : list() for key in vehicleKeys}

    for jdx in range(0,len(datasetArray)):
        key = str(datasetArray[jdx,vechileIDIndex])
        if len(dictionaryByVehicles[key])==0:
            dictionaryByVehicles[key].append(datasetArray[jdx])
            continue
        lastFrame = dictionaryByVehicles[key][-1][frameIDIndex]
        lastTime = dictionaryByVehicles[key][-1][globalTimeIndex]
        currentFrame = datasetArray[jdx][frameIDIndex]
        currentTime = datasetArray[jdx][globalTimeIndex]
        if(abs(currentFrame-lastFrame)==1 and abs(currentTime-lastTime)==100):
            dictionaryByVehicles[key].append(datasetArray[jdx])
        else:
            if key in mapper:
                updatedKey = mapper[key]
                lastFrame = dictionaryByVehicles[updatedKey][-1][frameIDIndex]
                lastTime = dictionaryByVehicles[updatedKey][-1][globalTimeIndex]
                currentFrame = datasetArray[jdx][frameIDIndex]
                currentTime = datasetArray[jdx][globalTimeIndex]
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

    return dictionaryByFrames,dictionaryByVehicles,mapper

# Calculate the Y location of the nearest junction and estimate the distance
def CalculateNearestJuncLoc(sectionID, intersectionID, poseX, poseY):

    # Initialize Y location and dist
    yLoc = -1
    juncDist = -1

    if(int(sectionID) == 0 and int(intersectionID) == 0):
        # print('unexpected section and intersection combination.........................')
        juncDist = 10  # quick fix for the time being
        return juncDist/maximumAllowabelJuncDist
        # sys.exit()

    # Check if the vehicle is at intersection if yes send the Y location of that intersection
    if(intersectionID!=0):
        yLoc = juncLocDict[str(intersectionID)]
        juncDist = abs(poseY-yLoc)
        if(juncDist > maximumAllowabelJuncDist):
            juncDist = maximumAllowabelJuncDist
        return juncDist/maximumAllowabelJuncDist

    # If the vehicle is at section 1 and left side return farthest possible distance keeping normalization in mind
    # If the vehicle is at section 5 and right side return farthest possible distance keeping normalization in mind
    if(poseX<=0 and sectionID==1.0):
        juncDist = maximumAllowabelJuncDist  # 250 feet is assumned based on a two standered junction separation in NGSIM data
        return juncDist/maximumAllowabelJuncDist
    if(poseX>0 and sectionID==5.0):
        juncDist = maximumAllowabelJuncDist  # 250 feet is assumned based on a two standered junction separation in NGSIM data
        return juncDist/maximumAllowabelJuncDist

    # if the vehicle is not at intersection identify which side of the road ans which section the car is
    # PoseX is negetive for vehicles on the left side
    if(poseX<=0):
        nearestIntersection = str(sectionID-1)
        # print('nearset intersection after -1 : ' + nearestIntersection)
        yLoc = juncLocDict[nearestIntersection]
        juncDist = abs(poseY-yLoc)
        if(juncDist > maximumAllowabelJuncDist):
            juncDist = maximumAllowabelJuncDist
        return juncDist/maximumAllowabelJuncDist
    
    # PoseX is positive for vehicles on the left side
    if(poseX>0):
        nearestIntersection = str(sectionID)
        yLoc = juncLocDict[nearestIntersection]
        juncDist = abs(poseY-yLoc)
        if(juncDist > maximumAllowabelJuncDist):
            juncDist = maximumAllowabelJuncDist
        return juncDist/maximumAllowabelJuncDist

    if(yLoc == -1):
        print('Junction distance calculation is not perfromed properly...')
        sys.exit()

# Pass the surrounding vechiles and current input list. It will extend the list with surrouding cars info.
def GetSurroundingCarsInfo(otherVechiles, tempInput, targetVehicleID, inputOrDecoder, localX, localY, initialX, initialY):

    # remove surrounding vehicles with 0 intersectionID and 0 sectionID #############################################3
    # popList = []
    # for odx,eachOtherVechiles in enumerate(otherVechiles):
    #     section = eachOtherVechiles[sectionIndex]
    #     intersection = eachOtherVechiles[intersectionIndex]
    #     if(section==0 and intersection==0):
    #         popList.append(odx)
    
    # for eachPopItem in popList:
    #     otherVechiles.pop(eachPopItem)

    otherVechilesCount = len(otherVechiles)

    # Atleast target vehicle shoudl be present in other vehicles
    if(otherVechilesCount < 1):
        print('Target vehicle not present in other vehicle')
        print('otherVechilesCount = ' + str(otherVechilesCount))
        sys.exit()

    # Target Vehicle should Present in Other vehicles
    otherVehicleArray = np.array(otherVechiles).reshape(otherVechilesCount,csvFeatureCount)
    otherIds = list(otherVehicleArray[:,vechileIDIndex])

    if(float(targetVehicleID) not in otherIds):
        print('Target Vehicle not is other list')
        print('Target Vehicle ID ' + targetVehicleID)
        print('Other IDs')
        print(otherIds)

    paddingCount = surroudingCarCounts + 1 - otherVechilesCount

    # Vehicle should be removed once due to the presence of target vehicle 
    removedFlag = 0

    # If other vehicle count is less than 5 (4 surronding + 1 target as it will present in the frame based list)
    # append all the vechiles info into input list. 
    if (otherVechilesCount <= (surroudingCarCounts+1)):
        # Process the gathered surrounding cars
        for eachOtherVechiles in otherVechiles:
            otherVehicleID = str(eachOtherVechiles[vechileIDIndex])
            # Ignore the target vechile as it is already added
            if(otherVehicleID == targetVehicleID):
                removedFlag = removedFlag + 1
                continue

            otherLocalX = eachOtherVechiles[localXIndex]
            otherLocalY = eachOtherVechiles[localYIndex]
            otherVelocity = eachOtherVechiles[velocityIndex]
            otherLaneID = eachOtherVechiles[laneIDIndex]
            otherDirection = eachOtherVechiles[directionIndex]
            otherMovement = eachOtherVechiles[movementIndex]
            otherRelativeX = abs(otherLocalX - initialX) # Added for the relative position
            otherRelativeY = abs(otherLocalY - initialY) # Added for the relative position
            # Nearest junction distance
            currentSection = eachOtherVechiles[sectionIndex]
            currentIntersection = eachOtherVechiles[intersectionIndex]
            juncDist = CalculateNearestJuncLoc(currentSection, currentIntersection, otherLocalX, otherLocalY)

            # If the other vehicle distance is more that allowable then append zeros
            otherDist = math.sqrt((((otherLocalX-localX)**2)+((otherLocalY-localY)**2)))

            if(inputOrDecoder == inputStr):
                if(otherDist>maximumSurroundingCarDist):
                    tempInput.extend([0,0,0,0,0,0,0])
                else:
                    tempInput.extend([otherRelativeX,otherRelativeY,otherVelocity,otherLaneID,otherDirection,otherMovement,juncDist])
            elif(inputOrDecoder == decoderStr):
                if(otherDist>maximumSurroundingCarDist):
                    tempInput.extend([0,0,0,0,0,0,0])
                else:
                    lastInputClassInfo = MovementToClassForm(otherMovement)
                    tempInput.extend([otherRelativeX,otherRelativeY,otherVelocity,lastInputClassInfo[0],lastInputClassInfo[1],lastInputClassInfo[2],juncDist])
            else:
                print('Unknown inputOrDecoder string : ' +  inputOrDecoder)
                sys.exit()
        
        # Append the zero padding based on the required padding count calculated using other vechile count and decided surrouning car count
        # As the number of input features and decoder input/output is same that is why we used the same zero padding width
        zeroList = []
        if(inputOrDecoder == inputStr):
            zeroList = [0,0,0,0,0,0,0]
        elif(inputOrDecoder == decoderStr):
            zeroList = [0,0,0,0,0,0,0]
        else:
            print('Unknown inputOrDecoder string : ' +  inputOrDecoder)
            sys.exit()

        for rdx in range(0,paddingCount):          
            tempInput.extend(zeroList)

    # Else the vechile count is more than 4. So select the nearest 4 vechicles.
    else:
        # Gather distance of each car from the target car
        otherCarIndexedDistanceList = []
        for sdx, eachOtherVechiles in enumerate(otherVechiles):
            otherVehicleID = str(eachOtherVechiles[vechileIDIndex])
            # Ignore the target vechile as the distance would be zero
            if(otherVehicleID == targetVehicleID):
                removedFlag = removedFlag + 1
                continue

            otherLocalX = eachOtherVechiles[localXIndex]
            otherLocalY = eachOtherVechiles[localYIndex]

            # Calculate distance of each other car append in the list with index value
            otherDist = math.sqrt((((otherLocalX-localX)**2)+((otherLocalY-localY)**2)))
            otherCarIndexedDistanceList.append([sdx,otherDist])
        
        # Sort the list based on distance and gather the lowest indexes
        otherCarIndexedDistanceList = sorted(otherCarIndexedDistanceList,key=lambda x: x[1])
        otherCarIndexedDistanceArray = np.array(otherCarIndexedDistanceList)
        releventOtherIndexes = otherCarIndexedDistanceArray[0:surroudingCarCounts,0:1]

        # Append other car infos to the temp input based on the decided index
        for eachReleventIndex in releventOtherIndexes:
            otherLocalX = otherVechiles[int(eachReleventIndex)][localXIndex]
            otherLocalY = otherVechiles[int(eachReleventIndex)][localYIndex]
            otherVelocity = otherVechiles[int(eachReleventIndex)][velocityIndex]
            otherLaneID = otherVechiles[int(eachReleventIndex)][laneIDIndex]
            otherDirection = otherVechiles[int(eachReleventIndex)][directionIndex]
            otherMovement = otherVechiles[int(eachReleventIndex)][movementIndex]
            otherRelativeX = abs(otherLocalX - initialX) # Added for the relative position
            otherRelativeY = abs(otherLocalY - initialY) # Added for the relative position
            # Nearest junction distance
            currentSection =  otherVechiles[int(eachReleventIndex)][sectionIndex]
            currentIntersection = otherVechiles[int(eachReleventIndex)][intersectionIndex]
            juncDist = CalculateNearestJuncLoc(currentSection, currentIntersection, otherLocalX, otherLocalY)

            # If the other vehicle distance is more than allowable then append zeros
            otherDist = math.sqrt((((otherLocalX-localX)**2)+((otherLocalY-localY)**2)))
            
            if(inputOrDecoder == inputStr):
                if(otherDist>maximumSurroundingCarDist):
                    tempInput.extend([0,0,0,0,0,0,0])
                else:
                    tempInput.extend([otherRelativeX,otherRelativeY,otherVelocity,otherLaneID,otherDirection,otherMovement,juncDist])
            elif(inputOrDecoder == decoderStr):
                if(otherDist>maximumSurroundingCarDist):
                    tempInput.extend([0,0,0,0,0,0,0])
                else:
                    lastInputClassInfo = MovementToClassForm(otherMovement)
                    tempInput.extend([otherRelativeX,otherRelativeY,otherVelocity,lastInputClassInfo[0],lastInputClassInfo[1],lastInputClassInfo[2],juncDist])

    if(removedFlag != 1):
        print('Vehicle not removed propoerly for vehicle ID ' + str(targetVehicleID))
        print('removedFlag = ' + str(removedFlag))


    if(inputOrDecoder == inputStr):
        if(len(tempInput) != globalInputFeatures):
            print('Unwanted input feature in GetSurroundingCarsInfo is : ' + str(len(tempInput)))
    elif(inputOrDecoder == decoderStr):
        if(len(tempInput) != globalDecoderFeatures):
            print('Unwanted decoder feature in GetSurroundingCarsInfo is : ' + str(len(tempInput)))
    else:
        print('Unknonw inputOrDecoder string : ' + str(inputOrDecoder))

    return tempInput

def ProcessByVehicle(processItme):

    currentID = processItme[0]   # Updated key string
    currentTrainOrValStr = processItme[1] # strig
    targetVehicleID = processItme[2]   # original key string

    if(targetVehicleID == None):
        print('Traget Vehicle ID is none!!!!')

    currentVehicleList = dictByVehicles[currentID]

    # Add the check for the side origins and side destination
    sideOrigin = currentVehicleList[0][originIndex]
    sideDestination = currentVehicleList[0][destinationIndex]
    if ((sideOrigin == 111 and sideDestination == 203) or (sideOrigin == 103 and sideDestination == 211) or (sideOrigin == 105 and sideDestination == 210) or (sideOrigin == 110 and sideDestination == 205) or (sideOrigin == 107 and sideDestination == 209) or (sideOrigin == 109 and sideDestination == 207)):
        return

    # and straight to straight vehicles
    if ((sideOrigin == 101 and sideDestination == 208) or (sideOrigin == 108 and sideDestination == 201)):
        return

    # Add check of the section and intersection both IDs are zero. Not expected behaviour
    initialSectionID = currentVehicleList[0][sectionIndex]
    initialIntersectionID = currentVehicleList[0][intersectionIndex]
    if(initialSectionID == 0 and initialIntersectionID == 0):
        return

    currentVehicleLength = len(currentVehicleList)

    for idx in range(historyTemporal,currentVehicleLength-futureTemporal):

        # Get the current vehicles as those are only eligible from prediction point of view 
        # vehicles appearing in first frame of the target vehicle
        currentTargetTime = currentVehicleList[idx-historyTemporal][globalTimeIndex]
        currentOtherVehicles = dictByFrames[str(currentTargetTime)]
        currentOtherEligibleVehicles = []
        for eachCurrentOtherVehicles in currentOtherVehicles:
            currentOtherID = eachCurrentOtherVehicles[vechileIDIndex]
            # # Update other Id in case it is present in mapper
            # if(str(currentOtherID) in mapper):
            #     updatedID = mapper[str(currentOtherID)]
            #     currentOtherID = updatedID
            if(currentOtherID == float(targetVehicleID)):
                currentOtherEligibleVehicles.append(currentOtherID)
                continue
            # vehicles having history + future temporal frames.
            currentOtherFrame = eachCurrentOtherVehicles[frameIDIndex]
            currentOtherTotalFrame = eachCurrentOtherVehicles[totoalFrameIndex]
            remainingFrames = currentOtherTotalFrame - currentOtherFrame
            if(remainingFrames>= historyTemporal+futureTemporal):
                currentOtherEligibleVehicles.append(currentOtherID)

        # Prepeare sequential Input Data
        localXData = []
        initialLocalX = currentVehicleList[idx-historyTemporal][localXIndex]
        initialLocalY = currentVehicleList[idx-historyTemporal][localYIndex]
        for jdx in range(idx-historyTemporal,idx):
            tempInput = []
            absoluteX = currentVehicleList[jdx][localXIndex]
            absoluteY = currentVehicleList[jdx][localYIndex]
            localX = abs(absoluteX - initialLocalX)
            localY = abs(absoluteY - initialLocalY)
            velocity = currentVehicleList[jdx][velocityIndex]
            laneID = currentVehicleList[jdx][laneIDIndex]
            direction = currentVehicleList[jdx][directionIndex]
            movement = currentVehicleList[jdx][movementIndex]

            # Nearest junction distance
            currentSection = currentVehicleList[jdx][sectionIndex]
            currentIntersection = currentVehicleList[jdx][intersectionIndex]
            juncDist = CalculateNearestJuncLoc(currentSection, currentIntersection, absoluteX, absoluteY)

            tempInput = [localX,localY,velocity,laneID,direction,movement,juncDist]

            # Prepare the surrounding cars information
            # Gather vehicles using the same frame using the Frame Dict
            currentInputFrame = currentVehicleList[jdx][frameIDIndex]
            currentInputTime = currentVehicleList[jdx][globalTimeIndex]
            otherVechiles = dictByFrames[str(currentInputTime)]

            # Remove the prediction not eligible vehicles
            eligibleOtherVehicles = []
            for eachOtherVehicle in otherVechiles:
                otherID = eachOtherVehicle[vechileIDIndex]
                if (otherID in currentOtherEligibleVehicles):
                    eligibleOtherVehicles.append(eachOtherVehicle)

            # Remove vehicles with a different global time which is not possible. Just adding check to be sure
            for fdx,eachOtherTime in enumerate(otherVechiles):
                otherTime = eachOtherTime[globalTimeIndex]
                if (otherTime != currentInputTime):
                    print('Mismatch in input global time..')
                    print('other Time ' + str(otherTime))
                    print('Current Time ' + str(currentInputTime))
                    sys.exit()

            # Extend the surrounding cars info into the target vehicles input   otherVechiles replaced by  eligibleOtherVehicles
            tempInput = GetSurroundingCarsInfo(eligibleOtherVehicles, tempInput, targetVehicleID, inputStr, absoluteX, absoluteY, initialLocalX, initialLocalY)

            if (len(tempInput) != globalInputFeatures):
                print('tempInput len is : ' + str(len(tempInput)) + ' instead of ' + str(globalInputFeatures))
                sys.exit()

            # Add the final list of target vehicles and other vehicles info into the local input
            localXData.append(tempInput)


        # Prepeare sequential Output Data and decoder input data
        localYMovementData = []
        localYVelData = []
        localYPoseData = []
        decoderInputData = []

        # Prepare the First Decoder Input
        lastInput = localXData[-1]
        firstDecoderInput = []
        for tdx in range(0,len(lastInput),inputFeatureCount):
            lastInputPoseX = lastInput[tdx]
            lastInputPoseY = lastInput[tdx+1]
            lastInputVelocity = lastInput[tdx+2]
            lastInputMovement = lastInput[tdx+5]
            lastInputClassInfo = MovementToClassForm(lastInputMovement)
            # Calculate the distance from the junction for the first decoder input 
            # For section, intersection, absoluteX and absoluteY use the last updated varibale as they hold the info for the last frame.
            juncDist = CalculateNearestJuncLoc(currentSection, currentIntersection, absoluteX, absoluteY)
            firstDecoderInput.extend([lastInputPoseX,lastInputPoseY,lastInputVelocity,lastInputClassInfo[0],lastInputClassInfo[1],lastInputClassInfo[2],juncDist])


        for kdx in range(idx,idx+futureTemporal):

            # Add the ground truth outputs
            nextMovement = currentVehicleList[kdx][movementIndex]
            nextMovementClassData = MovementToClassForm(nextMovement)

            localYMovementData.append(nextMovementClassData)
            
            nextVelocity = currentVehicleList[kdx][velocityIndex]
            deNormalizedNextVelocity = (nextVelocity*(maxVel-minVel))+minVel

            nextLocalX = currentVehicleList[kdx][localXIndex]
            nextRelativeX = abs(nextLocalX - initialLocalX)
            # denormalizedNextLocalX = (nextLocalX*(maxLocalX-minLocalX)+minLocalX) # no need to denormalize as it is not normalized
            nextLocalY = currentVehicleList[kdx][localYIndex]
            nextRelativeY = abs(nextLocalY - initialLocalY)
            # denormalizedNextLocalY = (nextLocalY*(maxLocalY-minLocalY)+minLocalY) # no need to denormalize as it is not normalized

            localYVelData.append([deNormalizedNextVelocity])
            localYPoseData.append([nextRelativeX,nextRelativeY])

            # Add the decoder input
            # Add the distance from the junc in the decoder as well
            nextSection = currentVehicleList[kdx][sectionIndex]
            nextIntersection = currentVehicleList[kdx][intersectionIndex]
            juncDist = CalculateNearestJuncLoc(nextSection, nextIntersection, nextLocalX, nextLocalY)

            decoderTemp = [nextRelativeX,nextRelativeY,nextVelocity,nextMovementClassData[0],nextMovementClassData[1],nextMovementClassData[2],juncDist]

            # Prepare the surrounding cars information for decoder input   # for decoder pass only the vehicles present in the last 30 frames..(not done....)
            # Gather vehicles using the same frame using the Frame Dict
            currentInputFrame = currentVehicleList[kdx][frameIDIndex]
            currentInputTime = currentVehicleList[kdx][globalTimeIndex]
            otherVechiles = dictByFrames[str(currentInputTime)]

            # Remove the prediction not eligible vehicles
            # Identify vehicles not in eligible list
            eligibleOtherVehicles = []
            for eachOtherVehicle in otherVechiles:
                otherID = eachOtherVehicle[vechileIDIndex]
                if (otherID in currentOtherEligibleVehicles):
                    eligibleOtherVehicles.append(eachOtherVehicle)

            # Remove vehicles with a different global time. Which is not possible. Just to double check
            for gdx,eachOtherTime in enumerate(otherVechiles):
                otherTime = eachOtherTime[globalTimeIndex]
                if (otherTime != currentInputTime):
                    print('Mismatch in decoder global time..')
                    print('other Time ' + str(otherTime))
                    print('Current Time ' + str(currentInputTime))
                    sys.exit()


            # Extend the surrounding cars info into the target vehicles decoder input   ##  otherVechiles replacd by eligibleOtherVehicles 
            decoderTemp = GetSurroundingCarsInfo(eligibleOtherVehicles, decoderTemp, targetVehicleID, decoderStr, nextLocalX, nextLocalY, initialLocalX, initialLocalY)

            # Check the decoder feature length
            if (len(decoderTemp) != globalDecoderFeatures):
                print('decoderTemp len is : ' + str(len(decoderTemp)) + ' instead of ' + str(globalDecoderFeatures))
                sys.exit()

            # Finally append the target car and surrounding cars info for the current frame into the final decoded input
            decoderInputData.append(decoderTemp)


        # Append in the final validation or training set based on decided vehicle ID
        if(currentTrainOrValStr == 'Validation'):
            # Shift one time stamp right and append Last input at the beggining 
            decoderInputData = decoderInputData[:-1]
            decoderInputData.insert(0,firstDecoderInput)

            validationProcessList.append([localXData,decoderInputData,localYMovementData,localYVelData,localYPoseData])
        elif(currentTrainOrValStr == 'Train'):
            # Shift one time stamp right and append last input at the beggining 
            decoderInputData = decoderInputData[:-1]
            decoderInputData.insert(0,firstDecoderInput)

            trainProcessList.append([localXData,decoderInputData,localYMovementData,localYVelData,localYPoseData])
        else:
            print('Unknown Train Val string')
            sys.exit()
        

        if((np.array(localXData).shape[0] != historyTemporal) or (np.array(localXData).shape[1] != globalInputFeatures)):
            print('localXData/Input Array Shape : ')
            print(np.array(localXData).shape)
            sys.exit()
        
        if((np.array(decoderInputData).shape[0] != futureTemporal) or (np.array(decoderInputData).shape[1] != globalDecoderFeatures)):
            print('decoderInputData Array Shape : ')
            print(np.array(decoderInputData).shape)
            sys.exit() 


    countList.append(0)
    totalSamplesProcessed = len(countList)
    print('Finished Processing Sample : ' + str(totalSamplesProcessed))

# Write the processed data to a file
def WriteToFile(writeFileName,samples):

    fsample = open(writeFileName, 'x')
    for eachSample in samples:
        for eachTemporal in eachSample:
            fsample.write("%s\n" % eachTemporal)
    fsample.close()

# Read the processed data from a folder
def ReadFromFile(readFileName, temporal):

    readFilePath = folderName + '/' + readFileName + '.txt'
    readFile = open(readFilePath, "r")
    loadedData = readFile.readlines()

    # Close the read file
    readFile.close()

    dataList = []

    # Enumarate temporal shift jump to get loadeddata [0:temporal] = sample
    totalLines = len(loadedData)
    for idx in range(0,totalLines,temporal):
        loadedSample = loadedData[idx:idx+temporal]
        sampleList = []
        for eachLoadedSample in loadedSample:
            currentSample = eachLoadedSample[1:-2].split(',')   
            currentSampleFloat = [float(i) for i in currentSample]
            sampleList.append(currentSampleFloat) 
        dataList.append(sampleList)

    loadedDataArray = np.array(dataList)
    print(readFileName + ' Array Shape : ' + str(loadedDataArray.shape))

    return loadedDataArray

# Plot all cars trajectory on the global GPS map
def TrainData(inputFileName):

    # Load the Vehicle and Frame based Dictionaries
    global dictByFrames, dictByVehicles, validationVehicles, mapper, trainProcessList, validationProcessList
    dictByFrames,dictByVehicles,unusedMapperDict = CreateVehicleAndFrameDict(inputFileName)
    finalVehicleKeys = list(dictByVehicles.keys())
    finalVehicleKeys.sort(key=float)
    finalFrameKeys = list(dictByFrames.keys())
    finalFrameKeys.sort(key=float)

    # Pin the new process to specified cores
    os.system("taskset -p -c 1,2,3,4,5,6,7,8,9,10,11 %d" % os.getpid()) 
    # os.system("taskset -p -c 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 %d" % os.getpid())
    processes = []
    numberofCores = 10

    pool = mp.Pool(numberofCores)

    print('Total Vehicle : ' + str(len(finalVehicleKeys)))

    selectedVehilces = random.sample(finalVehicleKeys,totalVehileCount)

    # Randomly sample 300 validation vehicles from selected vehicles
    validationList = random.sample(selectedVehilces,validationVehicleCount)

    # Write the validation vehicles to the data folder
    validationFileObj = open(validationFileName, 'x')
    for eachValidationCar in validationList:
        validationFileObj.write("%s\n" % eachValidationCar)
    validationFileObj.close()


    processList = []

    for eachKey in selectedVehilces:
        # If the value exists in mapper get the key as original ID of later surrounding car info target vehicle seperation   mydict.keys()[mydict.values().index(16)]
        originalID = None
        if(eachKey in unusedMapperDict.values()):
            valueList = list(unusedMapperDict.values())
            originalID = list(unusedMapperDict.keys())[valueList.index(eachKey)]
            # print('Original Key for ' + eachKey + ' is found and the Key is ' + str(originalID))
        else:
            originalID = eachKey

        processStr = ''
        if eachKey in validationList:
            processStr = validationStr
        else:
            processStr = trainStr
        processList.append([eachKey,processStr,originalID])

    pool.map(ProcessByVehicle,processList)
    # for eachItem in processList:
    #     ProcessByVehicle(eachItem)

    
    # Convert the Train manager list to normal list
    print('Converting the Train Manager list to normal lists.....')
    normalTrainList = list(trainProcessList)
    print('List converted!!!')

    # Prepare the final lists of train and validation data
    # Train final lists
    print('Prepering the individual lists')

    finalXTrain = [x[0] for x in normalTrainList]
    filePath = folderName + '/finalXTrain.txt'
    WriteToFile(filePath,finalXTrain)
    print('Finished XTrain Array!!!')

    finalTrainDecoderInput = [x[1] for x in normalTrainList]
    filePath = folderName + '/finalTrainDecoderInput.txt'
    WriteToFile(filePath,finalTrainDecoderInput)
    print('Finished decoderTrainInput Array!!!')

    finalYClassTrain = [x[2] for x in normalTrainList]
    filePath = folderName + '/finalYClassTrain.txt'
    WriteToFile(filePath,finalYClassTrain)
    print('Finished YClassTrain Array!!!')

    finalYVelTrain = [x[3] for x in normalTrainList]
    filePath = folderName + '/finalYVelTrain.txt'
    WriteToFile(filePath,finalYVelTrain)
    print('Finished finalYVelTrain Array!!!')

    finalYPoseTrain = [x[4] for x in normalTrainList]
    filePath = folderName + '/finalYPoseTrain.txt'
    WriteToFile(filePath,finalYPoseTrain)
    print('Finished finalYPoseTrain Array!!!')

    # Convert the Validation manager list to normal list
    print('Converting the Validation Manager list to normal lists.....')
    normalValList = list(validationProcessList)
    print('List converted!!!')

    # Validation final lists
    finalXVal = [x[0] for x in normalValList]
    filePath = folderName + '/finalXVal.txt'
    WriteToFile(filePath,finalXVal)
    print('Finished XVal Array!!!')

    finalValDecoderInput = [x[1] for x in normalValList]
    filePath = folderName + '/finalValDecoderInput.txt'
    WriteToFile(filePath,finalValDecoderInput)
    print('Finished decoderValInput Array!!!')

    finalYClassVal = [x[2] for x in normalValList]
    filePath = folderName + '/finalYClassVal.txt'
    WriteToFile(filePath,finalYClassVal)
    print('Finished YClassVal Array!!!')

    finalYVelVal = [x[3] for x in normalValList]
    filePath = folderName + '/finalYVelVal.txt'
    WriteToFile(filePath,finalYVelVal)
    print('Finished YVelVal Array!!!')

    finalYPoseVal = [x[4] for x in normalValList]
    filePath = folderName + '/finalYPoseVal.txt'
    WriteToFile(filePath,finalYPoseVal)
    print('Finished YPoseVal Array!!!')

    print('Finished All Array!!!')

    # Prepare the final Train arrays
    XTrain = np.array(finalXTrain)
    decoderTrainInput = np.array(finalTrainDecoderInput)
    YClassTrain = np.array(finalYClassTrain)
    YPoseTrain = np.array(finalYPoseTrain)
    YVelTrain = np.array(finalYVelTrain)

    # Prepare the final Validation arrays
    XVal = np.array(finalXVal)
    decoderValInput = np.array(finalValDecoderInput)
    YClassVal = np.array(finalYClassVal)
    YVelVal = np.array(finalYVelVal)
    YPoseVal = np.array(finalYPoseVal)

    # Print the shape of the Arrays
    filePath = folderName + '/arrayShapes.txt'
    with open(filePath, 'x') as fshape:
        fshape.write('XTrain shape : ' + str(XTrain.shape) + '\n')
        fshape.write('decoderTrainInput shape : ' + str(decoderTrainInput.shape) + '\n')
        fshape.write('YClassTrain shape : ' + str(YClassTrain.shape) + '\n')
        fshape.write('YPoseTrain shape : ' + str(YPoseTrain.shape) + '\n')
        fshape.write('YVelTrain shape : ' + str(YVelTrain.shape) + '\n')

        fshape.write('XVal shape : ' + str(XVal.shape) + '\n')
        fshape.write('decoderValInput shape : ' + str(decoderValInput.shape) + '\n')
        fshape.write('YClassVal shape : ' + str(YClassVal.shape) + '\n')
        fshape.write('YVelVal shape : ' + str(YVelVal.shape) + '\n')
        fshape.write('YPoseVal shape : ' + str(YPoseVal.shape) + '\n')

    return XTrain,decoderTrainInput,YClassTrain,YVelTrain,YPoseTrain,XVal,decoderValInput,YClassVal,YVelVal,YPoseVal

# Test the trained model just on the validation data
def ValidationSetPredict(encoder_model,decoder_model,XVal,decoderValInput,YPoseVal, maxRealtiveX, maxRealtiveY):

    print('Predicting the validation dataset...')

    validationSampleLenght = len(XVal)

    # Intialize the frame based distance error array with sample count as 0
    finalError = np.zeros(futureTemporal)
    count = 0

    # Predict sequence
    for pdx,eachXVal in enumerate(XVal):
        currentPredictInput = np.array(eachXVal).reshape(1,historyTemporal,globalInputFeatures)
        groundTruthPose = YPoseVal[pdx]
        decoderLocal = decoderValInput[pdx]

        state = encoder_model.predict(currentPredictInput)

        # First decoderVal entry is the first target sequence
        predDecoderInput = decoderLocal[0]

        target_seq = np.array(predDecoderInput).reshape(1,1,globalDecoderFeatures)

        outputPose = []

        # Calculate the euclidian error
        currentError = []

        # Perfrom the sequential prediction
        for t in range(futureTemporal):
            # predict next Features
            classPred, velcoityPred, posePred, h1, c1, h2, c2 = decoder_model.predict([target_seq] + state)

            # store prediction
            outputPose.append([posePred[0][0][0],posePred[0][0][1]])

            # Normalize the predicted velocity for next instance prediction
            normalizedPredVelocity = (velcoityPred[0][0][0]-minVel)/(maxVel-minVel)

            # Normalize the predicted local poses for next instance prediction
            normalizedPredPoseX = posePred[0][0][0]/maxRealtiveX
            normalizedPredPoseY = posePred[0][0][1]/maxRealtiveY

            # update state
            state = [h1, c1, h2, c2]

            # update target sequence
            # Update the target sequence till second last frame. At the last frame no need to update the seq as it will not be used
            if(t<(futureTemporal-1)):
                targetDecoder = [normalizedPredPoseX,normalizedPredPoseY,normalizedPredVelocity,classPred[0][0][0],classPred[0][0][1],classPred[0][0][2]]
                surroundingDecoder = decoderLocal[t+1][decoderFeatureCount-1:] # 6 -> target vehicle features (-1 is beacuse to get the target vehicle distance from junc as it is not possible to calculate here due to absence of absolute/initial distance)
                targetDecoder.extend(surroundingDecoder)
                target_seq = np.array(targetDecoder).reshape(1,1,globalDecoderFeatures)

            # Calculate the Euclidian Error
            truePoseX = groundTruthPose[t][0]
            truePoseY = groundTruthPose[t][1]

            predPoseX = outputPose[t][0]
            predPoseY = outputPose[t][1]

            euclidianError = math.sqrt(((truePoseX-predPoseX)**2) + ((truePoseY-predPoseY)**2))
            euclidianErrorMeter = euclidianError*0.3048

            currentError.append(euclidianErrorMeter)
        
        # Keep count for average calculation and display average error
        count = count + 1
        finalError = finalError + np.array(currentError)

        # Print in the same line 
        printList = np.around(np.array([finalError[0],finalError[4],finalError[9],finalError[14],finalError[19],finalError[24],finalError[29],finalError[34],finalError[39],finalError[44],finalError[49]])/count, 2)
        print('Current Frame : ' + str(pdx) + '/' + str(validationSampleLenght) + ' ', end=' ')
        print(*printList, end='\r', flush=True)
    
    print('All the validation vehicles are predicted...')

    print('Final Validation Distance Error')
    print(finalError/count)
    print('All the Validation cars are predcited in the dataset.')


# Define the Custome learing rate decays
def step_decay(epoch):
    initial_lrate = 0.001      # 0.001 for RMSProp and Adma 0.002 for Nadam
    drop = 0.5
    epochs_drop = 7.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    if lrate < 0.00001:
        lrate = 0.00001
    return lrate

# Define the Custome learing rate decays
def second_step_decay(epoch):
    initial_lrate = 0.002      # 0.001 for RMSProp and Adma 0.002 for Nadam
    drop = 0.5
    epochs_drop = 20.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    if lrate < 0.00001:
        lrate = 0.00001
    return lrate


# Create the model architecture 
def ModelArch():

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import  Input,LSTM,Dense,Concatenate, subtract, Lambda, BatchNormalization, LeakyReLU, ELU
    from tensorflow.keras.optimizers import RMSprop, Adam, Nadam
    from tensorflow.keras.losses import logcosh
    from tensorflow.keras import backend as K

    # Stacked LSTM encoder decoder with connected output    
    # define training encoder
    encoder_inputs = Input(shape=(None, globalInputFeatures))
    # First Encoder LSTM Layer
    encoder1 = LSTM(n_units, return_state=True, return_sequences=True)
    encoder_output, state_h1, state_c1 = encoder1(encoder_inputs)
    encoder_states1 = [state_h1, state_c1]
    # Second Encoder LSTM Layer
    encoder2 = LSTM(n_units, return_state=True)
    encoder_output, state_h2, state_c2 = encoder2(encoder_output)
    encoder_states2 = [state_h2, state_c2]
	# define training decoder
    decoder_inputs = Input(shape=(None, globalDecoderFeatures))
    # First Decoder LSTM Layer
    decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm1(decoder_inputs, initial_state=encoder_states1)
    # Second Decoder LSTM Layer
    decoder_lstm2 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm2(decoder_outputs, initial_state=encoder_states2)

    encoder_states = [state_h1, state_c1, state_h2, state_c2]

    # Common batchNorm
    # commonBatchNorm = BatchNormalization()
    # decoder_outputs = commonBatchNorm(decoder_outputs)

    # Decoder for ClassOut        
    # batchNorm1 = BatchNormalization()
    decoder_dense10a = Dense(1024)
    decoder_Leaky10a = ELU(alpha=leakyAlphaValue)
    decoder_output1 = decoder_dense10a(decoder_outputs)
    # decoder_output1 = batchNorm1(decoder_output1)
    decoder_output1 = decoder_Leaky10a(decoder_output1)
    # batchNorm2 = BatchNormalization()
    decoder_dense10 = Dense(512)
    decoder_Leaky10 = ELU(alpha=leakyAlphaValue)
    decoder_output1 = decoder_dense10(decoder_output1)
    # decoder_output1 = batchNorm2(decoder_output1)
    decoder_output1 = decoder_Leaky10(decoder_output1)
    # batchNorm3 = BatchNormalization()
    decoder_dense11 = Dense(256)
    decoder_Leaky11 = ELU(alpha=leakyAlphaValue)
    decoder_output1 = decoder_dense11(decoder_output1)
    # decoder_output1 = batchNorm3(decoder_output1)
    decoder_output1 = decoder_Leaky11(decoder_output1)
    # batchNorm4 = BatchNormalization()    
    decoder_dense12 = Dense(128)
    decoder_Leaky12 = ELU(alpha=leakyAlphaValue)
    decoder_output1 = decoder_dense12(decoder_output1)
    # decoder_output1 = batchNorm4(decoder_output1)
    decoder_output1 = decoder_Leaky12(decoder_output1)
    # batchNorm5 = BatchNormalization()
    decoder_dense13 = Dense(64)
    decoder_Leaky13 = ELU(alpha=leakyAlphaValue)
    decoder_output1 = decoder_dense13(decoder_output1)
    # decoder_output1 = batchNorm5(decoder_output1)
    decoder_output1 = decoder_Leaky13(decoder_output1)
    # batchNorm6 = BatchNormalization()    
    decoder_dense14 = Dense(32)
    decoder_Leaky14 = ELU(alpha=leakyAlphaValue)
    decoder_output1 = decoder_dense14(decoder_output1)
    # decoder_output1 = batchNorm6(decoder_output1)
    decoder_output1 = decoder_Leaky14(decoder_output1)
    decoder_dense15 = Dense(3, activation='softmax', name='Class')
    classOut = decoder_dense15(decoder_output1)

    # Decoder for Velocity Out
    decoder2_concat = Concatenate()
    decoder_output2 = decoder2_concat([decoder_outputs,classOut])
    # batchNorm7 = BatchNormalization()    
    decoder_dense20a = Dense(1024)
    decoder_Leaky20a = ELU(alpha=leakyAlphaValue)
    decoder_output2 = decoder_dense20a(decoder_output2)
    # decoder_output2 = batchNorm7(decoder_output2)
    decoder_output2 = decoder_Leaky20a(decoder_output2)
    # batchNorm8 = BatchNormalization()    
    decoder_dense20 = Dense(512)
    decoder_Leaky20 = ELU(alpha=leakyAlphaValue)
    decoder_output2 = decoder_dense20(decoder_output2)
    # decoder_output2 = batchNorm8(decoder_output2)
    decoder_output2 = decoder_Leaky20(decoder_output2)
    # batchNorm9 = BatchNormalization()
    decoder_dense21 = Dense(256)
    decoder_Leaky21 = ELU(alpha=leakyAlphaValue)
    decoder_output2 = decoder_dense21(decoder_output2)
    # decoder_output2 = batchNorm9(decoder_output2)
    decoder_output2 = decoder_Leaky21(decoder_output2)
    # batchNorm10 = BatchNormalization()    
    decoder_dense22 = Dense(128)
    decoder_Leaky22 = ELU(alpha=leakyAlphaValue)
    decoder_output2 = decoder_dense22(decoder_output2)
    # decoder_output2 = batchNorm10(decoder_output2)
    decoder_output2 = decoder_Leaky22(decoder_output2)
    # batchNorm11 = BatchNormalization()
    decoder_dense23 = Dense(64)
    decoder_Leaky23 = ELU(alpha=leakyAlphaValue)
    decoder_output2 = decoder_dense23(decoder_output2)
    # decoder_output2 = batchNorm11(decoder_output2)
    decoder_output2 = decoder_Leaky23(decoder_output2)
    # batchNorm12 = BatchNormalization()    
    decoder_dense24 = Dense(32)
    decoder_Leaky24 = ELU(alpha=leakyAlphaValue)
    decoder_output2 = decoder_dense24(decoder_output2)
    # decoder_output2 = batchNorm12(decoder_output2)
    decoder_output2 = decoder_Leaky24(decoder_output2)
    decoder_dense25 = Dense(1, activation='linear', name='Velcoity')
    velocityOut = decoder_dense25(decoder_output2)

    # Normalize output velocity to before concatenate
    minVelConst = K.constant(value=minVel, dtype='float32')
    minMaxVelDiffConst = K.constant(value=(maxVel-minVel), dtype='float32')

    velocityNormalized = Lambda(lambda x: (x-minVelConst)/minMaxVelDiffConst)
    velocityConcat = velocityNormalized(velocityOut)

    # Decoder for position out
    decoder3_concat = Concatenate()
    decoder_output3 = decoder3_concat([decoder_outputs,classOut,velocityConcat])
    # batchNorm13 = BatchNormalization()
    decoder_dense30b = Dense(2048)
    decoder_Leaky30b = ELU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense30b(decoder_output3)
    # decoder_output3 = batchNorm13(decoder_output3)
    decoder_output3 = decoder_Leaky30b(decoder_output3)
    # batchNorm14 = BatchNormalization()    
    decoder_dense30a = Dense(1024)
    decoder_Leaky30a = ELU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense30a(decoder_output3)
    # decoder_output3 = batchNorm14(decoder_output3)
    decoder_output3 = decoder_Leaky30a(decoder_output3)
    # batchNorm15 = BatchNormalization()
    decoder_dense30 = Dense(512)
    decoder_Leaky30 = ELU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense30(decoder_output3)
    # decoder_output3 = batchNorm15(decoder_output3)
    decoder_output3 = decoder_Leaky30(decoder_output3)
    # batchNorm16 = BatchNormalization()    
    decoder_dense31 = Dense(256)
    decoder_Leaky31 = ELU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense31(decoder_output3)
    # decoder_output3 = batchNorm16(decoder_output3)
    decoder_output3 = decoder_Leaky31(decoder_output3)
    # batchNorm17 = BatchNormalization()    
    decoder_dense32 = Dense(128)
    decoder_Leaky32 = ELU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense32(decoder_output3)
    # decoder_output3 = batchNorm17(decoder_output3)
    decoder_output3 = decoder_Leaky32(decoder_output3)
    # batchNorm18 = BatchNormalization()    
    decoder_dense33 = Dense(64)
    decoder_Leaky33 = ELU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense33(decoder_output3)
    # decoder_output3 = batchNorm18(decoder_output3)
    decoder_output3 = decoder_Leaky33(decoder_output3)
    # batchNorm19 = BatchNormalization()    
    decoder_dense34 = Dense(32)
    decoder_Leaky34 = ELU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense34(decoder_output3)
    # decoder_output3 = batchNorm19(decoder_output3)
    decoder_output3 = decoder_Leaky34(decoder_output3)
    decoder_dense35 = Dense(2, activation='linear', name='Position')
    positionOut = decoder_dense35(decoder_output3)
    
    model = Model([encoder_inputs, decoder_inputs], [classOut, velocityOut, positionOut])

	# define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)

	# define inference decoder
    decoder_state_input_h1 = Input(shape=(n_units,))
    decoder_state_input_c1 = Input(shape=(n_units,))
    decoder_state_input_h2 = Input(shape=(n_units,))
    decoder_state_input_c2 = Input(shape=(n_units,))
    decoder_states_inputs1 = [decoder_state_input_h1, decoder_state_input_c1]
    decoder_states_inputs2 = [decoder_state_input_h2, decoder_state_input_c2]
    decoder_states_inputs = [decoder_state_input_h1, decoder_state_input_c1,decoder_state_input_h2, decoder_state_input_c2]
    decoder_outputs, state_h1, state_c1 = decoder_lstm1(decoder_inputs, initial_state=decoder_states_inputs1)
    decoder_outputs, state_h2, state_c2 = decoder_lstm2(decoder_outputs, initial_state=decoder_states_inputs2)
    decoder_states = [state_h1, state_c1, state_h2, state_c2]

    # Common batchNorm
    # decoder_outputs = commonBatchNorm(decoder_outputs)

    # Inference decoder for Class out
    decoder_output1 = decoder_dense10a(decoder_outputs)
    # decoder_output1 = batchNorm1(decoder_output1)
    decoder_output1 = decoder_Leaky10a(decoder_output1)
    decoder_output1 = decoder_dense10(decoder_output1)
    # decoder_output1 = batchNorm2(decoder_output1)
    decoder_output1 = decoder_Leaky10(decoder_output1)
    decoder_output1 = decoder_dense11(decoder_output1)
    # decoder_output1 = batchNorm3(decoder_output1)
    decoder_output1 = decoder_Leaky11(decoder_output1)
    decoder_output1 = decoder_dense12(decoder_output1)
    # decoder_output1 = batchNorm4(decoder_output1)
    decoder_output1 = decoder_Leaky12(decoder_output1)
    decoder_output1 = decoder_dense13(decoder_output1)
    # decoder_output1 = batchNorm5(decoder_output1)
    decoder_output1 = decoder_Leaky13(decoder_output1)
    decoder_output1 = decoder_dense14(decoder_output1)
    # decoder_output1 = batchNorm6(decoder_output1)
    decoder_output1 = decoder_Leaky14(decoder_output1)
    classOut = decoder_dense15(decoder_output1)

    # Inference Decoder for Velocity Out
    decoder_output2 = decoder2_concat([decoder_outputs,classOut])
    decoder_output2 = decoder_dense20a(decoder_output2)
    # decoder_output2 = batchNorm7(decoder_output2)
    decoder_output2 = decoder_Leaky20a(decoder_output2)
    decoder_output2 = decoder_dense20(decoder_output2)
    # decoder_output2 = batchNorm8(decoder_output2)
    decoder_output2 = decoder_Leaky20(decoder_output2)
    decoder_output2 = decoder_dense21(decoder_output2)
    # decoder_output2 = batchNorm9(decoder_output2)
    decoder_output2 = decoder_Leaky21(decoder_output2)
    decoder_output2 = decoder_dense22(decoder_output2)
    # decoder_output2 = batchNorm10(decoder_output2)
    decoder_output2 = decoder_Leaky22(decoder_output2)
    decoder_output2 = decoder_dense23(decoder_output2)
    # decoder_output2 = batchNorm11(decoder_output2)
    decoder_output2 = decoder_Leaky23(decoder_output2)
    decoder_output2 = decoder_dense24(decoder_output2)
    # decoder_output2 = batchNorm12(decoder_output2)
    decoder_output2 = decoder_Leaky24(decoder_output2)
    velocityOut = decoder_dense25(decoder_output2)

    # Inference Decoder Velocity Normalizer
    velocityConcat = velocityNormalized(velocityOut)

    #Inference  Decoder for position out
    decoder_output3 = decoder3_concat([decoder_outputs,classOut,velocityConcat])    
    decoder_output3 = decoder_dense30b(decoder_output3)
    # decoder_output3 = batchNorm13(decoder_output3)
    decoder_output3 = decoder_Leaky30b(decoder_output3)    
    decoder_output3 = decoder_dense30a(decoder_output3)
    # decoder_output3 = batchNorm14(decoder_output3)
    decoder_output3 = decoder_Leaky30a(decoder_output3)    
    decoder_output3 = decoder_dense30(decoder_output3)
    # decoder_output3 = batchNorm15(decoder_output3)
    decoder_output3 = decoder_Leaky30(decoder_output3)
    decoder_output3 = decoder_dense31(decoder_output3)
    # decoder_output3 = batchNorm16(decoder_output3)
    decoder_output3 = decoder_Leaky31(decoder_output3)    
    decoder_output3 = decoder_dense32(decoder_output3)
    # decoder_output3 = batchNorm17(decoder_output3)
    decoder_output3 = decoder_Leaky32(decoder_output3)
    decoder_output3 = decoder_dense33(decoder_output3)
    # decoder_output3 = batchNorm18(decoder_output3)
    decoder_output3 = decoder_Leaky33(decoder_output3)    
    decoder_output3 = decoder_dense34(decoder_output3)
    # decoder_output3 = batchNorm19(decoder_output3)
    decoder_output3 = decoder_Leaky34(decoder_output3)
    positionOut = decoder_dense35(decoder_output3)
    
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [classOut, velocityOut, positionOut] + decoder_states)

    opt =  Nadam()    #  RMSprop()

    model.compile(optimizer=opt, loss=['categorical_crossentropy', logcosh, euclidean_distance_loss])

    return model,encoder_model,decoder_model

# Class to hold all the relevet vehicle ID specific predicition intermediate information
class PredictionInfos():
    def __init__(self, input = [], decoderInput = [], state = [], output=[], groundTruth = [], initialPose = [], sectionIntersection = []):
        self.input = input
        self.decoderInput = decoderInput
        self.state = state
        self.output = output
        self.groundTruth = groundTruth
        self.initialPose = initialPose
        self.sectionIntersection = sectionIntersection


def IntermediatePredictionForTraining(processItem):

    # maxRelativeX and maxRelativeY should not be -999, check to be sure the global scope
    if(maxRealtiveX == -999 or maxRealtiveY == -999):
        print('maxRealitveX/Y values not set properly!!!')
        sys.exit()

    # Retrive the Process item
    eachRelevenatVehicle = processItem[0] #string
    currentTrainOrValStr = processItem[1]  #string
    gpuStr = processItem[2]  #string

    currentReleventVehicleList = dictByVehicles[eachRelevenatVehicle]
    currentReleventVehicleLength = len(currentReleventVehicleList)


    # Add the check for the side origins and side destination
    sideOrigin = currentReleventVehicleList[0][originIndex]
    sideDestination = currentReleventVehicleList[0][destinationIndex]
    if ((sideOrigin == 111 and sideDestination == 203) or (sideOrigin == 103 and sideDestination == 211) or (sideOrigin == 105 and sideDestination == 210) or (sideOrigin == 110 and sideDestination == 205) or (sideOrigin == 107 and sideDestination == 209) or (sideOrigin == 109 and sideDestination == 207)):
        return

    # and straight to straight vehicles
    if ((sideOrigin == 101 and sideDestination == 208) or (sideOrigin == 108 and sideDestination == 201)):
        return


    # Get the current target vehicle ID
    targetUpdatedID = eachRelevenatVehicle


    import tensorflow as tf


    with tf.device(gpuStr):

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=config)


        print('Waiting for GPU devices!!!')
        sleep(0.1)

        # Compile the model and load the weights
        model,encoder_model,decoder_model = ModelArch()

        encoder_model.load_weights(encoderModelFilename)
        print('Encoder loaded!!!')
        decoder_model.load_weights(decoderModelFilename)
        print('Decoder loaded!!!')


    # Extract the original key for surrounding car target vehicle seperation
    # If the value exists in mapper get the key as original ID of later surrounding car info target vehicle seperation   mydict.keys()[mydict.values().index(16)]
    targetOriginalID = None
    if(eachRelevenatVehicle in mapperDict.values()):
        valueList = list(mapperDict.values())
        targetOriginalID = list(mapperDict.keys())[valueList.index(eachRelevenatVehicle)]
        # print('Original Key for ' + eachKey + ' is found and the Key is ' + str(originalID))
    else:
        targetOriginalID = eachRelevenatVehicle

    # for idx in range(historyTemporal,currentReleventVehicleLength-futureTemporal): # for time saving
    for idx in range(historyTemporal,currentReleventVehicleLength-futureTemporal,10):

        # Prepare the trakcer Dict
        trackerDict = dict()
        trackerDict[targetUpdatedID] = []

        # for input
        for jdx in range(idx-historyTemporal,idx+futureTemporal):

            currentVechicleID = currentReleventVehicleList[jdx][vechileIDIndex]
            currentLocalX = currentReleventVehicleList[jdx][localXIndex]
            currentLocalY = currentReleventVehicleList[jdx][localYIndex]
            currentVelocity = currentReleventVehicleList[jdx][velocityIndex]
            currentLaneID = currentReleventVehicleList[jdx][laneIDIndex]
            currentDirection = currentReleventVehicleList[jdx][directionIndex]
            currentMovement = currentReleventVehicleList[jdx][movementIndex]
            currentTime = currentReleventVehicleList[jdx][globalTimeIndex]
            currentFrame = currentReleventVehicleList[jdx][frameIDIndex]
            currentSection = currentReleventVehicleList[jdx][sectionIndex]
            currentIntersection = currentReleventVehicleList[jdx][intersectionIndex]

            # Prepeare the target vehicle current input and append and at the end of the tracker dict list
            dictInput = [currentLocalX,currentLocalY,currentVelocity,currentLaneID,currentDirection,currentMovement,currentTime,currentFrame,currentSection,currentIntersection]
            trackerDict[targetUpdatedID].append(dictInput)


            # Get the surrounding cars
            otherVehicles = dictByFrames[str(currentTime)]

            # Target vehicle removal flag 
            targetRemovedFlag = 0

            for eachOtherVehicle in otherVehicles:
                currentVechicleID = eachOtherVehicle[vechileIDIndex]

                if(str(currentVechicleID) == targetOriginalID):
                    targetRemovedFlag = 1
                    continue

                currentLocalX = eachOtherVehicle[localXIndex]
                currentLocalY = eachOtherVehicle[localYIndex]
                currentVelocity = eachOtherVehicle[velocityIndex]
                currentLaneID = eachOtherVehicle[laneIDIndex]
                currentDirection = eachOtherVehicle[directionIndex]
                currentMovement = eachOtherVehicle[movementIndex]
                currentTime = eachOtherVehicle[globalTimeIndex]
                currentFrame = eachOtherVehicle[frameIDIndex]
                currentSection = eachOtherVehicle[sectionIndex]
                currentIntersection = eachOtherVehicle[intersectionIndex]

                dictInput = [currentLocalX,currentLocalY,currentVelocity,currentLaneID,currentDirection,currentMovement,currentTime,currentFrame,currentSection,currentIntersection]

                # append the surrounding car info in the trakcer dict
                # Check if the vehicle ID exist in mapper dict
                # if yes use the updated key to avoid duplication
                # Vehicle Birth in tracker Dict
                if (str(currentVechicleID) not in trackerDict.keys()):
                    trackerDict[str(currentVechicleID)] = []
                    trackerDict[str(currentVechicleID)].append(dictInput)
                else:
                    # Check the diff of last frame and last time with current frame and current time to avide duplicate vehicle IDs
                    lastTime = trackerDict[str(currentVechicleID)][-1][6]  # -1 for last item and 6 is time index in TrakcerDict list
                    lastFrame = trackerDict[str(currentVechicleID)][-1][7] # -1 for last item and 7 is frame index in TrakcerDict list
                    # If the check pass then the ID is original
                    if ((abs(currentTime-lastTime) == 100) and (abs(currentFrame-lastFrame) == 1)):
                        trackerDict[str(currentVechicleID)].append(dictInput)
                    # If the check fails then duplicate key. Look for the updated key from the mapper dict
                    else:
                        updatedVehicleKey = mapperDict[str(currentVechicleID)]
                        if (str(updatedVehicleKey) not in trackerDict.keys()):
                            trackerDict[str(updatedVehicleKey)] = []
                            trackerDict[str(updatedVehicleKey)].append(dictInput)
                        else:
                            trackerDict[str(updatedVehicleKey)].append(dictInput)

            # Check the target removed flag
            if(targetRemovedFlag != 1):
                print('Traget vehicle not removed properly!!!')
                sys.exit()

        # Do the prediction stuff here
        ######################################################.

        # print('Tracker populated!!! goind for predition')

        # Identify prediction eligible vehicles having 30 frames history and 50 Frames Future
        eligibleVehicleKeys = []
        for trackerKey in trackerDict.keys():
            vehicleFrameLength = len(trackerDict[trackerKey])
            if(vehicleFrameLength == (historyTemporal+futureTemporal)):
                eligibleVehicleKeys.append(trackerKey)
            
            if(vehicleFrameLength > (historyTemporal+futureTemporal)):
                print('Tracker Over populated for vehicle: ' + trackerKey)
                print('Vehicle Length is ' + str(vehicleFrameLength))
                print('This is unwanted event....')
                sys.exit()
        
        # If no Eligible vehicles not expected. Atleast target vehicle should be eligible
        if not eligibleVehicleKeys:
            print('No eligible vehicle for the during intermediate training phase!!!')
            sys.exit()


        # Ignore the vehicles far from the target vehicle list 
        modifiedEligibleKeys = []
        eligibleTargetPoseX = trackerDict[targetUpdatedID][0][0]   # 0 for the first item in the list and 0 for poseX index in tracker dict list
        eligibleTeargetPoseY = trackerDict[targetUpdatedID][0][1]   # 0 for the first item in the list and 1 for poseY index in tracker dict list
    
        for eachDistCheckKey in eligibleVehicleKeys:
            eligibleOtherPoseX = trackerDict[eachDistCheckKey][0][0]   # 0 for the first item in the list and 0 for poseX index in tracker dict list
            eligibleOtherPoseY = trackerDict[eachDistCheckKey][0][1]   # 0 for the first item in the list and 1 for poseY index in tracker dict list
            checkDist = math.sqrt((((eligibleTargetPoseX-eligibleOtherPoseX)**2)+((eligibleTeargetPoseY-eligibleOtherPoseY)**2)))
            if(checkDist <= predictionDistanceThreshold):
                modifiedEligibleKeys.append(eachDistCheckKey)


        # Prepare a dictionary to hold all the prediction relevent information (input, decoderInput, state, predictedOutput and GroundTurthOutput) against each vehicle
        predictionDict = dict()

        # Populate all the input/gournd truth infos in the prediction dictionary  
        ########################## eligibleVehicleKeys replaced by modifiedEligibleKeys (for faster)    ######################################
        for eachEligibleKey in modifiedEligibleKeys:
            predictionInfoObj = PredictionInfos([],[],[],[],[],[])
            predictionDict[eachEligibleKey] = predictionInfoObj
            # Get all the input infos from the traker dict for that specific vehicle
            totalInfo = trackerDict[eachEligibleKey].copy()
            inputInfo = totalInfo[0:historyTemporal]
            # Get the first element for relative movement calculation and add in the prediction object
            intitalX = inputInfo[0][0]  # 0 for first item and 0 for poseX index is 0 in trakcer dict
            intitalY = inputInfo[0][1]  # 0 for first item and 1 for poseY index is 0 in trakcer dict
            predictionDict[eachEligibleKey].initialPose = [intitalX,intitalY]
            predicitionInputList = []
            for udx, eachInputInfo in enumerate(inputInfo):

                targetLocalX = eachInputInfo[0]  # 0 is poseX index in trakcer dict list
                targetLocalY = eachInputInfo[1]  # 1 is poseY index in trakcer dict list
                targetSection = eachInputInfo[8]  # 8 is section index in trakcer dict list
                targetIntersection = eachInputInfo[9]  # 9 is intersection index in trakcer dict list

                # Add check for time
                targetTime = eachInputInfo[6]  # 6 is time index in trakcer dict list


                tempPredictionInput = eachInputInfo.copy()[:-4] # Ignore the last four items (section intersection FrameID and Time) for the input
                # convert the absolute position to normalized relative position
                tempPredictionInput[0] = abs(tempPredictionInput[0]-intitalX)/maxRealtiveX
                tempPredictionInput[1] = abs(tempPredictionInput[1]-intitalY)/maxRealtiveY

                # Calculate Nearest junction distance and extend to the input temporary row list
                juncDist = CalculateNearestJuncLoc(targetSection, targetIntersection, targetLocalX, targetLocalY)
                tempPredictionInput.insert(len(tempPredictionInput),juncDist)

                # Get the surrounding cars for the same frame
                # Get the surrounding car IDs by getting all keys and removing the current key
                ############################# eligibleVehicleKeys replaced by modifiedEligibleKeys (for faster) ###############
                surroudingCarIds = modifiedEligibleKeys[:]
                surroudingCarIds.remove(eachEligibleKey)

                # If the surrouding car count is less than or equal to the decided surrouding car count then append all the poses and manage by zero padding
                predSurroudingCarCount = len(surroudingCarIds)
                predictionPaddingCount = surroudingCarCounts - predSurroudingCarCount
                if(predSurroudingCarCount <= surroudingCarCounts):
                    for eachSurroundingCarID in surroudingCarIds:
                        # Check for time
                        surroundingCarTime = trackerDict[eachSurroundingCarID][udx][6]  # udx for coresponnding Frame and 6 is time index in TrakcerDict list
                        if(surroundingCarTime!=targetTime):
                            print('Surrounding vehicle time mismatch during prediction!!!')
                            print('Surrounding vehicle time : ' + str(surroundingCarTime))
                            print('Target vehicle time : ' + str(targetTime))
                            sys.exit()

                        # Extract the absolute pose and convert to normalized relative pose
                        surroundingCarAbsoluteX = trackerDict[eachSurroundingCarID][udx][0]  # udx for coresponnding Frame and 0 is poseX index in TrakcerDict list
                        surroundingCarAbsoluteY = trackerDict[eachSurroundingCarID][udx][1]  # udx for coresponnding Frame and 1 is poseY index in TrakcerDict list
                        surroundingCarLocalX = abs(surroundingCarAbsoluteX - intitalX)/maxRealtiveX
                        surroundingCarLocalY = abs(surroundingCarAbsoluteY - intitalY)/maxRealtiveY
                        # Extract rest of the features
                        surroundingCarVelocity = trackerDict[eachSurroundingCarID][udx][2]  # udx for coresponnding Frame and 2 is velocity index in TrakcerDict list
                        surroundingCarLaneID = trackerDict[eachSurroundingCarID][udx][3]  # udx for coresponnding Frame and 3 is lane ID index in TrakcerDict list
                        surroundingCarDirection = trackerDict[eachSurroundingCarID][udx][4]  # udx for coresponnding Frame and 4 is Direction index in TrakcerDict list
                        surroundingCarMovement = trackerDict[eachSurroundingCarID][udx][5]  # udx for coresponnding Frame and 5 is Movement index in TrakcerDict list
                        # Extract distance from the nearest junction
                        surroundingCarSection = trackerDict[eachSurroundingCarID][udx][8]  # udx for coresponnding Frame and 8 is section index in TrakcerDict list
                        surroundingCarIntersection = trackerDict[eachSurroundingCarID][udx][9]  # udx for coresponnding Frame and 9 is intersection index in TrakcerDict list
                        juncDist = CalculateNearestJuncLoc(surroundingCarSection, surroundingCarIntersection, surroundingCarAbsoluteX, surroundingCarAbsoluteY)

                        # If the other vehicle distance is more that allowable then append zeros
                        otherDist = math.sqrt((((surroundingCarAbsoluteX-targetLocalX)**2)+((surroundingCarAbsoluteY-targetLocalY)**2)))
                        
                        if(otherDist>maximumSurroundingCarDist):
                            tempPredictionInput.extend([0,0,0,0,0,0,0])
                        else:
                            tempPredictionInput.extend([surroundingCarLocalX,surroundingCarLocalY,surroundingCarVelocity,surroundingCarLaneID,surroundingCarDirection,surroundingCarMovement,juncDist])
                    
                    predZeroPadList = [0,0,0,0,0,0,0]
                    for adx in range(0,predictionPaddingCount):
                        tempPredictionInput.extend(predZeroPadList)

                # Else the surrouding car count is more than the decided surrouding car count then select the nearest 4 cars
                else:
                    # Get the surrounding car's coresponding frame position and calculate distance
                    surroundingCarDistanceList = []
                    for eachSurroundingCarID in surroudingCarIds:
                        # Check for time
                        surroundingCarTime = trackerDict[eachSurroundingCarID][udx][6]  # udx for coresponnding Frame and 6 is time index in TrakcerDict list
                        if(surroundingCarTime!=targetTime):
                            print('Surrounding vehicle time mismatch during prediction!!!')
                            print('Surrounding vehicle time : ' + str(surroundingCarTime))
                            print('Target vehicle time : ' + str(targetTime))
                            sys.exit()

                        surroundingCarLocalX = trackerDict[eachSurroundingCarID][udx][0]  # udx for coresponnding Frame and 0 is poseX index in tracker dict list
                        surroundingCarLocalY = trackerDict[eachSurroundingCarID][udx][1]  # udx for coresponnding Frame and 1 is poseY index in tracker dict list
                        surroundingDist = math.sqrt(((surroundingCarLocalX-targetLocalX)**2) + ((surroundingCarLocalY-targetLocalY)**2))
                        surroundingCarDistanceList.append([eachSurroundingCarID,surroundingDist])
                    
                    # Sort the list based on distance and gather the lowest distance car IDs
                    surroundingCarDistanceList = sorted(surroundingCarDistanceList,key=lambda x: x[1])
                    surroundingCarDistanceArray = np.array(surroundingCarDistanceList)
                    releventSurroundingIds = surroundingCarDistanceArray[0:surroudingCarCounts,0:1]

                    # Add the relevent input of nearest cars to temp list
                    for eachReleventSurroundingID in releventSurroundingIds:
                        # Extract the absolute pose and convert to normalized relative pose
                        surroundingCarAbsoluteX = trackerDict[eachReleventSurroundingID[0]][udx][0]  # udx for coresponnding Frame and 0 is poseX index in TrakcerDict list
                        surroundingCarAbsoluteY = trackerDict[eachReleventSurroundingID[0]][udx][1]  # udx for coresponnding Frame and 1 is poseY index in TrakcerDict list
                        surroundingCarLocalX = abs(surroundingCarAbsoluteX - intitalX)/maxRealtiveX
                        surroundingCarLocalY = abs(surroundingCarAbsoluteY - intitalY)/maxRealtiveY
                        # Extract rest of the features
                        surroundingCarVelocity = trackerDict[eachReleventSurroundingID[0]][udx][2]  # udx for coresponnding Frame and 2 is velocity index in TrakcerDict list
                        surroundingCarLaneID = trackerDict[eachReleventSurroundingID[0]][udx][3]  # udx for coresponnding Frame and 3 is Lane index in TrakcerDict list
                        surroundingCarDirection = trackerDict[eachReleventSurroundingID[0]][udx][4]  # udx for coresponnding Frame and 4 is Direction  index in TrakcerDict list
                        surroundingCarMovement = trackerDict[eachReleventSurroundingID[0]][udx][5]  # udx for coresponnding Frame and 5 is Movement index in TrakcerDict list
                        # Extract distance from the nearest junction
                        surroundingCarSection = trackerDict[eachReleventSurroundingID[0]][udx][8]  # udx for coresponnding Frame and 8 is section index in TrakcerDict list
                        surroundingCarIntersection = trackerDict[eachReleventSurroundingID[0]][udx][9]  # udx for coresponnding Frame and 9 is intersection index in TrakcerDict list
                        juncDist = CalculateNearestJuncLoc(surroundingCarSection, surroundingCarIntersection, surroundingCarAbsoluteX, surroundingCarAbsoluteY)

                        # If the other vehicle distance is more that allowable then append zeros
                        otherDist = math.sqrt((((surroundingCarAbsoluteX-targetLocalX)**2)+((surroundingCarAbsoluteY-targetLocalY)**2)))
                        if(otherDist>maximumSurroundingCarDist):
                            tempPredictionInput.extend([0,0,0,0,0,0,0])
                        else:
                            tempPredictionInput.extend([surroundingCarLocalX,surroundingCarLocalY,surroundingCarVelocity,surroundingCarLaneID,surroundingCarDirection,surroundingCarMovement,juncDist])
                
                # Add the current frame input info in the input list
                predicitionInputList.append(tempPredictionInput)

            # Add the current Vehicles all history frame input to the prediction dict object (input field)
            predictionDict[eachEligibleKey].input = predicitionInputList

            # print('Prediction input preped!!!!')

            # Add the ground truth output pose in to the prediction dict object for error calculation
            outputInfo = totalInfo[historyTemporal:historyTemporal+futureTemporal]
            #tempGroundTruthPoseList = []
            for eachOutputInfo in outputInfo:
                groundTruthPoseX = abs(eachOutputInfo[0] - intitalX)  # 0 is poseX index in trakcer dict list
                groundTruthPoseY = abs(eachOutputInfo[1] - intitalY)  # 1 is poseY index in trakcer dict list
                groundTruthVelocity = eachOutputInfo[2]               # 2 is velocity index in trakcer dict list
                trueMovement = eachOutputInfo[5]                      # 5 is movement index in trakcer dict list
                nextMovementClassData = MovementToClassForm(trueMovement)


                # Add the section intersetion values for decoder distane from junc calculation
                truthSection = eachOutputInfo[8]  # 8 is section index in trakcer dict list
                truthIntersection = eachOutputInfo[9]  # 9 is intersection index in trakcer dict list

                # Denormalize poseX and poseY as the traker dict is for input and it is normalized
                # No need to denormalize the poses are absolute and not normalized
                # denormPoseX = (groundTruthPoseX*(maxLocalX-minLocalX)+minLocalX)
                # denormPoseY = (groundTruthPoseY*(maxLocalY-minLocalY)+minLocalY)

                predictionDict[eachEligibleKey].groundTruth.append([nextMovementClassData[0],nextMovementClassData[1],nextMovementClassData[2],groundTruthVelocity,groundTruthPoseX,groundTruthPoseY])
                predictionDict[eachEligibleKey].sectionIntersection.append([truthSection,truthIntersection])
        
        # Initialize decoderInputData list to hold the target decoder input data        
        decoderInputData = []
        # Add the decoder inputs in the prediction dict against each vehicle
        # Predict the encoder state for each vehicle and update the prediction dict state values
        for eachPredDictKey in predictionDict.keys():
            lastInput = predictionDict[eachPredDictKey].input[-1]
            predDecoderInput = []
            for bdx in range(0,len(lastInput),inputFeatureCount):  # inputFeatureCount 7 is number of input features for each car
                lastInputPoseX = lastInput[bdx]
                lastInputPoseY = lastInput[bdx+1]
                lastInputVelocity = lastInput[bdx+2]
                lastInputMovement = lastInput[bdx+5]
                lastInputClassInfo = MovementToClassForm(lastInputMovement)
                lastDistFromJunc = lastInput[bdx+6]
                predDecoderInput.extend([lastInputPoseX,lastInputPoseY,lastInputVelocity,lastInputClassInfo[0],lastInputClassInfo[1],lastInputClassInfo[2],lastDistFromJunc])
            # Add the prepered decoder input in the prediction dict object (decoder input field)
            predictionDict[eachPredDictKey].decoderInput = predDecoderInput[:]

            # If the predictKey is targetID then add the first decoder Input in the decoderInputData
            if(eachPredDictKey == targetUpdatedID):
                decoderInputData.append(predDecoderInput[:])

            # Get the input for the current vehicle to predict the encoder state
            currentPredInput = np.array(predictionDict[eachPredDictKey].input).reshape(1,historyTemporal,globalInputFeatures)

            # Predict the Encoder state for that specific vehicle and update the prediction dict
            currentState = encoder_model.predict(currentPredInput)
            predictionDict[eachPredDictKey].state = currentState

            # print('state predicted!!!!')
        
        # Predict till the decided future temporal
        for cdx in range(futureTemporal):

            # print('Processing time : ' + str(cdx))

            # Predict the next frame for each vechicle in the prediction dict
            for eachPredDictKey in predictionDict.keys():
                # Prepare the target seq and state for the current Vehicle
                target_seq = np.array(predictionDict[eachPredDictKey].decoderInput).reshape(1,1,globalDecoderFeatures)
                predState = predictionDict[eachPredDictKey].state

                # Predict the next frame for the current vehicle
                classPred, velcoityPred, posePred, h1, c1, h2, c2 = decoder_model.predict([target_seq] + predState)

                # Store the prediction in the prediction dict w.r.t the coresponding vehicle
                finalOutput = [posePred[0][0][0],posePred[0][0][1],velcoityPred[0][0][0],classPred[0][0][0],classPred[0][0][1],classPred[0][0][2]]
                predictionDict[eachPredDictKey].output.append(finalOutput)

                # Update the state for each vehicle in the prediction dict
                predictionDict[eachPredDictKey].state = [h1, c1, h2, c2]
            
            # Update the decoder input for each vechicle in the prediction dict
            for eachPredDictKey in predictionDict.keys():

                # Get the last output of the target vehicle 
                lastOutput = predictionDict[eachPredDictKey].output[-1]

                # Add the target vechicle's last output in the decoder input as basic decoder input (after normalization)
                # Get the coresponding vehicles last output
                lastOutputPoseX = lastOutput[0]   # 0 is poseX index in output list of prediction dict
                lastOutputPoseY = lastOutput[1]   # 1 is poseX index in output list of prediction dict
                lastOutputVelocity = lastOutput[2]   # 2 is velocity index in output list of prediction dict
                lastClassOutput0 = lastOutput[3]   # 3 is 0 class info index in output list of prediction dict
                lastClassOutput1 = lastOutput[4]   # 4 is 1 class info index in output list of prediction dict
                lastClassOutput2 = lastOutput[5]   # 5 is 2 class info index in output list of prediction dict

                # Calculate the absolute position using initial pose and relative predicted pose to estimate the surrounding car dist
                targetInitialPose = predictionDict[eachPredDictKey].initialPose
                targetAbsolutePoseX = targetInitialPose[0] + lastOutputPoseX   # 0 is the index for poseX in prediction object initialPose field
                targetAbsolutePoseY = targetInitialPose[1] + lastOutputPoseY   # 1 is the index for poseY in prediction object initialPose field

                # Normalize poseX, poseY and velocity before adding to the decoder input
                normalizedPredPoseX = lastOutputPoseX/maxRealtiveX
                normalizedPredPoseY = lastOutputPoseY/maxRealtiveY
                normalizedPredVelocity = (lastOutputVelocity-minVel)/(maxVel-minVel)

                # Calculate the distance from the nearest junction using the section intersection from prediction dict object and converted absolute values using predicted local values
                currentTargetSection = predictionDict[eachPredDictKey].sectionIntersection[cdx][0] # cdx is current future time which is same as the index in the sectionIntersection list in pred object and 0 is for first item is Section
                currentTargetIntersection = predictionDict[eachPredDictKey].sectionIntersection[cdx][1] # cdx is current future time which is same as the index in the sectionIntersection list in pred object and 1 is for second item is Intersection
                targetDistFromJunc = CalculateNearestJuncLoc(currentTargetSection, currentTargetIntersection, targetAbsolutePoseX, targetAbsolutePoseY)

                # Finally add the normalized values into the temp decoder input
                tempDecoderInput = [normalizedPredPoseX,normalizedPredPoseY,normalizedPredVelocity,lastClassOutput0,lastClassOutput1,lastClassOutput2,targetDistFromJunc]

                # Add the nearest vehicle's info into the decoder list
                # Get the surrounding car IDs by getting all keys and removing the current key/target vehicle
                allPredictionKeys = list(predictionDict.keys())
                decoderSurroudingCarIds = allPredictionKeys[:]
                decoderSurroudingCarIds.remove(eachPredDictKey)
                
                decoderSurroudingCarCount = len(decoderSurroudingCarIds)
                predictionPaddingCount = surroudingCarCounts - decoderSurroudingCarCount
                # If the surrouding car count is less than or equal to the decided surrouding car count then append all the poses and manage by zero padding
                if(decoderSurroudingCarCount <= surroudingCarCounts):
                    for eachDecoderSurroundingCarID in decoderSurroudingCarIds:
                        # Get the last output of the coresponding vehicle 
                        lastOutput = predictionDict[eachDecoderSurroundingCarID].output[-1]

                        # Add the coresponding surrounding vechicle's last output in the decoder input as basic decoder input (after normalization)
                        # Get the coresponding vehicles last output
                        lastOutputPoseX = lastOutput[0]   # 0 is poseX index in output list of prediction dict
                        lastOutputPoseY = lastOutput[1]   # 1 is poseX index in output list of prediction dict
                        lastOutputVelocity = lastOutput[2]   # 2 is velocity index in output list of prediction dict
                        lastClassOutput0 = lastOutput[3]   # 3 is 0 class info index in output list of prediction dict
                        lastClassOutput1 = lastOutput[4]   # 4 is 1 class info index in output list of prediction dict
                        lastClassOutput2 = lastOutput[5]   # 5 is 2 class info index in output list of prediction dict

                        # Normalize poseX, poseY and velocity before adding to the decoder input
                        normalizedPredPoseX = lastOutputPoseX/maxRealtiveX
                        normalizedPredPoseY = lastOutputPoseY/maxRealtiveY
                        normalizedPredVelocity = (lastOutputVelocity-minVel)/(maxVel-minVel)

                        # Calculate the absolute position using initial pose and relative predicted pose to estimate its distance from target car
                        surroundingInitialPose = predictionDict[eachDecoderSurroundingCarID].initialPose
                        surroundingAbsoluteX = surroundingInitialPose[0] + lastOutputPoseX # 0 is the index for poseX in prediction object initialPose field
                        surroundingAbsoluteY = surroundingInitialPose[1] + lastOutputPoseY # 1 is the index for poseY in prediction object initialPose field

                        # Calculate the distance from the nearest junction using the section intersection from prediction dict object and converted absolute values using predicted local values
                        currentSurroundingSection = predictionDict[eachDecoderSurroundingCarID].sectionIntersection[cdx][0] # cdx is current future time which is same as the index in the sectionIntersection list in pred object and 0 is for first item is Section
                        currentSurroundingIntersection = predictionDict[eachDecoderSurroundingCarID].sectionIntersection[cdx][1] # cdx is current future time which is same as the index in the sectionIntersection list in pred object and 1 is for first item is Intersection
                        surroundingDistFromJunc = CalculateNearestJuncLoc(currentSurroundingSection, currentSurroundingIntersection, surroundingAbsoluteX, surroundingAbsoluteY)

                        # If the other vehicle distance is more that allowable then append zeros
                        otherDist = math.sqrt((((surroundingAbsoluteX-targetAbsolutePoseX)**2)+((surroundingAbsoluteY-targetAbsolutePoseY)**2)))

                        # Finally add the normalized values into the temp decoder input
                        if(otherDist>maximumSurroundingCarDist):
                            tempDecoderInput.extend([0,0,0,0,0,0,0])
                        else:
                            tempDecoderInput.extend([normalizedPredPoseX,normalizedPredPoseY,normalizedPredVelocity,lastClassOutput0,lastClassOutput1,lastClassOutput2,surroundingDistFromJunc])
                    
                    predZeroPadList = [0,0,0,0,0,0,0]
                    for adx in range(0,predictionPaddingCount):
                        tempDecoderInput.extend(predZeroPadList)

                # Else the surrouding car count is more than the decided surrouding car count then select the nearest 4 cars
                else:
                    decoderSurroundingCarDistanceList = []
                    for eachDecoderSurroundingCarID in decoderSurroudingCarIds:
                        # Get the last output poseX and poseY to calculate distance with the target vehicle
                        lastSurroundingOutput = predictionDict[eachDecoderSurroundingCarID].output[-1]
                        lastSurroundingOutputPoseX = lastSurroundingOutput[0]   # 0 is poseX index in output list of prediction dict
                        lastSurroundingOutputPoseY = lastSurroundingOutput[1]   # 1 is poseX index in output list of prediction dict

                        # Calculate the absolute position using initial pose and relative predicted pose to estimate its distance from target car
                        surroundingInitialPose = predictionDict[eachDecoderSurroundingCarID].initialPose
                        surroundingAbsoluteX = surroundingInitialPose[0] + lastSurroundingOutputPoseX # 0 is the index for poseX in prediction object initialPose field
                        surroundingAbsoluteY = surroundingInitialPose[1] + lastSurroundingOutputPoseY # 1 is the index for poseY in prediction object initialPose field

                        surroundingDist =  math.sqrt((((surroundingAbsoluteX-targetAbsolutePoseX)**2)+((surroundingAbsoluteY-targetAbsolutePoseY)**2)))
                        decoderSurroundingCarDistanceList.append([eachDecoderSurroundingCarID,surroundingDist])
                    
                    # Sort the list based on distance and gather the lowest distance car IDs
                    decoderSurroundingCarDistanceList = sorted(decoderSurroundingCarDistanceList,key=lambda x: x[1])
                    decoderSurroundingCarDistanceArray = np.array(decoderSurroundingCarDistanceList)
                    decoderReleventSurroundingIds = decoderSurroundingCarDistanceArray[0:surroudingCarCounts,0:1]

                    # Add the relevent input of nearest cars to temp list
                    for eachDecoderReleventSurroundingID in decoderReleventSurroundingIds:
                        # Add the coresponding surrounding vechicle's last output in the decoder input (after normalization)
                        # Get the coresponding vehicles last output
                        lastSurroundingOutput = predictionDict[eachDecoderReleventSurroundingID[0]].output[-1]
                        lastOutputPoseX = lastSurroundingOutput[0]   # 0 is poseX index in output list of prediction dict
                        lastOutputPoseY = lastSurroundingOutput[1]   # 1 is poseX index in output list of prediction dict
                        lastOutputVelocity = lastSurroundingOutput[2]   # 2 is velocity index in output list of prediction dict
                        lastClassOutput0 = lastSurroundingOutput[3]   # 3 is 0 class info index in output list of prediction dict
                        lastClassOutput1 = lastSurroundingOutput[4]   # 4 is 1 class info index in output list of prediction dict
                        lastClassOutput2 = lastSurroundingOutput[5]   # 5 is 2 class info index in output list of prediction dict

                        # Normalize poseX, poseY and velocity before adding to the decoder input
                        normalizedPredPoseX = lastOutputPoseX/maxRealtiveX
                        normalizedPredPoseY = lastOutputPoseY/maxRealtiveY
                        normalizedPredVelocity = (lastOutputVelocity-minVel)/(maxVel-minVel)

                        # Calculate the absolute position using initial pose and relative predicted pose to estimate its distance from target car
                        surroundingInitialPose = predictionDict[eachDecoderReleventSurroundingID[0]].initialPose
                        surroundingAbsoluteX = surroundingInitialPose[0] + lastOutputPoseX # 0 is the index for poseX in prediction object initialPose field
                        surroundingAbsoluteY = surroundingInitialPose[1] + lastOutputPoseY # 1 is the index for poseY in prediction object initialPose field

                        # If the other vehicle distance is more that allowable then append zeros
                        otherDist = math.sqrt((((surroundingAbsoluteX-targetAbsolutePoseX)**2)+((surroundingAbsoluteY-targetAbsolutePoseY)**2)))

                        # Calculate the distance from the nearest junction using the section intersection from prediction dict object and converted absolute values using predicted local values
                        currentSurroundingSection = predictionDict[eachDecoderReleventSurroundingID[0]].sectionIntersection[cdx][0] # cdx is current future time which is same as the index in the sectionIntersection list in pred object and 0 is for first item is Section
                        currentSurroundingIntersection = predictionDict[eachDecoderReleventSurroundingID[0]].sectionIntersection[cdx][1] # cdx is current future time which is same as the index in the sectionIntersection list in pred object and 1 is for first item is Intersection
                        surroundingDistFromJunc = CalculateNearestJuncLoc(currentSurroundingSection, currentSurroundingIntersection, surroundingAbsoluteX, surroundingAbsoluteY)

                        if(otherDist>maximumSurroundingCarDist):
                            tempDecoderInput.extend([0,0,0,0,0,0,0])
                        else:
                            # Finally add the normalized values into the temp decoder input
                            tempDecoderInput.extend([normalizedPredPoseX,normalizedPredPoseY,normalizedPredVelocity,lastClassOutput0,lastClassOutput1,lastClassOutput2,surroundingDistFromJunc])
                
                # Finally update the decoder input in the prediction dict
                predictionDict[eachPredDictKey].decoderInput = tempDecoderInput

                # Add the current decoder input to the localDecoderInput if the key is targetVehicleID
                if(eachPredDictKey == targetUpdatedID):
                    decoderInputData.append(tempDecoderInput)

        # DecoderInputData is populated already. Extract rest of the input/output from the prediction dict
        # For decoder input data remove the last item as that is extra due to the last step prediction
        decoderInputData.pop(-1)
        localXData = predictionDict[targetUpdatedID].input
        totalGroundTruthOutputList = predictionDict[targetUpdatedID].groundTruth
        localYMovementData = []
        localYVelData = []
        localYPoseData = []
        for eachGroundTurthOutput in totalGroundTruthOutputList:
            localYMovementData.append([eachGroundTurthOutput[0],eachGroundTurthOutput[1],eachGroundTurthOutput[2]])   # GroundTruth class0, class1, clas2 index 0,1,2
            localYVelData.append(eachGroundTurthOutput[3])   # GroundTruth velocity index 3
            localYPoseData.append([eachGroundTurthOutput[4],eachGroundTurthOutput[5]])   # GroundTruth PoseX and PoseY index 4,5

        # Calculate the current error for each sample and append to the global manager list for intermediate error calculation
        # Get the predicted and ground truth poses from the predict dict object
        predictedIntermediatePose = predictionDict[targetUpdatedID].output

        # Length of both these lists should be equal
        if(len(totalGroundTruthOutputList) != len(predictedIntermediatePose)):
            print('Ground truth and predicted pose lists are not equal while intermediate error calculation')
            sys.exit()

        localErrorList = []
        for errorIdx,eachPose in enumerate(predictedIntermediatePose):
            predX = predictedIntermediatePose[errorIdx][0] # 0 is poseX index in output list of prediction dict 
            predY = predictedIntermediatePose[errorIdx][1] # 1 is poseY index in output list of prediction dict
            trueX = totalGroundTruthOutputList[errorIdx][4] # 4 is poseX index in Ground Truth list of prediction dict 
            trueY = totalGroundTruthOutputList[errorIdx][5] # 5 is poseY index in Ground Truth list of prediction dict
            euclidianError = math.sqrt(((predX-trueX)**2) + ((predY-trueY)**2)) * feetToMeter
            localErrorList.append(euclidianError)

        # Check the error list should be of lenght futureTemporal
        errorListLen = len(localErrorList)
        if(errorListLen != futureTemporal):
            print('Error list is of not expected length!!!')
            print('Expected error list length : ' + str(futureTemporal))
            print('Received error list length : ' + str(errorListLen))
            sys.exit()

        # Append the current local list to the main manager list
        errorManagerList.append(localErrorList)
        errorCountList.append(0)

        # Check the length for each thing and finally append to the main list
        # localXData length check
        localXDataLength = len(localXData)
        if(localXDataLength != historyTemporal):
            print('localXData in intermediate prediction step is not expected!!!')
            print('localXData expected length : ' + str(historyTemporal))
            print('localXData actual  length : ' + str(localXDataLength))
        # decoderInputData length check
        decoderInputDataLength = len(decoderInputData)
        if(decoderInputDataLength != futureTemporal):
            print('decoderInputData in intermediate prediction step is not expected!!!')
            print('decoderInputData expected length : ' + str(futureTemporal))
            print('decoderInputData actual  length : ' + str(decoderInputDataLength))
        
        # localYPoseData length check
        localYPoseDataLength = len(localYPoseData)
        if(localYPoseDataLength != futureTemporal):
            print('localYPoseData in intermediate prediction step is not expected!!!')
            print('localYPoseData expected length : ' + str(futureTemporal))
            print('localYPoseData actual  length : ' + str(localYPoseDataLength))
        
        # localYVelData length check
        localYVelDataLength = len(localYVelData)
        if(localYVelDataLength != futureTemporal):
            print('localYVelData in intermediate prediction step is not expected!!!')
            print('localYVelData expected length : ' + str(futureTemporal))
            print('localYVelData actual  length : ' + str(localYVelDataLength))

        # localYMovementData length check
        localYMovementDataLength = len(localYMovementData)
        if(localYMovementDataLength != futureTemporal):
            print('localYMovementData in intermediate prediction step is not expected!!!')
            print('localYMovementData expected length : ' + str(futureTemporal))
            print('localYMovementData actual  length : ' + str(localYMovementDataLength))


        # Append in the final validation or training set based on decided vehicle ID
        if(currentTrainOrValStr == validationStr):
            validationProcessList.append([localXData,decoderInputData,localYMovementData,localYVelData,localYPoseData])
        elif(currentTrainOrValStr == trainStr):
            trainProcessList.append([localXData,decoderInputData,localYMovementData,localYVelData,localYPoseData])
        else:
            print('Unknown Train Val string')
            sys.exit()


    countList.append(0)
    totalSamplesProcessed = len(countList)
    print('Finished Processing Sample : ' + str(totalSamplesProcessed))


def IntermediatePredictionProcess(vehicleFileName,trainOrValStr):


    # Clean up the manager list for training and validation
    errorManagerList[:] = []
    errorCountList[:] = []

    print('errorManagerList Initialized!!! Current length should be zero!!!')
    print('Current errorManagerList len ' + str(len(errorManagerList)))

    print('errorCountList Initialized!!! Current length should be zero!!!')
    print('Current errorCountList len ' + str(len(errorCountList)))


    # Read the vehicle ID list and populate the process list
    vehicleFileObj = open(vehicleFileName, "r")
    vehicleLoadedData = vehicleFileObj.readlines()
    vehicleProcessList = []

    # Assign alternative GPU ids to equally distribute model load
    totalGPUCount = 3
    gpuStrZero = '/gpu:0'
    gpuStrOne = '/gpu:1'
    gpuStrTwo = '/gpu:2'
    gpuCounter = 0

    for eachVehicle in vehicleLoadedData:
        # Get the vehicle ID
        vehicleIDStr = eachVehicle.rstrip()

        # Decide the GPU ID str
        selectedGPUId = gpuCounter%totalGPUCount
        if(selectedGPUId == 0):
            gpuSelectStr = gpuStrZero
        elif(selectedGPUId == 1):
            gpuSelectStr = gpuStrOne
        elif(selectedGPUId == 2):
            gpuSelectStr = gpuStrTwo
        else:
            print('Module division with 3 should not give anything other than 0,1,2!!!')
            print('selectedGPUId : ' + str(selectedGPUId))
            sys.exit()

        # Incremeent the GPU counter 
        gpuCounter = gpuCounter + 1

        vehicleProcessList.append([vehicleIDStr,trainOrValStr,gpuSelectStr])
    
    # valFileObj.close()
    vehicleFileObj.close()

    # Select the cores
    os.system("taskset -p -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32 %d" % os.getpid())

    # n is the number of item in each sublist decided based on the GPU memory
    n = 100       # 60

    # Change this back to big list....
    splittedList = [vehicleProcessList[i * n:(i + 1) * n] for i in range((len(vehicleProcessList) + n - 1) // n )] 

    for eachSplitedList in splittedList:
        # Create the process list inside the outer loop for each n vehicles
        processes = []
        for eachVehiclePorcItem in eachSplitedList:
            p = mp.Process(target=IntermediatePredictionForTraining, args=(eachVehiclePorcItem,))
            processes.append(p)
            p.start()

        # Wait for all the current n process to finish. 
        for process in processes:
            process.join()

    # # # # IntermediatePredictionForTraining(splittedList[0][0])


    # Process the manager list and calculate average the error
    print('Converting the Error Manager list to normal lists.....')
    errorNormalList = list(errorManagerList)
    print('List converted!!!')

    print('Process the error count!!!')
    errorCountListLen = len(list(errorCountList))

    intermediateErrorArray = np.zeros(futureTemporal)

    for eachErrorItem in errorNormalList:
        intermediateErrorArray = intermediateErrorArray + np.array(eachErrorItem)

    intermediateErrorArray = intermediateErrorArray/errorCountListLen

    print('Error till now sample count: ' + str(errorCountListLen))

    # Print the error till now and write to the intermediate error file
    print('Error till now!!!')
    print('####################################')
    print(intermediateErrorArray)
    print('####################################')

    # Write the error results to file for future analysis
    resultFileObj = open(resultFileName, 'a')
    resultFileObj.write(trainOrValStr + ' error till now!!! \n')
    resultFileObj.write('#################################### \n')

    for writeIndex, eachIntermediateArray in enumerate(intermediateErrorArray):
        errorStr = str(round(eachIntermediateArray, 2)) + ','
        resultFileObj.write(errorStr)
        if(writeIndex%10 == 0 and writeIndex>1):
            resultFileObj.write('\n')

    resultFileObj.write('\n')
    resultFileObj.write('####################################\n')
    resultFileObj.close()


def _training_worker(XTrain,decoderTrainInput,YClassTrain,YVelTrain,YPoseTrain,NumberEpochs,XVal,decoderValInput,YClassVal,YVelVal,YPoseVal,loadWeights):

    import tensorflow as tf
    from tensorflow.keras.callbacks import LearningRateScheduler
    from tensorflow.keras import callbacks

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # # config = tf.compat.v1.ConfigProto()
    # # config.gpu_options.allow_growth = True
    # # session = tf.compat.v1.Session(config=config)

    # Compile the model
    model,encoder_model,decoder_model = ModelArch()

    # Early stopping callback 
    class EarlyStoppingByLossVal(callbacks.Callback):
        def __init__(self, monitor='val_loss', value=1.25, verbose=0):
            #super(Callback, self).__init__()
            self.monitor = monitor
            self.value = value
            self.verbose = verbose

        def on_epoch_end(self, epoch, logs={}):
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

            if current < self.value:
                if self.verbose > 0:
                    print("Epoch %05d: early stopping THR" % epoch)
                self.model.stop_training = True


    # If the loadWeight is true load the pre trained weights from the h5 file
    # Means second or later round, use the second_Step_Decay function
    if(loadWeights):
        model.load_weights(mainModelFileName)
        lrate = LearningRateScheduler(second_step_decay)
    else:
        lrate = LearningRateScheduler(step_decay)

    # create the final callback
    esObj = EarlyStoppingByLossVal()
    callbacks_list = [esObj,lrate]   #[loss_history, lrate]

    print('Starting Model fit!!!!!!!!!!!')

    history = model.fit([XTrain,decoderTrainInput], [YClassTrain,YVelTrain,YPoseTrain], batch_size=batchSize, epochs=NumberEpochs, verbose=1, validation_data=([XVal,decoderValInput],[YClassVal,YVelVal,YPoseVal]), callbacks=callbacks_list, shuffle=True)

    print('Saving the model weights!!!')  
    model.save_weights(mainModelFileName)
    sleep(0.5)
    encoder_model.save_weights(encoderModelFilename)
    sleep(0.5)
    decoder_model.save_weights(decoderModelFilename)
    sleep(0.5)
    print('Model weights Saved!!!')

    # dump the training loss histroy
    # convert the history.history dict to a pandas DataFrame:     
    histDF = pd.DataFrame(history.history) 

    with open(historyFileName, mode='a') as h:
        histDF.to_csv(h)


if __name__ == '__main__':

    # global maxRealtiveX,maxRealtiveY,dictByFrames,dictByVehicles,validationVehicles,mapperDict
    # global dictByFrames,dictByVehicles,validationVehicles,mapperDict

    if(processOrRead == processStr):
        os.mkdir(folderName)
        XTrain,decoderTrainInput,YClassTrain,YVelTrain,YPoseTrain,XVal,decoderValInput,YClassVal,YVelVal,YPoseVal = TrainData(testTrajFilePath)
        print('All data processed!!!')
        sys.exit()

    # Re-Load the Vehicle and Frame based Dictionaries to populate the min max gloab values and global dicts
    # # # global dictByFrames, dictByVehicles, validationVehicles, mapperDict
    dictByFrames,dictByVehicles, mapperDict = CreateVehicleAndFrameDict(testTrajFilePath)
    finalVehicleKeys = list(dictByVehicles.keys())
    finalVehicleKeys.sort(key=float)
    finalFrameKeys = list(dictByFrames.keys())
    finalFrameKeys.sort(key=float)

    # Read the files and populate the arrays
    # Prepare the final lists of train and validation data
    # Train final lists
    XTrain = ReadFromFile('finalXTrain', historyTemporal)
    print('Finished XTrain Array!!!')
    decoderTrainInput = ReadFromFile('finalTrainDecoderInput', futureTemporal)
    print('Finished decoderTrainInput Array!!!')
    YClassTrain = ReadFromFile('finalYClassTrain', futureTemporal)
    print('Finished YClassTrain Array!!!')
    YVelTrain = ReadFromFile('finalYVelTrain', futureTemporal)
    print('Finished finalYVelTrain Array!!!')
    YPoseTrain = ReadFromFile('finalYPoseTrain', futureTemporal)
    print('Finished finalYPoseTrain Array!!!')

    # Validation final lists
    XVal = ReadFromFile('finalXVal', historyTemporal)
    print('Finished XVal Array!!!')
    decoderValInput = ReadFromFile('finalValDecoderInput', futureTemporal)
    print('Finished decoderValInput Array!!!')
    YClassVal = ReadFromFile('finalYClassVal', futureTemporal)
    print('Finished YClassVal Array!!!')
    YVelVal = ReadFromFile('finalYVelVal', futureTemporal)
    print('Finished YVelVal Array!!!')
    YPoseVal = ReadFromFile('finalYPoseVal', futureTemporal)
    print('Finished YPoseVal Array!!!')
    print('Finished All Array!!!')


    # Normalize the relative X/Y positions based on decoder Input
    # Identify the min and max locations among both train and val arrays
    print('Normalizing the X/Y poses!!!')
    for ydx in range(len(XTrain)):
        for zdx in range(0,globalInputFeatures,inputFeatureCount):
            currentXMax = max(XTrain[ydx,:,zdx])
            currentYMax = max(XTrain[ydx,:,zdx+1])
            if(currentXMax>maxRealtiveX):
                maxRealtiveX = currentXMax
            if(currentYMax>maxRealtiveY):
                maxRealtiveY = currentYMax
    
    for ydx in range(len(XVal)):
        for zdx in range(0,globalInputFeatures,inputFeatureCount):
            currentXMax = max(XVal[ydx,:,zdx])
            currentYMax = max(XVal[ydx,:,zdx+1])
            if(currentXMax>maxRealtiveX):
                maxRealtiveX = currentXMax
            if(currentYMax>maxRealtiveY):
                maxRealtiveY = currentYMax

    for ydx in range(len(decoderTrainInput)):
        for zdx in range(0,globalDecoderFeatures,decoderFeatureCount):
            currentXMax = max(decoderTrainInput[ydx,:,zdx])
            currentYMax = max(decoderTrainInput[ydx,:,zdx+1])
            if(currentXMax>maxRealtiveX):
                maxRealtiveX = currentXMax
            if(currentYMax>maxRealtiveY):
                maxRealtiveY = currentYMax

    for ydx in range(len(decoderValInput)):
        for zdx in range(0,globalDecoderFeatures,decoderFeatureCount):
            currentXMax = max(decoderValInput[ydx,:,zdx])
            currentYMax = max(decoderValInput[ydx,:,zdx+1])
            if(currentXMax>maxRealtiveX):
                maxRealtiveX = currentXMax
            if(currentYMax>maxRealtiveY):
                maxRealtiveY = currentYMax


    # Normalize train and val input plus train and val decoder input based of the identified min max pose
    for ydx in range(len(XTrain)):
        for zdx in range(0,globalInputFeatures,inputFeatureCount):
            XTrain[ydx,:,zdx] = XTrain[ydx,:,zdx]/maxRealtiveX
            XTrain[ydx,:,zdx+1] = XTrain[ydx,:,zdx+1]/maxRealtiveY
    
    for ydx in range(len(XVal)):
        for zdx in range(0,globalInputFeatures,inputFeatureCount):
            XVal[ydx,:,zdx] = XVal[ydx,:,zdx]/maxRealtiveX
            XVal[ydx,:,zdx+1] = XVal[ydx,:,zdx+1]/maxRealtiveY

    for ydx in range(len(decoderTrainInput)):
        for zdx in range(0,globalDecoderFeatures,decoderFeatureCount):
            decoderTrainInput[ydx,:,zdx] = decoderTrainInput[ydx,:,zdx]/maxRealtiveX
            decoderTrainInput[ydx,:,zdx+1] = decoderTrainInput[ydx,:,zdx+1]/maxRealtiveY

    for ydx in range(len(decoderValInput)):
        for zdx in range(0,globalDecoderFeatures,decoderFeatureCount):
            decoderValInput[ydx,:,zdx] = decoderValInput[ydx,:,zdx]/maxRealtiveX
            decoderValInput[ydx,:,zdx+1] = decoderValInput[ydx,:,zdx+1]/maxRealtiveY

    print('All X/Y poses are normalized!!!')

    # Print relative X Y max and each normalized array min max
    print('Relative X max :' + str(maxRealtiveX))
    print('Relative Y max :' + str(maxRealtiveY))

    print('XTrain max :' + str(np.amax(XTrain)))
    print('decoderTrainInput max :' + str(np.amax(decoderTrainInput)))
    print('XVal max :' + str(np.amax(XVal)))
    print('decoderValInput max :' + str(np.amax(decoderValInput)))


    # Read the validation file 
    valFileObj = open(validationFileName, "r")
    valLoadedData = valFileObj.readlines()
    validationVehicleList = []

    for eachValVehicle in valLoadedData:
        validationVehicleList.append(eachValVehicle.rstrip())
    
    valFileObj.close()

    print('XTrain Shape : ' + str(XTrain.shape))
    print('DecoderInput Shape : ' + str(decoderTrainInput.shape))
    print('DecoderInput Shape : ' + str(decoderTrainInput.shape))
    print('YClassTrain Shape : ' + str(YClassTrain.shape))
    print('YVelTrain Shape : ' + str(YVelTrain.shape))
    print('YPoseTrain Shape : ' + str(YPoseTrain.shape))
    print('XVal Shape : ' + str(XVal.shape))
    print('decoderValInput Shape : ' + str(decoderValInput.shape))
    print('YClassVal Shape : ' + str(YClassVal.shape))
    print('YVelVal Shape : ' + str(YVelVal.shape))
    print('YPoseVal Shape : ' + str(YPoseVal.shape))

    ######################################################################
    ###############   First Round Training ###############################
    ######################################################################

    # # # # # First Round Training (last param False, no need to load weight)
    # # # # training_process = mp.Process(target=_training_worker, args = (XTrain,decoderTrainInput,YClassTrain,YVelTrain,YPoseTrain,initialNumberEpochs,XVal,decoderValInput,YClassVal,YVelVal,YPoseVal,False))
    # # # # training_process.start()
    # # # # training_process.join()

    # # # # print('Waiting for memeory clear!!!')
    # # # # sleep(1)

    ######################################################################



    ######################################################################
    ###############   Loop based training   ##############################
    ######################################################################

    for eachLoop in range(numberOfTrainingLoop):

        # Clean up the train and process manager list
        trainProcessList[:] = []
        validationProcessList[:] = []

        # Prepeare the intermediate arrays
        data_process = mp.Process(target=IntermediatePredictionProcess, args = (trainingFileName,trainStr))
        data_process.start()
        data_process.join()

        print('Waiting for memeory clear!!!')
        sleep(1)

        print('Converting the Train Manager list to normal lists.....')
        trainNormalList = list(trainProcessList)
        print('List converted!!!')

        # Prepare the final lists of train and validation data
        print('Prepering the Training individual lists')

        XTrain = np.array([x[0] for x in trainNormalList])
        print('Finished XTrain Array!!!')
        decoderTrainInput = np.array([x[1] for x in trainNormalList])
        print('Finished decoderTrainInput Array!!!')
        YClassTrain = np.array([x[2] for x in trainNormalList])
        print('Finished YClassTrain Array!!!')
        YVelTrain = np.array([x[3] for x in trainNormalList])
        print('Finished YVelTrain Array!!!')
        YPoseTrain = np.array([x[4] for x in trainNormalList])
        print('Finished YPoseTrain Array!!!')

        # Reshape the velocity train array
        currntShape = YVelTrain.shape
        YVelTrain = YVelTrain.reshape(currntShape[0],currntShape[1],1)

        # Prepeare the intermediate arrays
        data_process = mp.Process(target=IntermediatePredictionProcess, args = (validationFileName,validationStr))
        data_process.start()
        data_process.join()
        
        print('Waiting for memeory clear!!!')
        sleep(1)

        print('Converting the Validation Manager list to normal lists.....')
        valNormalList = list(validationProcessList)
        print('List converted!!!')

        # Prepare the final lists of train and validation data
        print('Prepering the Training individual lists')

        XVal = np.array([x[0] for x in valNormalList])
        print('Finished XTrain Array!!!')
        decoderValInput = np.array([x[1] for x in valNormalList])
        print('Finished decoderTrainInput Array!!!')
        YClassVal = np.array([x[2] for x in valNormalList])
        print('Finished YClassTrain Array!!!')
        YVelVal = np.array([x[3] for x in valNormalList])
        print('Finished YVelTrain Array!!!')
        YPoseVal = np.array([x[4] for x in valNormalList])
        print('Finished YPoseTrain Array!!!')

        # Reshape the velocity train array
        currntShape = YVelVal.shape
        YVelVal = YVelVal.reshape(currntShape[0],currntShape[1],1)

        print('Second rouund data preperation fininshed!!!')

        # Loop training process (last param True, to load the weights)
        training_process = mp.Process(target=_training_worker, args = (XTrain,decoderTrainInput,YClassTrain,YVelTrain,YPoseTrain,secondNumberEpochs,XVal,decoderValInput,YClassVal,YVelVal,YPoseVal,True))
        training_process.start()
        training_process.join()

        print('Waiting for memeory clear!!!')
        sleep(1)

    ######################################################################


    ######################################################################
    ###############  Final Prediction ####################################
    ###################################################################### 

     
    # Calculate the final error    
    print('Print the final error!!!!')
    data_process = mp.Process(target=IntermediatePredictionProcess, args = (validationFileName,validationStr))
    data_process.start()
    data_process.join()
    
    print('Waiting for memeory clear!!!')
    sleep(5)

    ###################################################################### 