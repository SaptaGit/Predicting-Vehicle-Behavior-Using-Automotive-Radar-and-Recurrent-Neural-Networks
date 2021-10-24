########################################################################################################################
# Orginal File copied from the FilterNetworkV20.py (Junction folder) from this folder. For any issues compare with the mentioned file.
########################################################################################################################
# This version is only for data creation with the I-80 or US-101 dataset and to be made public
# Two different csv files are used to create sepearte train, val and test vehicles, so that both TV and SVs can bes added
# for both input/output as well as RMSE calculations 
# Remove the dist from the junction feature map and use the maneuver class as lane change -> Done
# Use the last 3 and next 3 frames lane number to identify the lane change maneuver  -> Done
# To be done 
# Increase frame count for lane change estimation and ignore vehicles with zero lane changes-> Done
# Remove the unused functions once the results are close....
# Just use a flag and set it to true when there is a lane change happend, and use that flag at the end to add or ignore sample append -> Done
# Move the vanilla data V2 first.
########################################################################################################################
############################### Chanages from local to server and vice versa ###########################################
# 1. keras to tensorflow.keras, 
# 2. .compact.v1 for both config proto and warning log in training_worker_
# 3. .fit one with the workers and multi processing arguments and not fit_generator
# 4. batchSize 2048 from 256
# 5. server folder and lankershim.csv file path and model paths
# 6. ValidationSetProcess numberofCores 2 to 17 
# 7. Core number (os.system('1,2,3,4..... os.getpid()')) do it for both validationSetProcess and training_worker based on other running files
# 8. epochs from 4 to 40
# 9. Increase vehicle Count for train and val before data preepration
# 10. Startght to Straight check
# 11. Use pool map
# 12. Increase number of core in the TrainData function
######################################################################################################################## 
# Sen server 1st terminal
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"  #5
import numpy as np
import sys
import math
from scipy.io import savemat
import csv
from math import log, exp, tan, atan, pi, ceil, cos, sin
import random
import multiprocessing as mp
from multiprocessing import Process, Manager
import time
from time import sleep
# import PFHelper

# multiprocessing.set_start_method('spawn', True)

# # # gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
# # # sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# Specify the test trajectory csv file
# Path for local folder
# testTrajFilePath = '/home/saptarshi/PythonCode/FilterNet/data/I80RawData/0400pm-0415pm/trajectories-0400-0415.csv'   # Straight to starigh check.......
# testTrajFilePath = '/home/saptarshi/PythonCode/FilterNet/data/I80RawData/0400pm-0415pm/i80Small.csv'
# Path for Sen Server trajectory csv file
# testTrajFilePath = '/media/disk1/sap/FilterNet/data/I80RawData/0400pm-0415pm/trajectories-0400-0415.csv'
# Path for Big Screen Server trajectory csv file
trainTrajFilePath = '/media/sdd/sap/Junction/I80RawData/0400pm-0415pm/trajectories-0400-0415.csv'
valTrajFilePath = '/media/sdd/sap/Junction/I80RawData/0515pm-0530pm/trajectories-0515-0530.csv' 
testTrajFilePath = '/media/sdd/sap/Junction/I80RawData/0500pm-0515pm/trajectories-0500-0515.csv' 
# Path for small Server trajectory csv file
# testTrajFilePath = '/home/sap/PythonCode/FilterNet/data/I80RawData/0400pm-0415pm/trajectories-0400-0415.csv'

# Specify if process the data or read the processed data
#  'read' -> Read Data and train;  'process' -> Process data ;'predictStr'-> Only load the validation and predict (to save memory);
readStr = 'read'
processStr = 'process'
predictValStr = 'predictVal'
processOrRead = processStr

# Specify the folder name for the sample to read/write based on the above flag (DO NOT ADD '/' AT THE END)
# Path for local folder
# folderName = '/home/saptarshi/PythonCode/FilterNet/data/I80Surrounding4DataV2'   
# Path for Sen Server folder
# folderName ='/media/disk5/sap/FilterNet/data/I80MoreLaneV5'         # I80MoreLaneV4          
# Path for Big Screen Server folder
folderName = '/media/sdd/sap/FilterNet/data/I80TrainValTestMixed2048V2'  #   I80Surrounding4MoreLaneV1'  I804SurroundingMixed1024V1
# Path for small Server folder
# folderName = '/home/sap/PythonCode/FilterNet/data/I80MoreLaneV4'

# Sepcify the file paths for the encoder decoder model to save
# Path for local folder
# encoderModelFilename = '/home/saptarshi/PythonCode/FilterNet/ServerModels/V4FilterEncoder9.h5'
# decoderModelFilename = '/home/saptarshi/PythonCode/FilterNet/ServerModels/V4FilterDecoder9.h5'
# Path for Sen server folder
# # encoderModelFilename = '/media/disk1/sap/FilterNet/models/V4FilterEncoder10.h5'  # keep the same model names.....
# # decoderModelFilename = '/media/disk1/sap/FilterNet/models/V4FilterDecoder10.h5'
# Path for Big Screen Server folder
encoderModelFilename = '/home/sap/Sap/Junction/Encoder13.h5'
decoderModelFilename = '/home/sap/Sap/Junction/Decoder13.h5'

# Specify the result file to store each sample error
resultFileName = '/media/sdd/sap/Junction/results/I80DataPrepRun2.txt'
resultFileObj = open(resultFileName, 'x')
resultFileObj.close()


# Specify the training vehicle file name
trainingFileName = folderName + '/' + 'training.txt'
# Specify the validation vehicle file name
validationFileName = folderName + '/' + 'validation.txt'
# Specify the test vehicle file name
testingFileName = folderName + '/' + 'test.txt'

# Train and validation data folder names (DO NOT CHANGE)
trainFolderName = '/TrainData'
valFolderName = '/ValidationData'
testFolderName = '/TestData'

# Create the visible window
# cv2.namedWindow('test', cv2.WINDOW_NORMAL)

##########################################
# For actual manager  list
# Don't forget to convert the pool function
#########################################
# Train and Validation process lists
# No need to maintain train and validation seperate manager list
# as now it called seperatly. Make sure to clean up after each call
manager = Manager()
generalProcessList = manager.list()
# validationProcessList = manager.list()

# To keep count of number of sample processed and samples ignore due to coordinate transfer
countList = manager.list()
ignoreSampleCountList = manager.list()

# Add 20-25 straight vehicles to see the histogram and other distributions
straightVehicles = manager.list()
includedStraightVehicles = 10

# Manager lists for errors
trajErrorList = manager.list()               # Final error list to hold the error for the predicted pose normally from the network
motionErrorList = manager.list()             # Final error list to hold the error for the predicted pose estimated using the predicted motion
positionMotionFilterError = manager.list()   # Final error list to hold the error through the filter using both predicted pose and motion 
positionMotionFilterVar = manager.list()     # Final variance list to hold both X and Y variance output from the filter using both position and motion
onlyPositionFilterError = manager.list()     # Final error list to hold the error through the filter using only predicted pose and last true motion 
onlyPositionFilterVar = manager.list()       # Final variance list to hold both X and Y variance output from the filter using only position
nonReGenFilterError = manager.list()         # Final error list to hold the error through the filter using both predicted pose and motion without particle regeneration
nonReGenFilterVar = manager.list()           # Final variance list to hold both X and Y variance output from the filter with low and constant regeneration variance
sampleCountList = manager.list()
###############################################

##########################################
# For local debugging normal list
# Don't forget to convert the pool function
#########################################
# # # # For local testing normal lists
# # # # Train and Validation process lists
# # # trainProcessList = []
# # # validationProcessList = []

# # # # To keep count of number of sample processed
# # # countList = []

# # # # Manager lists for errors
# # # trajErrorList = []                 # Final error list to hold the error for the predicted pose normally from the network
# # # motionErrorList = []               # Final error list to hold the error for the predicted pose estimated using the predicted motion
# # # positionMotionFilterError = []     # Final error list to hold the error through the filter using both predicted pose and motion 
# # # onlyPositionFilterError = []       # Final error list to hold the error through the filter using only predicted pose and last true motion 
# # # sampleCountList = []
###############################################

# Model parametrs 
batchSize = 2048  #   # 2048   #2048    #2048    #2048     #128    ##
nepochs = 60   # 40   #30 #30
historyTemporal = 30   #30
futureTemporal = 50   #50
surroudingCarCounts = 4
inputFeatureCount = 6  # 6 -> [localX,localY,velocityX,velocityY,laneID,movement]
globalInputFeatures = (surroudingCarCounts+1)*inputFeatureCount  

# These feature counts are for the modified encoder decoder with surrounding vehicles in the decoder
globalOutputFeatures = 7                          # 6 -> (poseX,poseY,velocityX,velocityY,Class0,Class1,Class2)
decoderFeatureCount = 7 # output 
globalDecoderFeatures = (surroudingCarCounts+1)*decoderFeatureCount 

# These feature counts are for the vanilla encoder decoder without surrounding vehicles in the decoder
# # # globalOutputFeatures = 2                          # 2 -> (poseX,poseY)
# # # decoderFeatureCount = 2 # output 
# # # globalDecoderFeatures = globalOutputFeatures

leakyAlphaValue = 0.1  # (0.3 -> Leaky ReLU)
maximumAllowabelJuncDist = 300
intersectionJuncDist = 350
maximumSurroundingCarDist = 80 # as this is straight road increaed the surroudning dist     40     #(25 Feet)
predictionDistanceThreshold = 250  #  250  #(100 Feet )
ignoreFrameCount = 100

# Number of previous and next frames to be considered for the maneuver estimatation
prevNextFrameCount = 20   # Do not change as the same value will be used for the retrain technique

# Input and decoder padding for non eligible surrpounding vehicles
inputZeroPadding = [0,0,0,0,0,0]
decoderZeroPadding = [0,0,0,0,0,0,0]

# Network params
classOut = 3
poseOut = 2
velcoityOut = 2
n_units = 256


# Train, Test and Validation vehiles
trainVehileCount = 1700   
validationVehicleCount = 400
testVehicleCount = 450

# Min Max values for normalize or denormalize
minLocalY = 0    # remove the extras.....
maxLocalY = 0
minLocalX = 0
maxLocalX = 0
minVel = 0
maxVel = 0
minVelocityX = 999
maxVelocityX = -999
minVelocityY = 999
maxVelocityY = -999
maxRealtiveX = -9999
maxRealtiveY = -9999
minRealtiveX = 9999
minRealtiveY = 9999
minAcc = 999
maxAcc = -999
minHeadwaySpace = 999
maxHeadwaySpace = -999
minHeadwayTime = 999
maxHeadwayTime = -999

# Index of different features in the csv file
vechileIDIndex = 0
frameIDIndex = 1
totoalFrameIndex = 2
globalTimeIndex = 3
localXIndex = 4
localYIndex = 5
gobalXIndex = 6
gobalYIndex = 7
velocityIndex = 11
accIndex = 12
laneIDIndex = 13
headwaySpaceIndex = 16
headwayTimeIndex = 17


# String Constants 
inputStr = 'Input'
decoderStr = 'Decoder'
trainStr = 'Train'
validationStr = 'Validation'
testStr = 'Test'

# File types for each sample
sampleFileNameList = ['/finalX.txt','/finalDecoderInput.txt','/finalYClass.txt','/finalYVel.txt','/finalYPose.txt']  # Commentning out for vanialla
# sampleFileNameList = ['/finalX.txt','/finalDecoderInput.txt','/finalYPose.txt']  

# Class strings
straightStr = 'Straight'
leftTurnStr = 'Left Turn'
rightTurnStr = 'Right Turn'

# Unit constants
feetToMeter = 0.3048

# csv feature count
csvFeatureCount = 18    #for I-80 18, for Lankershim 24

# Make the frame dictionary global for use during prediction
dictByFrames = dict()
# Make the Vehicle dictionary global for use multi processing
dictByVehicles = dict()
# Make the mapper dict global
# Create Dictionary for Mapper
mapper = dict()


# Custome Loss function
def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    """
    from keras import backend as K
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

# Covert the movement float to movementStr for confusion calculation
def MovementToStr(movementFloat):
    returnStr = ''
    if(movementFloat == 0):
        returnStr = straightStr
    elif(movementFloat == 0.5):
        returnStr = leftTurnStr
    elif(movementFloat == 1.0):
        returnStr = rightTurnStr
    else:
        print('Unknown movement float in MovementToStr func!!!')
        print('Received movement float value is : ' + str(movementFloat))
        sys.exit()
        
    return returnStr

# Calculate the confusion matrix
def CalcConfusionMatrix(predList):
    straightArray = np.array([0,0,0])
    leftTurnArray = np.array([0,0,0])
    rightTurnArray = np.array([0,0,0])

    for eachPredValues in predList:
        gtValue = eachPredValues[0]
        predValue = eachPredValues[1]
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

    # Normalize each feature columns except localX and localY for relative movement
    # # normalizeIndexList = [velocityIndex,laneIDIndex,directionIndex,movementIndex,accIndex,headwaySpaceIndex,headwayTimeIndex]
    normalizeIndexList = [velocityIndex,laneIDIndex,accIndex,headwaySpaceIndex,headwayTimeIndex]

    # Save the original min max value for further denormalization
    # global minLocalX,maxLocalX,minLocalY,maxLocalY,minAcc,maxAcc,minHeadwaySpace,maxHeadwaySpace,minHeadwayTime,maxHeadwayTime
    global minLocalX,maxLocalX,minLocalY,maxLocalY,minAcc,maxAcc,minLaneIDVals,maxLaneIDVals,minHeadwaySpace,maxHeadwaySpace,minHeadwayTime,maxHeadwayTime

    minLocalX = min(datasetArray[:,localXIndex])
    maxLocalX = max(datasetArray[:,localXIndex])

    minLocalY = min(datasetArray[:,localYIndex])
    maxLocalY = max(datasetArray[:,localYIndex])

    minVel = min(datasetArray[:,velocityIndex])
    maxVel = max(datasetArray[:,velocityIndex])

    minAcc = min(datasetArray[:,accIndex])
    maxAcc = max(datasetArray[:,accIndex])

    minLaneIDVals = min(datasetArray[:,laneIDIndex])
    maxLaneIDVals = max(datasetArray[:,laneIDIndex])

    minHeadwaySpace = min(datasetArray[:,headwaySpaceIndex])
    maxHeadwaySpace = max(datasetArray[:,headwaySpaceIndex])

    minHeadwayTime = min(datasetArray[:,headwayTimeIndex])
    maxHeadwayTime = max(datasetArray[:,headwayTimeIndex])

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

    loadFile.close()

    return dictionaryByFrames,dictionaryByVehicles,mapper


# Estimate the lane change maneuver based on the passed list of lane ids for target vehicle
# Straight -> 0 lef lane chagne -> 0.5 right lane change 1.0
def TargetLaneChanageManeuver(laneIDList):
    similarIDCheck = True
    laneIdListLen = len(laneIDList)
    # Intitialize with 0 
    maneuverInfoRet = 0

    # All IDs are same means, Else chek item by item if current and prev items are same for all means straight 
    # Dring current and prev comparison, if mis match check is prev is greater or less than current
    # if current greater than prev right lane change else left lane change
    for laneIdx in range(1,laneIdListLen):
        prevLaneId = laneIDList[laneIdx-1]
        currentLaneID = laneIDList[laneIdx]
        if(prevLaneId != currentLaneID):
            if(currentLaneID > prevLaneId):
                # Right lane change
                maneuverInfoRet = 1.0
                return maneuverInfoRet
            elif(currentLaneID < prevLaneId):
                # left lane change
                maneuverInfoRet = 0.5
                return maneuverInfoRet
            else:
                # Since this one only for mis match it should always ne either greater or less not equall
                print('Lane ID based maneuver check faild with unexpected behaviour!!!')
                nanVal = 10000/0
                sys.exit()
    
    # If noo of the mismatch checked passed means all are same ID, means straight retrun straighht maneuver
    return maneuverInfoRet


# For surrounding we only have the current and last frame only. So only use those at the moment for lanne change estimation
def SurrLaneChanageManeuver(surrLaneIDList):
    surrPrevLaneID = surrLaneIDList[0]
    surrCurrentLaneID = surrLaneIDList[1]

    surrManeuverInfoRet = 99

    # If they are equall means straight, if current is greater than prev right lane change and if current is less than prev left lane change
    if(surrPrevLaneID == surrCurrentLaneID):
        surrManeuverInfoRet = 0.0
    elif(surrCurrentLaneID > surrPrevLaneID):
        surrManeuverInfoRet = 1.0
    elif(surrCurrentLaneID < surrPrevLaneID):
        surrManeuverInfoRet = 0.5
    else:
        # It shoud be either of these combo nothing else
        print('Surroudning vehicle manneuver estmiation failed!!!')
        nanVal = 10000/0
        sys.exit()

    # Surroudning maneuver should be not 99
    if(surrManeuverInfoRet == 99):
        print('Surroudning vehicle manneuver no check passed !!!')
        nanVal = 10000/0
        sys.exit()

    return surrManeuverInfoRet


# Pass the surrounding vechiles and current input list. It will extend the list with surrouding cars info.
def GetSurroundingCarsInfo(otherVechiles, prevOtherVehicles, nextOtherVehicles, tempInput, targetVehicleID, inputOrDecoder, localX, localY, initialX, initialY):

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

            # Get the other vehicle current laneID
            otherCurrentLaneID = eachOtherVechiles[laneIDIndex]

            # # # # Get the other velocity from the csv
            # # # otherVelocity = eachOtherVechiles[velocityIndex]

            # Calculate the other velocity using prev or next pose
            # Initialize the otherVelocityX and otherVelocityY withh -9999 for scope and check if it actually happened correctly
            otherVelocityX = -9999
            otherVelocityY = -9999
            # Initialize the othermaneuver -9999 for scope and check if it actually happened correctly
            otherManeuver = -9999
            # Search the same ID in the prevOtherVehicles list to get the previous local pose
            for eachOtherItem in prevOtherVehicles:
                otherCheckId = str(eachOtherItem[vechileIDIndex])
                if(otherVehicleID == otherCheckId):
                    prevOtherLocalPoseX = eachOtherItem[localXIndex]
                    prevOtherLocalPoseY = eachOtherItem[localYIndex]
                    # Other velocity is not that important, use absolute for Y as always forward
                    # Keep the sign for X, don't bother which side, as the combined direction field in the data along with postive//negetive velocityX
                    # will hopefully handle the combination
                    otherVelocityX = otherLocalX-prevOtherLocalPoseX          
                    otherVelocityY = abs(otherLocalY-prevOtherLocalPoseY)
                    # Find the lane ID of the same othher vehicle for previous frame
                    otherPrevLaneID = eachOtherItem[laneIDIndex]
                    # Estimate the lane change maneuver for the surroudning vehice
                    otherLaneIDList = [otherPrevLaneID,otherCurrentLaneID]
                    otherManeuver = SurrLaneChanageManeuver(otherLaneIDList)
                    # Once found break from the curret for loop
                    break
            # If OtherVelocityX and OtherVelocityY still -9999 means vehicle not found in prev frames
            # Check the nextFrames to get the same other vehicle
            if(otherVelocityX == -9999 or otherVelocityY == -9999 or otherManeuver == -9999):
                # Search the same ID in the nextOtherVehicles list to get the next local pose
                for eachOtherItem in nextOtherVehicles:
                    otherCheckId = str(eachOtherItem[vechileIDIndex])
                    if(otherVehicleID == otherCheckId):
                        nextOtherLocalPoseX = eachOtherItem[localXIndex]
                        nextOtherLocalPoseY = eachOtherItem[localYIndex]
                        # Other velocity is not that important, use absolute for Y as always forward
                        # Keep the sign for X, don't bother which side, as the combined direction field in the data along with postive//negetive velocityX
                        # will hopefully handle the combination
                        otherVelocityX = nextOtherLocalPoseX-otherLocalX       
                        otherVelocityY = abs(nextOtherLocalPoseY-otherLocalY)
                        # Find the lane ID of the same other vehicle for next frame
                        otherNextLaneID = eachOtherItem[laneIDIndex]
                        # Estimate the lane change maneuver for the surroudning vehice
                        otherLaneIDList = [otherCurrentLaneID,otherNextLaneID]
                        otherManeuver = SurrLaneChanageManeuver(otherLaneIDList)
                        # Once found break from the curret for loop
                        break

            # After going through both prev and next frames the otherVelocity should be populated
            # Final check for not -9999 else exit
            if(otherVelocityX == -9999 or otherVelocityY == -9999 or otherManeuver == -9999):
                print('In GetSurroundingInfo otherVelocity or other maneuver not pupulated successfully!!!')
                print('OtherVelocityX is still : ' + str(otherVelocityX))
                print('OtherVelocityY is still : ' + str(otherVelocityY))
                print('OtherManeuver is still : ' + str(otherManeuver))
                nanVal = 1000/0
                sys.exit()

            otherLaneID = eachOtherVechiles[laneIDIndex]
            

            # For I-80 it is always moving forward
            otherRelativeX = otherLocalX-initialX
            otherRelativeY = otherLocalY-initialY

            # If the other vehicle distance is more that allowable then append zeros
            otherDist = math.sqrt((((otherLocalX-localX)**2)+((otherLocalY-localY)**2)))

            # If target vehicles and other vehicles are opposite side ignore
            # Check by matching the signs, Multiply the values, if the sign is negetive means not matching
            # cause +*+=+ or -*-=+
            # # # whichSideSign = otherLocalX*initialX

            if(inputOrDecoder == inputStr):
                if((otherDist>maximumSurroundingCarDist)):
                    tempInput.extend(inputZeroPadding)
                else:
                    # Safety check otherRealtiveY should not be less than -60
                    if(otherRelativeY<-(maximumSurroundingCarDist+20)):
                        print('Other RelativeY is :' + str(otherRelativeY))
                        nanVal = 1000/0
                        sys.exit()
                    # Traget append main input tempInput = [localX,localY,velocityX,velocityY,laneID,movement]
                    # Without the headway space and time
                    # # tempInput.extend([otherRelativeX,otherRelativeY,otherVelocityX,otherVelocityY,otherLaneID,otherManeuver])
                    # With the headway space and time
                    tempInput.extend([otherRelativeX,otherRelativeY,otherVelocityX,otherVelocityY,otherLaneID,otherManeuver])
            elif(inputOrDecoder == decoderStr):
                if((otherDist>maximumSurroundingCarDist)):
                    tempInput.extend(decoderZeroPadding)
                else:
                    # Safety check otherRealtiveY should not be less than -60
                    if(otherRelativeY<-(maximumSurroundingCarDist+20)):
                        print('Other RelativeY is :' + str(otherRelativeY))
                        nanVal = 1000/0
                        sys.exit()
                    lastInputClassInfo = MovementToClassForm(otherManeuver)
                    tempInput.extend([otherRelativeX,otherRelativeY,otherVelocityX,otherVelocityY,lastInputClassInfo[0],lastInputClassInfo[1],lastInputClassInfo[2]])
            else:
                print('Unknown inputOrDecoder string : ' +  inputOrDecoder)
                sys.exit()
        
        # Append the zero padding based on the required padding count calculated using other vechile count and decided surrouning car count
        # As the number of input features and decoder input/output is same that is why we used the same zero padding width
        zeroList = []
        if(inputOrDecoder == inputStr):
            zeroList = inputZeroPadding
        elif(inputOrDecoder == decoderStr):
            zeroList = decoderZeroPadding
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

            # # # # Using the direct velocity in the csv
            # # # otherVelocity = otherVechiles[int(eachReleventIndex)][velocityIndex]

            # Calculating the VelocityX and velocityY for the surrounding vehicle
            # Check if the vehicle ID exists in the prev timestamp. Dont worry about the ID is orginal or mapped
            # As otherVehicles prevotherVehicles and nextotherVehicles all three is prepeared using frame dict, so Ids will be same (orginial) 
            otherOriginalId = str(otherVechiles[int(eachReleventIndex)][vechileIDIndex])
            # Initialize the otherVelocityX and otherVelocityY withh -9999 for scope and check if it actually happened correctly
            otherVelocityX = -9999
            otherVelocityY = -9999
            # Initialize the othermaneuver -9999 for scope and check if it actually happened correctly
            otherManeuver = -9999
            # Get the other vehicle current laneID
            otherCurrentLaneID = otherVechiles[int(eachReleventIndex)][laneIDIndex]

            # Search the same ID in the prevOtherVehicles list to get the previous local pose
            for eachOtherItem in prevOtherVehicles:
                otherCheckId = str(eachOtherItem[vechileIDIndex])
                if(otherOriginalId == otherCheckId):
                    prevOtherLocalPoseX = eachOtherItem[localXIndex]
                    prevOtherLocalPoseY = eachOtherItem[localYIndex]
                    # Other velocity is not that important, use absolute for Y as always forward
                    # Keep the sign for X, don't bother which side, as the combined direction field in the data along with postive//negetive velocityX
                    # will hopefully handle the combination
                    otherVelocityX = otherLocalX-prevOtherLocalPoseX          
                    otherVelocityY = abs(otherLocalY-prevOtherLocalPoseY)
                    # Find the lane ID of the same othher vehicle for previous frame
                    otherPrevLaneID = eachOtherItem[laneIDIndex]
                    # Estimate the lane change maneuver for the surroudning vehice
                    otherLaneIDList = [otherPrevLaneID,otherCurrentLaneID]
                    otherManeuver = SurrLaneChanageManeuver(otherLaneIDList)
                    # Once found break from the curret for loop
                    break
            # If OtherVelocityX and OtherVelocityY still -9999 means vehicle not found in prev frames
            # Check the nextFrames to get the same other vehicle
            if(otherVelocityX == -9999 or otherVelocityY == -9999 or otherManeuver == -9999):
                # Search the same ID in the nextOtherVehicles list to get the next local pose
                for eachOtherItem in nextOtherVehicles:
                    otherCheckId = str(eachOtherItem[vechileIDIndex])
                    if(otherOriginalId == otherCheckId):
                        nextOtherLocalPoseX = eachOtherItem[localXIndex]
                        nextOtherLocalPoseY = eachOtherItem[localYIndex]
                        # Other velocity is not that important, use absolute for Y as always forward
                        # Keep the sign for X, don't bother which side, as the combined direction field in the data along with postive//negetive velocityX
                        # will hopefully handle the combination
                        otherVelocityX = nextOtherLocalPoseX-otherLocalX       
                        otherVelocityY = abs(nextOtherLocalPoseY-otherLocalY)
                        # Find the lane ID of the same other vehicle for next frame
                        otherNextLaneID = eachOtherItem[laneIDIndex]
                        # Estimate the lane change maneuver for the surroudning vehice
                        otherLaneIDList = [otherCurrentLaneID,otherNextLaneID]
                        otherManeuver = SurrLaneChanageManeuver(otherLaneIDList)
                        # Once found break from the curret for loop
                        break

            # After going through both prev and next frames the otherVelocity should be populated
            # Final check for not -9999 else exit
            if(otherVelocityX == -9999 or otherVelocityY == -9999 or otherManeuver == -9999):
                print('In GetSurroundingInfo otherVelocity not pupulated successfully!!!')
                print('OtherVelocityX is still : ' + str(otherVelocityX))
                print('OtherVelocityY is still : ' + str(otherVelocityY))
                print('OtherManeuver is still : ' + str(otherManeuver))
                nanVal = 1000/0
                sys.exit()

            otherLaneID = otherVechiles[int(eachReleventIndex)][laneIDIndex]

            # If IntialX positive means right side, means increasing Y else left side, decreasing Y, similar for the positive or negetive X based on side
            # Right bent positive and left bent negetive
            # For I-80 it always going forward
            otherRelativeX = otherLocalX-initialX
            otherRelativeY = otherLocalY-initialY

            # # # Nearest junction distance
            # # currentSection =  otherVechiles[int(eachReleventIndex)][sectionIndex]
            # # currentIntersection = otherVechiles[int(eachReleventIndex)][intersectionIndex]
            # # juncDist = CalculateNearestJuncLoc(currentSection, currentIntersection, otherLocalX, otherLocalY)

            # If the other vehicle distance is more than allowable then append zeros
            otherDist = math.sqrt((((otherLocalX-localX)**2)+((otherLocalY-localY)**2)))

            # # # # # If target vehicles and other vehicles are opposite side ignore
            # # # # # Check by matching the signs, Multiply the values, if the sign is negetive means not matching
            # # # # # cause +*+=+ or -*-=+
            # # # # whichSideSign = otherLocalX*initialX
            
            if(inputOrDecoder == inputStr):
                if((otherDist>maximumSurroundingCarDist)):
                    tempInput.extend(inputZeroPadding)
                else:
                    # Safety check otherRealtiveY should not be less than -60
                    if(otherRelativeY<-(maximumSurroundingCarDist+20)):
                        print('Other RelativeY is :' + str(otherRelativeY))
                        nanVal = 1000/0
                    # Traget append main input tempInput = [localX,localY,velocityX,velocityY,laneID,movement]
                    # Without the headway space and time
                    # # tempInput.extend([otherRelativeX,otherRelativeY,otherVelocityX,otherVelocityY,otherLaneID,otherManeuver])
                    # With the headway space and time
                    tempInput.extend([otherRelativeX,otherRelativeY,otherVelocityX,otherVelocityY,otherLaneID,otherManeuver])
            elif(inputOrDecoder == decoderStr):
                if((otherDist>maximumSurroundingCarDist)):
                    tempInput.extend(decoderZeroPadding)
                else:
                    # Safety check otherRealtiveY should not be less than -60
                    if(otherRelativeY<-(maximumSurroundingCarDist+20)):
                        print('Other RelativeY is :' + str(otherRelativeY))
                        nanVal = 1000/0
                    lastInputClassInfo = MovementToClassForm(otherManeuver)
                    tempInput.extend([otherRelativeX,otherRelativeY,otherVelocityX,otherVelocityY,lastInputClassInfo[0],lastInputClassInfo[1],lastInputClassInfo[2]])

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
    targetVehicleID = processItme[1]   # original key string

    if(targetVehicleID == None):
        print('Traget Vehicle ID is none!!!!')

    currentVehicleList = dictByVehicles[currentID]


    currentVehicleLength = len(currentVehicleList)

    # loop through the vehicle list to check if there is any single change of laneID. If not means zero lane changes measn ignore
    # Get the intitial lane first to compare
    changeFlag = False
    initialLaneID = currentVehicleList[0][laneIDIndex]
    for eachTargetItem in currentVehicleList:
        currentLaneID = eachTargetItem[laneIDIndex]
        if(currentLaneID != initialLaneID):
            changeFlag = True
            break

    
    # Check if any lane chnage happend, false means no lane chnage happend
    if(changeFlag == False):
        straightVehicles.append(0)
        if(len(straightVehicles) > includedStraightVehicles):
            return


    # Instead of historyTemporal start from (historyTemporal+1). One extra prev pose for the velocity calculate
    # Insted of going till end go till end-1 in case we need next frame for velocity calculation
    # Basically for surrounding vehicle velocity calculation if prev frame missing using next frame
    # Start from +prevNextFrameCount frames to get the last prevNextFrameCount lane ids for lane change estimation
    # And go upto last but leaving prevNextFrameCount for next prevNextFrameCount lane ids to estimate lane change

    for idx in range(historyTemporal+10+prevNextFrameCount,currentVehicleLength-futureTemporal-10-prevNextFrameCount): 

        #################################################################################################
        ########################### Removing this check just to be sure !!!! ############################
        #################################################################################################

        # # # # # Get the current vehicles as those are only eligible from prediction point of view 
        # # # # # vehicles appearing in first frame of the target vehicle
        # # # # currentTargetTime = currentVehicleList[idx-historyTemporal][globalTimeIndex]
        # # # # currentOtherVehicles = dictByFrames[str(currentTargetTime)]
        # # # # currentOtherEligibleVehicles = []
        # # # # for eachCurrentOtherVehicles in currentOtherVehicles:
        # # # #     currentOtherID = eachCurrentOtherVehicles[vechileIDIndex]
        # # # #     # # Update other Id in case it is present in mapper
        # # # #     # if(str(currentOtherID) in mapper):
        # # # #     #     updatedID = mapper[str(currentOtherID)]
        # # # #     #     currentOtherID = updatedID
        # # # #     if(currentOtherID == float(targetVehicleID)):
        # # # #         currentOtherEligibleVehicles.append(currentOtherID)
        # # # #         continue
        # # # #     # vehicles having history + future temporal frames.
        # # # #     currentOtherFrame = eachCurrentOtherVehicles[frameIDIndex]
        # # # #     currentOtherTotalFrame = eachCurrentOtherVehicles[totoalFrameIndex]
        # # # #     remainingFrames = currentOtherTotalFrame - currentOtherFrame
        # # # #     if(remainingFrames>= historyTemporal+futureTemporal):
        # # # #         currentOtherEligibleVehicles.append(currentOtherID)

        #################################################################################################

        # Maintain a flag to check the negetive y local values and ignore those samples
        ignoreSampleFlag = False

        # Prepeare sequential Input Data
        localXData = []
        initialLocalX = currentVehicleList[idx-historyTemporal][localXIndex]
        initialLocalY = currentVehicleList[idx-historyTemporal][localYIndex]

        ##############################################################################
        ############### TRANSFER Learning Part ######################################
        # Identify the intital section and intersection of the sample
        # Based in that decide train or validation sample, not incoming flag based in vehicle ID
        # Also dont forget to mosify the append at the end of the function
        ################################################################################
        # # # # # Identify the initial section or intersection to seperate validation vehicles
        # # # # sampleSectionId = currentVehicleList[idx-historyTemporal][sectionIndex]
        # # # # sampleIntersectionId = currentVehicleList[idx-historyTemporal][intersectionIndex]
        # # # # # Initialize validation Flag as False. If condition satisfy then chnage to true
        # # # # validationFlag = False
        # # # # if(sampleSectionId == 4 or sampleSectionId == 5 or sampleIntersectionId == 4):
        # # # #     validationFlag = True
        ##############################################################################

        for jdx in range(idx-historyTemporal,idx):
            tempInput = []
            absoluteX = currentVehicleList[jdx][localXIndex]
            absoluteY = currentVehicleList[jdx][localYIndex]

            # For I-80 all the vehicles are moving forward.
            localX = absoluteX-initialLocalX
            localY = absoluteY-initialLocalY

            # Safety check localY should not be high negetive
            if(localY<-2):
                ignoreSampleFlag = True
                # # print('localY is :' + str(localY))
                # # print('Re mapped Key : ' + str(currentID))
                # # print('Origina Key : ' + str(targetVehicleID))

            # # # # Old velocity get from the csv file directly
            # # # velocity = currentVehicleList[jdx][velocityIndex]

            # New velocity calculate from the prev pose as VelocityX and velocityY
            prevAbsoluteX = currentVehicleList[jdx-1][localXIndex]
            prevAbsoluteY = currentVehicleList[jdx-1][localYIndex]

            # If IntialX positive means right side, means increasing Y else left side, decreasing Y, similar for the positive or negetive X based on side
            # Right bent positive and left bent negetive
            velocityX = absoluteX-prevAbsoluteX        
            velocityY = absoluteY-prevAbsoluteY

            # Safety check velocityY should not be high negetive
            if(velocityY<-2):
                ignoreSampleFlag = True
                # # print('Velocity Y is :' + str(velocityY))
                # # print('Re mapped Key : ' + str(currentID))
                # # print('Origina Key : ' + str(targetVehicleID))
                # # nanVal = 1000/0

            laneID = currentVehicleList[jdx][laneIDIndex]
            # # # # direction = currentVehicleList[jdx][directionIndex]

            # Get the last prevNextFrameCount and next prevNextFrameCount lane ids to estimate the lane change maneuver 
            lastThreeLaneIds = np.array(currentVehicleList[jdx-prevNextFrameCount:jdx])[:,laneIDIndex]
            nextThreeLaneIds = np.array(currentVehicleList[jdx:jdx+prevNextFrameCount])[:,laneIDIndex]
            totoalLaneIDs = list(lastThreeLaneIds)
            totoalLaneIDs.extend(list(nextThreeLaneIds))

            # Estmiate the lane change maneuver
            movement = TargetLaneChanageManeuver(totoalLaneIDs)       #  currentVehicleList[jdx][movementIndex]

            # Without the headway space and time 
            tempInput = [localX,localY,velocityX,velocityY,laneID,movement]

            # Prepare the surrounding cars information
            # Gather vehicles using the same frame using the Frame Dict
            currentInputFrame = currentVehicleList[jdx][frameIDIndex]
            currentInputTime = currentVehicleList[jdx][globalTimeIndex]
            otherVechiles = dictByFrames[str(currentInputTime)]
            # Get the previous timestamp (-100) other vehicles velcotiy calculations
            prevOtherVehicles = dictByFrames[str(currentInputTime-100)]
            # Get the next timestamp (+100) also in case the vehicle is not present in prev timestamp
            nextOtherVehicles = dictByFrames[str(currentInputTime+100)]

            #################################################################################################
            ########################### Removing this check just to be sure !!!!
            #################################################################################################

            # # # # # # Remove the prediction not eligible vehicles
            # # # # # eligibleOtherVehicles = []
            # # # # # for eachOtherVehicle in otherVechiles:
            # # # # #     otherID = eachOtherVehicle[vechileIDIndex]
            # # # # #     if (otherID in currentOtherEligibleVehicles):
            # # # # #         eligibleOtherVehicles.append(eachOtherVehicle)

            #################################################################################################

            # Remove vehicles with a different global time which is not possible. Just adding check to be sure
            for fdx,eachOtherTime in enumerate(otherVechiles):
                otherTime = eachOtherTime[globalTimeIndex]
                if (otherTime != currentInputTime):
                    print('Mismatch in input global time..')
                    print('other Time ' + str(otherTime))
                    print('Current Time ' + str(currentInputTime))
                    sys.exit()

            # Extend the surrounding cars info into the target vehicles input   otherVechiles replaced by  eligibleOtherVehicles
            ##################################################################################################################
            # Changed from eligibleOtherVehicles -> otherVechiles as the extra checke removed
            # # # tempInput = GetSurroundingCarsInfo(eligibleOtherVehicles, tempInput, targetVehicleID, inputStr, absoluteX, absoluteY, initialLocalX, initialLocalY)
            tempInput = GetSurroundingCarsInfo(otherVechiles, prevOtherVehicles, nextOtherVehicles, tempInput, targetVehicleID, inputStr, absoluteX, absoluteY, initialLocalX, initialLocalY)
            ##################################################################################################################

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
        # Main input structure tempInput = [localX,localY,velocityX,velocityY,laneID,movement]
        lastInput = localXData[-1]
        firstDecoderInput = []
        for tdx in range(0,len(lastInput),inputFeatureCount):
            lastInputPoseX = lastInput[tdx]
            lastInputPoseY = lastInput[tdx+1]
            lastInputVelocityX = lastInput[tdx+2]
            lastInputVelocityY = lastInput[tdx+3]
            lastInputMovement = lastInput[tdx+5]
            lastInputClassInfo = MovementToClassForm(lastInputMovement)
            # Calculate the distance from the junction for the first decoder input 
            # For section, intersection, absoluteX and absoluteY use the last updated varibale as they hold the info for the last frame.
            # # # # juncDist = CalculateNearestJuncLoc(currentSection, currentIntersection, absoluteX, absoluteY)
            firstDecoderInput.extend([lastInputPoseX,lastInputPoseY,lastInputVelocityX,lastInputVelocityY,lastInputClassInfo[0],lastInputClassInfo[1],lastInputClassInfo[2]])


        for kdx in range(idx,idx+futureTemporal):

            # Add the ground truth outputs

            # Get the maneuver info for target vehicle 

            # Get the last prevNextFrameCount and next prevNextFrameCount lane ids to estimate the lane change maneuver 
            lastThreeLaneIds = np.array(currentVehicleList[kdx-prevNextFrameCount:kdx])[:,laneIDIndex]
            nextThreeLaneIds = np.array(currentVehicleList[kdx:kdx+prevNextFrameCount])[:,laneIDIndex]
            totoalLaneIDs = list(lastThreeLaneIds)
            totoalLaneIDs.extend(list(nextThreeLaneIds))

            # Estmiate the lane change maneuver
            nextMovement = TargetLaneChanageManeuver(totoalLaneIDs)       #  currentVehicleList[jdx][movementIndex]
            nextMovementClassData = MovementToClassForm(nextMovement)

            # Get the pose info
            nextLocalX = currentVehicleList[kdx][localXIndex]
            nextLocalY = currentVehicleList[kdx][localYIndex]
            # If IntialX positive means right side, means increasing Y else left side, decreasing Y, similar for the positive or negetive X based on side
            # Right bent positive and left bent negetive
            # For I-80 all vehicles are moving forward
            nextRelativeX = nextLocalX-initialLocalX     
            nextRelativeY = nextLocalY-initialLocalY

             # Safety check velocityY should not be high negetive
            if(nextRelativeY<-2):
                ignoreSampleFlag = True
                # # print('Re mapped Key : ' + str(currentID))
                # # print('Origina Key : ' + str(targetVehicleID))
                # # print('nextRelativeY : ' + str(nextRelativeY))
                # # nanVal = 1000/0


            # Get the velocity
            # # # # Get the velocity directly from the csv         
            # # # nextVelocity = currentVehicleList[kdx][velocityIndex]
            # # # deNormalizedNextVelocity = (nextVelocity*(maxVel-minVel))+minVel

            # Calculate the velocity from the prev pose
            prevLocalPoseX = currentVehicleList[kdx-1][localXIndex]
            prevLocalPoseY = currentVehicleList[kdx-1][localYIndex]
            # If IntialX positive means right side, means increasing Y else left side, decreasing Y, similar for the positive or negetive X based on side
            # Right bent positive and left bent negetive
            # For I-80 all vehicles are movig forward
            nextVelocityX = nextLocalX-prevLocalPoseX    
            nextVelocityY = nextLocalY-prevLocalPoseY

            # Safety check velocityY should not be high negetive
            if(nextVelocityY<-2):
                ignoreSampleFlag = True
                # # print('Re mapped Key : ' + str(currentID))
                # # print('Origina Key : ' + str(targetVehicleID))
                # # print('nextVelocityY : ' + str(nextVelocityY))
                # # nanVal = 1000/0
                

            # Finally append all the values
            localYMovementData.append(nextMovementClassData)
            localYVelData.append([nextVelocityX,nextVelocityY])
            localYPoseData.append([nextRelativeX,nextRelativeY])

            # Add the decoder input
            # # # # Add the distance from the junc in the decoder as well
            # # # nextSection = currentVehicleList[kdx][sectionIndex]
            # # # nextIntersection = currentVehicleList[kdx][intersectionIndex]
            # # # juncDist = CalculateNearestJuncLoc(nextSection, nextIntersection, nextLocalX, nextLocalY)

            decoderTemp = [nextRelativeX,nextRelativeY,nextVelocityX,nextVelocityY,nextMovementClassData[0],nextMovementClassData[1],nextMovementClassData[2]]

            # Prepare the surrounding cars information for decoder input   # for decoder pass only the vehicles present in the last 30 frames..(not done....)
            # Gather vehicles using the same frame using the Frame Dict
            currentInputFrame = currentVehicleList[kdx][frameIDIndex]
            currentInputTime = currentVehicleList[kdx][globalTimeIndex]
            otherVechiles = dictByFrames[str(currentInputTime)]
            # Get the othervehicles from the prev frame to calculate the velocity
            prevOtherVehicles = dictByFrames[str(currentInputTime-100)]
            # Get the othervehicles from the next frame to calculate the velocity (in case a vehicle not present in prev frame)
            nextOtherVehicles = dictByFrames[str(currentInputTime+100)]


            #################################################################################################
            ########################### Removing this check just to be sure !!!! ############################
            #################################################################################################

            # # # # Remove the prediction not eligible vehicles
            # # # # Identify vehicles not in eligible list
            # # # eligibleOtherVehicles = []
            # # # for eachOtherVehicle in otherVechiles:
            # # #     otherID = eachOtherVehicle[vechileIDIndex]
            # # #     if (otherID in currentOtherEligibleVehicles):
            # # #         eligibleOtherVehicles.append(eachOtherVehicle)

            #################################################################################################

            # Remove vehicles with a different global time. Which is not possible. Just to double check
            for gdx,eachOtherTime in enumerate(otherVechiles):
                otherTime = eachOtherTime[globalTimeIndex]
                if (otherTime != currentInputTime):
                    print('Mismatch in decoder global time..')
                    print('other Time ' + str(otherTime))
                    print('Current Time ' + str(currentInputTime))
                    sys.exit()


            # Extend the surrounding cars info into the target vehicles decoder input   ##  otherVechiles replacd by eligibleOtherVehicles 
            #################################################################################################
            # Changed from eligibleOtherVehicles -> otherVechiles as the extra checked removed
            # # # decoderTemp = GetSurroundingCarsInfo(eligibleOtherVehicles, decoderTemp, targetVehicleID, decoderStr, nextLocalX, nextLocalY, initialLocalX, initialLocalY)
            decoderTemp = GetSurroundingCarsInfo(otherVechiles, prevOtherVehicles, nextOtherVehicles, decoderTemp, targetVehicleID, decoderStr, nextLocalX, nextLocalY, initialLocalX, initialLocalY)
            #################################################################################################

            # Check the decoder feature length
            if (len(decoderTemp) != globalDecoderFeatures):
                print('decoderTemp len is : ' + str(len(decoderTemp)) + ' instead of ' + str(globalDecoderFeatures))
                sys.exit()

            # Finally append the target car and surrounding cars info for the current frame into the final decoded input
            decoderInputData.append(decoderTemp)

        # Shift one time stamp right and append Last input at the beggining 
        decoderInputData = decoderInputData[:-1]
        decoderInputData.insert(0,firstDecoderInput)

        ##########################################################################
        ################## NON TRANSFER LEARNING ##################################
        # Append the sample to the train or validation list based on the passed str
        # Append in the final validation or training set based on decided vehicle ID
        # Also check if the sample is ignored
        if(ignoreSampleFlag):
            ignoreSampleCountList.append(0)
            # # # print('In ognore smaple count!!!')
        else:
            generalProcessList.append([localXData,decoderInputData,localYMovementData,localYVelData,localYPoseData])
            # # # # print('generalProcessList Len: ' + str(len(generalProcessList)))
        ##########################################################################

        ##########################################################################
        ####################  TRANSFER LEARNING ##################################
        # Append the sample to the train or validation based on the decided Flag
        # Based on the Intersection and section ID of the sample its decider
        ######################################################################### 
        # # # # # if(validationFlag == True):
        # # # # #     validationProcessList.append([localXData,decoderInputData,localYMovementData,localYVelData,localYPoseData])
        # # # # # elif(validationFlag == False):
        # # # # #     trainProcessList.append([localXData,decoderInputData,localYMovementData,localYVelData,localYPoseData])
        # # # # # else:
        # # # # #     print('Unknown Train Val string')
        # # # # #     sys.exit()
        ######################################################################### 

       
        # Clean up the processed list.. (not a good idea....)
        # localXData = []
        # decoderInputData = []
        # localYMovementData = []
        # localYVelData = []
        # localYPoseData = []

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

# Data preparation for vanilla data woth surroudning and auxialliary steps
def ProcessByVehicleForVanilla(processItme):

    currentID = processItme[0]   # Updated key string
    targetVehicleID = processItme[1]   # original key string

    if(targetVehicleID == None):
        print('Traget Vehicle ID is none!!!!')

    currentVehicleList = dictByVehicles[currentID]


    currentVehicleLength = len(currentVehicleList)

    # Instead of historyTemporal start from (historyTemporal+1). One extra prev pose for the velocity calculate
    # Insted of going till end go till end-1 in case we need next frame for velocity calculation
    # Basically for surrounding vehicle velocity calculation if prev frame missing using next frame
    # Start from +4 frames to get the last 3 lane ids for lane change estimation
    # And go upto last but leaving 4 for next 4 lane ids to estimate lane change

    for idx in range(historyTemporal+4,currentVehicleLength-futureTemporal-4): 

        # Maintain a flag to check the negetive y local values and ignore those samples
        ignoreSampleFlag = False

        # Prepeare sequential Input Data
        localXData = []
        initialLocalX = currentVehicleList[idx-historyTemporal][localXIndex]
        initialLocalY = currentVehicleList[idx-historyTemporal][localYIndex]

        ##############################################################################
        ############### TRANSFER Learning Part ######################################
        # Identify the intital section and intersection of the sample
        # Based in that decide train or validation sample, not incoming flag based in vehicle ID
        # Also dont forget to mosify the append at the end of the function
        ################################################################################
        # # # # # Identify the initial section or intersection to seperate validation vehicles
        # # # # sampleSectionId = currentVehicleList[idx-historyTemporal][sectionIndex]
        # # # # sampleIntersectionId = currentVehicleList[idx-historyTemporal][intersectionIndex]
        # # # # # Initialize validation Flag as False. If condition satisfy then chnage to true
        # # # # validationFlag = False
        # # # # if(sampleSectionId == 4 or sampleSectionId == 5 or sampleIntersectionId == 4):
        # # # #     validationFlag = True
        ##############################################################################

        for jdx in range(idx-historyTemporal,idx):
            tempInput = []
            absoluteX = currentVehicleList[jdx][localXIndex]
            absoluteY = currentVehicleList[jdx][localYIndex]

            # For I-80 all the vehicles are moving forward.
            localX = absoluteX-initialLocalX
            localY = absoluteY-initialLocalY

            # Safety check localY should not be high negetive
            if(localY<-2):
                ignoreSampleFlag = True
                # # print('localY is :' + str(localY))
                # # print('Re mapped Key : ' + str(currentID))
                # # print('Origina Key : ' + str(targetVehicleID))

            # # # # Old velocity get from the csv file directly
            # # # velocity = currentVehicleList[jdx][velocityIndex]

            # New velocity calculate from the prev pose as VelocityX and velocityY
            prevAbsoluteX = currentVehicleList[jdx-1][localXIndex]
            prevAbsoluteY = currentVehicleList[jdx-1][localYIndex]

            # If IntialX positive means right side, means increasing Y else left side, decreasing Y, similar for the positive or negetive X based on side
            # Right bent positive and left bent negetive
            velocityX = absoluteX-prevAbsoluteX        
            velocityY = absoluteY-prevAbsoluteY

            # Safety check velocityY should not be high negetive
            if(velocityY<-2):
                ignoreSampleFlag = True
                # # print('Velocity Y is :' + str(velocityY))
                # # print('Re mapped Key : ' + str(currentID))
                # # print('Origina Key : ' + str(targetVehicleID))
                # # nanVal = 1000/0

            laneID = currentVehicleList[jdx][laneIDIndex]
            # # # # direction = currentVehicleList[jdx][directionIndex]

            # Get the last 3 and next 3 lane ids to estimate the lane change maneuver 
            lastThreeLaneIds = np.array(currentVehicleList[jdx-3:jdx])[:,laneIDIndex]
            nextThreeLaneIds = np.array(currentVehicleList[jdx:jdx+3])[:,laneIDIndex]
            totoalLaneIDs = list(lastThreeLaneIds)
            totoalLaneIDs.extend(list(nextThreeLaneIds))

            # Estmiate the lane change maneuver
            movement = TargetLaneChanageManeuver(totoalLaneIDs)       #  currentVehicleList[jdx][movementIndex]


            # csvVelocity = currentVehicleList[jdx][velocityIndex]
            # csvAccVal = currentVehicleList[jdx][accIndex]
            # # csvHeadwayDistVal = currentVehicleList[jdx][headwaySpaceIndex]
            # # csvHeadwayTimeVal = currentVehicleList[jdx][headwayTimeIndex]

            # # # Nearest junction distance
            # # currentSection = currentVehicleList[jdx][sectionIndex]
            # # currentIntersection = currentVehicleList[jdx][intersectionIndex]
            # # juncDist = CalculateNearestJuncLoc(currentSection, currentIntersection, absoluteX, absoluteY)

            tempInput = [localX,localY,velocityX,velocityY,laneID,movement]

            # Prepare the surrounding cars information
            # Gather vehicles using the same frame using the Frame Dict
            currentInputFrame = currentVehicleList[jdx][frameIDIndex]
            currentInputTime = currentVehicleList[jdx][globalTimeIndex]
            otherVechiles = dictByFrames[str(currentInputTime)]
            # Get the previous timestamp (-100) other vehicles velcotiy calculations
            prevOtherVehicles = dictByFrames[str(currentInputTime-100)]
            # Get the next timestamp (+100) also in case the vehicle is not present in prev timestamp
            nextOtherVehicles = dictByFrames[str(currentInputTime+100)]

            #################################################################################################
            ########################### Removing this check just to be sure !!!!
            #################################################################################################

            # # # # # # Remove the prediction not eligible vehicles
            # # # # # eligibleOtherVehicles = []
            # # # # # for eachOtherVehicle in otherVechiles:
            # # # # #     otherID = eachOtherVehicle[vechileIDIndex]
            # # # # #     if (otherID in currentOtherEligibleVehicles):
            # # # # #         eligibleOtherVehicles.append(eachOtherVehicle)

            #################################################################################################

            # Remove vehicles with a different global time which is not possible. Just adding check to be sure
            for fdx,eachOtherTime in enumerate(otherVechiles):
                otherTime = eachOtherTime[globalTimeIndex]
                if (otherTime != currentInputTime):
                    print('Mismatch in input global time..')
                    print('other Time ' + str(otherTime))
                    print('Current Time ' + str(currentInputTime))
                    sys.exit()

            # Extend the surrounding cars info into the target vehicles input   otherVechiles replaced by  eligibleOtherVehicles
            ##################################################################################################################
            # Changed from eligibleOtherVehicles -> otherVechiles as the extra checke removed
            # # # tempInput = GetSurroundingCarsInfo(eligibleOtherVehicles, tempInput, targetVehicleID, inputStr, absoluteX, absoluteY, initialLocalX, initialLocalY)
            tempInput = GetSurroundingCarsInfo(otherVechiles, prevOtherVehicles, nextOtherVehicles, tempInput, targetVehicleID, inputStr, absoluteX, absoluteY, initialLocalX, initialLocalY)
            ##################################################################################################################

            if (len(tempInput) != globalInputFeatures):
                print('tempInput len is : ' + str(len(tempInput)) + ' instead of ' + str(globalInputFeatures))
                sys.exit()

            # Add the final list of target vehicles and other vehicles info into the local input
            localXData.append(tempInput)


        # Prepeare sequential Output Data and decoder input data
        localYPoseData = []
        decoderInputData = []

        # Prepare the First Decoder Input  
        # Main input structure tempInput = [localX,localY,velocityX,velocityY,laneID,movement]
        lastInput = localXData[-1]
        lastInputPoseX = lastInput[0]
        lastInputPoseY = lastInput[1]
        firstDecoderInput = [lastInputPoseX,lastInputPoseY]

        for kdx in range(idx,idx+futureTemporal):

            # Add the ground truth outputs

            # Get the pose info
            nextLocalX = currentVehicleList[kdx][localXIndex]
            nextLocalY = currentVehicleList[kdx][localYIndex]
            # If IntialX positive means right side, means increasing Y else left side, decreasing Y, similar for the positive or negetive X based on side
            # Right bent positive and left bent negetive
            # For I-80 all vehicles are moving forward
            nextRelativeX = nextLocalX-initialLocalX     
            nextRelativeY = nextLocalY-initialLocalY

             # Safety check velocityY should not be high negetive
            if(nextRelativeY<-2):
                ignoreSampleFlag = True
                # # print('Re mapped Key : ' + str(currentID))
                # # print('Origina Key : ' + str(targetVehicleID))
                # # print('nextRelativeY : ' + str(nextRelativeY))
                # # nanVal = 1000/0


            # Get the velocity
            # # # # Get the velocity directly from the csv         
            # # # nextVelocity = currentVehicleList[kdx][velocityIndex]
            # # # deNormalizedNextVelocity = (nextVelocity*(maxVel-minVel))+minVel

            # Calculate the velocity from the prev pose
            prevLocalPoseX = currentVehicleList[kdx-1][localXIndex]
            prevLocalPoseY = currentVehicleList[kdx-1][localYIndex]

            # # # # # Safety check velocityY should not be high negetive
            # # # # if(nextVelocityY<-2):
            # # # #     ignoreSampleFlag = True
            # # # #     # # print('Re mapped Key : ' + str(currentID))
            # # # #     # # print('Origina Key : ' + str(targetVehicleID))
            # # # #     # # print('nextVelocityY : ' + str(nextVelocityY))
            # # # #     # # nanVal = 1000/0
                

            # Finally append all the values
            localYPoseData.append([nextRelativeX,nextRelativeY])

            # Add the decoder input
            # # # # Add the distance from the junc in the decoder as well
            # # # nextSection = currentVehicleList[kdx][sectionIndex]
            # # # nextIntersection = currentVehicleList[kdx][intersectionIndex]
            # # # juncDist = CalculateNearestJuncLoc(nextSection, nextIntersection, nextLocalX, nextLocalY)

            decoderTemp = [nextRelativeX,nextRelativeY]

            # Check the decoder feature length
            if (len(decoderTemp) != globalDecoderFeatures):
                print('decoderTemp len is : ' + str(len(decoderTemp)) + ' instead of ' + str(globalDecoderFeatures))
                sys.exit()

            # Finally append the target car and surrounding cars info for the current frame into the final decoded input
            decoderInputData.append(decoderTemp)

        # Shift one time stamp right and append Last input at the beggining 
        decoderInputData = decoderInputData[:-1]
        decoderInputData.insert(0,firstDecoderInput)

        ##########################################################################
        ################## NON TRANSFER LEARNING ##################################
        # Append the sample to the train or validation list based on the passed str
        # Append in the final validation or training set based on decided vehicle ID
        # Also check if the sample is ignored
        if(ignoreSampleFlag):
            ignoreSampleCountList.append(0)
        else:
            if(currentTrainOrValStr == validationStr):
                validationProcessList.append([localXData,decoderInputData,localYPoseData])
            elif(currentTrainOrValStr == trainStr):
                trainProcessList.append([localXData,decoderInputData,localYPoseData])
            else:
                print('Unknown Train Val string')
                sys.exit()
        ##########################################################################

        ##########################################################################
        ####################  TRANSFER LEARNING ##################################
        # Append the sample to the train or validation based on the decided Flag
        # Based on the Intersection and section ID of the sample its decider
        ######################################################################### 
        # # # # # if(validationFlag == True):
        # # # # #     validationProcessList.append([localXData,decoderInputData,localYMovementData,localYVelData,localYPoseData])
        # # # # # elif(validationFlag == False):
        # # # # #     trainProcessList.append([localXData,decoderInputData,localYMovementData,localYVelData,localYPoseData])
        # # # # # else:
        # # # # #     print('Unknown Train Val string')
        # # # # #     sys.exit()
        ######################################################################### 

       
        # Clean up the processed list.. (not a good idea....)
        # localXData = []
        # decoderInputData = []
        # localYMovementData = []
        # localYVelData = []
        # localYPoseData = []

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

# Function to write each sample in a single folder for generator
def GeneretorWriteData(genFolderName,genFileNameList,totalGenArray):

    # Create parent folder
    genFolderPath = folderName + genFolderName
    os.mkdir(genFolderPath)
    genSampleCount = 0
    arrayLen = -999

    # All array length should be same
    # Both options are for either vanilla or proposed
    if(len(totalGenArray[0]) == len(totalGenArray[1]) == len(totalGenArray[2]) == len(totalGenArray[3]) == len(totalGenArray[4])):
        # Means all arrays are of same length. Use any one as index
        arrayLen = len(totalGenArray[0])
    else:
        print('Array lengths are not matched in GeneretorWriteData function!!!')
        sys.exit()

    # Split the index list into small lists each of size batchsize
    allArrayIndexes = list(range(arrayLen))
    splittedIndexes = [allArrayIndexes[i * batchSize:(i + 1) * batchSize] for i in range((len(allArrayIndexes) + batchSize - 1) // batchSize )]

    for eachIndexList in splittedIndexes:
        # Create the sample batch folder
        currentSampleFolder = genFolderPath + '/' + str(genSampleCount).zfill(6)
        os.mkdir(currentSampleFolder)

        for eachIndex in eachIndexList:
            # Write individual samples input and outouts
            # Both options are for either vanilla or proposed
            for writeIdx in range(0,5):
            # # # #for writeIdx in range(0,3):
                # Extract the current sample
                genSample = totalGenArray[writeIdx][eachIndex]
                genSampleFilePath = currentSampleFolder + genFileNameList[writeIdx]

                # Write the sample into the file
                with open(genSampleFilePath, 'a') as fholder:
                    for item in genSample:
                        fholder.write("%s\n" % list(item))

        # After writing all 5 items (2 inputs and 3 outputs increase the folder path count)
        genSampleCount = genSampleCount + 1

# Identifies the min and max value of the array for normalizes the array
def ArrayMinMaxFinder(arryaName, totalLength, stepCount):
    # Declare all the global Velocity max min 
    global minVelocityX, maxVelocityX, minVelocityY, maxVelocityY, maxRealtiveX, maxRealtiveY, minRealtiveX, minRealtiveY

    # Start all the min max calculation
    for ydx in range(len(arryaName)):
        for zdx in range(0,totalLength, stepCount):
            # Identify min max for pose X and Y
            currentXMax = max(arryaName[ydx,:,zdx])
            currentYMax = max(arryaName[ydx,:,zdx+1])
            if(currentXMax>maxRealtiveX):
                maxRealtiveX = currentXMax
            if(currentYMax>maxRealtiveY):
                maxRealtiveY = currentYMax

            currentXMin = min(arryaName[ydx,:,zdx])
            currentYMin = min(arryaName[ydx,:,zdx+1])
            if(currentXMin<minRealtiveX):
                minRealtiveX = currentXMin
            if(currentYMin<minRealtiveY):
                minRealtiveY = currentYMin

            # Identify min max for velocity X and Y
            currentVelocityXMax = max(arryaName[ydx,:,zdx+2])
            currentVelocityYMax = max(arryaName[ydx,:,zdx+3])
            if(currentVelocityXMax>maxVelocityX):
                maxVelocityX = currentVelocityXMax
            if(currentVelocityYMax>maxVelocityY):
                maxVelocityY = currentVelocityYMax

            currentVelocityXMin = min(arryaName[ydx,:,zdx+2])
            currentVelocityYMin = min(arryaName[ydx,:,zdx+3])
            if(currentVelocityXMin<minVelocityX):
                minVelocityX = currentVelocityXMin
            if(currentVelocityYMin<minVelocityY):
                minVelocityY = currentVelocityYMin

# Normalizes the array with previsouly updated min and max value
def ArrayNormalizer(arryaName, totalLength, stepCount):
    # Normalize the arrays using the globally initialized min and max
    for ydx in range(len(arryaName)):
        for zdx in range(0,totalLength,stepCount):
            # Normalize pose X and Y
            arryaName[ydx,:,zdx] = (arryaName[ydx,:,zdx]-minRealtiveX)/(maxRealtiveX-minRealtiveX)
            arryaName[ydx,:,zdx+1] = (arryaName[ydx,:,zdx+1]-minRealtiveY)/(maxRealtiveY-minRealtiveY)

            # Normalize velocity X and Y
            arryaName[ydx,:,zdx+2] = (arryaName[ydx,:,zdx+2]-minVelocityX)/(maxVelocityX-minVelocityX)
            arryaName[ydx,:,zdx+3] = (arryaName[ydx,:,zdx+3]-minVelocityY)/(maxVelocityY-minVelocityY)


# Identifies the min and max value of the array for normalizes the array
def ArrayMinMaxFinderVanilla(arryaName, totalLength, stepCount, isInput):
    # Declare all the global Velocity max min 
    global minVelocityX, maxVelocityX, minVelocityY, maxVelocityY, maxRealtiveX, maxRealtiveY, minRealtiveX, minRealtiveY

    # Start all the min max calculation
    for ydx in range(len(arryaName)):
        for zdx in range(0,totalLength, stepCount):
            # Identify min max for pose X and Y
            currentXMax = max(arryaName[ydx,:,zdx])
            currentYMax = max(arryaName[ydx,:,zdx+1])
            if(currentXMax>maxRealtiveX):
                maxRealtiveX = currentXMax
            if(currentYMax>maxRealtiveY):
                maxRealtiveY = currentYMax

            currentXMin = min(arryaName[ydx,:,zdx])
            currentYMin = min(arryaName[ydx,:,zdx+1])
            if(currentXMin<minRealtiveX):
                minRealtiveX = currentXMin
            if(currentYMin<minRealtiveY):
                minRealtiveY = currentYMin

            if(isInput):
                # Identify min max for velocity X and Y
                currentVelocityXMax = max(arryaName[ydx,:,zdx+2])
                currentVelocityYMax = max(arryaName[ydx,:,zdx+3])
                if(currentVelocityXMax>maxVelocityX):
                    maxVelocityX = currentVelocityXMax
                if(currentVelocityYMax>maxVelocityY):
                    maxVelocityY = currentVelocityYMax

                currentVelocityXMin = min(arryaName[ydx,:,zdx+2])
                currentVelocityYMin = min(arryaName[ydx,:,zdx+3])
                if(currentVelocityXMin<minVelocityX):
                    minVelocityX = currentVelocityXMin
                if(currentVelocityYMin<minVelocityY):
                    minVelocityY = currentVelocityYMin


# Normalizes the array with previsouly updated min and max value
def ArrayNormalizerVanilla(arryaName, totalLength, stepCount, isInput):
    # Normalize the arrays using the globally initialized min and max
    for ydx in range(len(arryaName)):
        for zdx in range(0,totalLength,stepCount):
            # Normalize pose X and Y
            arryaName[ydx,:,zdx] = (arryaName[ydx,:,zdx]-minRealtiveX)/(maxRealtiveX-minRealtiveX)
            arryaName[ydx,:,zdx+1] = (arryaName[ydx,:,zdx+1]-minRealtiveY)/(maxRealtiveY-minRealtiveY)

            if(isInput):
                # Normalize velocity X and Y
                arryaName[ydx,:,zdx+2] = (arryaName[ydx,:,zdx+2]-minVelocityX)/(maxVelocityX-minVelocityX)
                arryaName[ydx,:,zdx+3] = (arryaName[ydx,:,zdx+3]-minVelocityY)/(maxVelocityY-minVelocityY)



# Plot all cars trajectory on the global GPS map
def CreateData(inputFileName,totalTrainOrValStr):

    # Load the Vehicle and Frame based Dictionaries
    global dictByFrames, dictByVehicles, mapper

    # Clean up the dictionary before re use in the second call
    dictByFrames = dict()
    dictByVehicles = dict()
    # Check the dictionary lengths just to be sure
    frameDictLen = len(dictByFrames)
    vehicleDictLen = len(dictByVehicles)
    if(frameDictLen != 0 or vehicleDictLen != 0):
        print('Either dictionary is not clear yet!!!')
        sys.exit()
        nanVal = 1000/0

    # Clean up general process list
    generalProcessList[:] = []
    # Check the length of general process list. Its shoudl be zero. Else through errir
    generaProcListLen = len(generalProcessList)
    if(generaProcListLen != 0):
        print('General Process list length is not zero!!!')
        print('Expected process list len = 0')
        print('Actual process list length = ' + str(generaProcListLen))
        sys.exit()
        nanVal = 1000/0


    dictByFrames,dictByVehicles,unusedMapperDict = CreateVehicleAndFrameDict(inputFileName)
    finalVehicleKeys = list(dictByVehicles.keys())
    finalVehicleKeys.sort(key=float)
    finalFrameKeys = list(dictByFrames.keys())
    finalFrameKeys.sort(key=float)

    # Pin the new process to specified cores
    # os.system("taskset -p -c 1,2,3,4,5,6,7,8,9,10,11 %d" % os.getpid()) 
    os.system("taskset -p -c 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28 %d" % os.getpid())
    processes = []
    numberofCores =  30  #    30

    pool = mp.Pool(numberofCores)

    print('Total Vehicle : ' + str(len(finalVehicleKeys)))

    # Select vehicles from the total list based on the training of test
    if(totalTrainOrValStr == trainStr):
        selectedVehilces = random.sample(finalVehicleKeys,trainVehileCount)
    elif(totalTrainOrValStr == validationStr):
        selectedVehilces = random.sample(finalVehicleKeys,validationVehicleCount)
    elif(totalTrainOrValStr == testStr):
        selectedVehilces = random.sample(finalVehicleKeys,testVehicleCount)
    else:
        print('Unknown train or validation total str!!!')
        nanVal = 1000/0
        sys.exit()
        

    ################################################################################
    # NON-TRANSFER learning part. Select random vehicle ids to use as validation data
    # Also writethe ids in the training and validation files for later use
    ################################################################################
    # Depeding on the current totalTrainOrValStr str write all the selected vehicle IDs to the corresponding file
    if(totalTrainOrValStr == trainStr):
        # Write the training vehicles to the data folder for intermediate prediction
        trainingFileObj = open(trainingFileName, 'x')
        for eachTrainingCar in selectedVehilces:
            trainingFileObj.write("%s\n" % eachTrainingCar)
        trainingFileObj.close()

    elif(totalTrainOrValStr == validationStr):
        # Write the validation vehicles to the data folder
        validationFileObj = open(validationFileName, 'x')
        for eachValidationCar in selectedVehilces:
            validationFileObj.write("%s\n" % eachValidationCar)
        validationFileObj.close()

    elif(totalTrainOrValStr == testStr):
        # Write the validation vehicles to the data folder
        testFileObj = open(testingFileName, 'x')
        for eachTestCar in selectedVehilces:
            testFileObj.write("%s\n" % eachTestCar)
        testFileObj.close()

    else:
        print('Unknown train or validation total str!!!')
        nanVal = 1000/0
        sys.exit()
    #################################################################################


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

        ############################################################################
        # NON-TRANSFER learning part, Also chnage in the "ProcessByVehicle" function
        # Change the validation Flag setting... In this case we will pass the flag
        # Also change the passed values indexing (3 items: Key, train/val and originalKey)
        ############################################################################
        # # # # Chnaged for seperate data train/val/test
        processList.append([eachKey,originalID])
        ############################################################################

        ############################################################################
        # TRANSFER learning part, Also chnage in the "ProcessByVehicle" function
        # Change the validation Flag setting... In this case based on the section and intersection the function will decide
        # Also change the passed values indexing (2 items: Key and originalKey, NO train/val flag)
        ############################################################################
        # # # # Chnaged for transfer learning
        # # # processList.append([eachKey,originalID])
        ############################################################################


    # For local test use the "for-loop" and for final use the map. Everything all same, dont forget the manager...
    # This one is for the modifed encoder decoder with the surroudnig info in decoder
    pool.map(ProcessByVehicle,processList)
    # # # This one is for the vanilla one
    # # pool.map(ProcessByVehicleForVanilla,processList)

    # # # for eachItem in processList:
    # # #     ProcessByVehicle(eachItem)

    # Print the nuuumber of ignored samples
    ignoredSampleCount = len(ignoreSampleCountList)
    print('Number of samples ignored: ' + str(ignoredSampleCount))

    print('General process list length after sample addition!!!')
    print(len(generalProcessList))

    
    # Convert the General manager list to normal list
    print('Converting the General Manager list to normal lists.....')
    normalGeneralList = list(generalProcessList)
    print('List converted!!!')


    # Extract individual lists to write the shapes
    # Train final lists
    print('Prepering the individual lists')

    finalXGeneral = [x[0] for x in normalGeneralList]
    finalGeneralDecoderInput = [x[1] for x in normalGeneralList]
    finalYClassGeneral = [x[2] for x in normalGeneralList]  # Commenting out for vanilla
    finalYVelGeneral = [x[3] for x in normalGeneralList]    # Commenting out for vanilla
    finalYPoseGeneral = [x[4] for x in normalGeneralList]

    print('Individual list prepeared!!!')

    print('Casting it to Array!!!')

    # Prepare the final Train/Val/Test arrays
    XGeneral = np.array(finalXGeneral)
    decoderGeneralInput = np.array(finalGeneralDecoderInput)
    YClassGeneral = np.array(finalYClassGeneral)       # Commenting out for vanilla
    YVelGeneral = np.array(finalYVelGeneral)           # Commenting out for vanilla
    YPoseGeneral = np.array(finalYPoseGeneral)

    print('All casted to Array!!!')

    # Normalize the arrays and at the same time the min max values will be updated
    # Normalize the relative X/Y positions based on decoder Input
    # Identify the min and max locations among both train and val arrays
    print('Normalizing the X/Y poses and X/Y velocity!!!')

    ArrayMinMaxFinder(XGeneral, globalInputFeatures, inputFeatureCount)       
    ArrayMinMaxFinder(decoderGeneralInput, globalDecoderFeatures, decoderFeatureCount)

    # Normalize train and val input plus train and val decoder input based of the identified min max pose
    ArrayNormalizer(XGeneral, globalInputFeatures, inputFeatureCount)        
    ArrayNormalizer(decoderGeneralInput, globalDecoderFeatures, decoderFeatureCount)

    print('All X/Y poses are normalized!!!')

    # Print relative X Y max and each normalized array min max
    print('Relative X max :' + str(maxRealtiveX))
    print('Relative Y max :' + str(maxRealtiveY))
    print('Relative X min :' + str(minRealtiveX))
    print('Relative Y min :' + str(minRealtiveY))
    
    print('VelocityX min :' + str(minVelocityX))  
    print('VelocityX max :' + str(maxVelocityX))  
    print('VelocityY min :' + str(minVelocityY))  
    print('VelocityY max :' + str(maxVelocityY))  


    print('XGeneral max :' + str(np.amax(XGeneral)))
    print('decoderGeneralInput max :' + str(np.amax(decoderGeneralInput)))

    # Check all the array to see if the max value not exceds 1.0
    if(np.amax(XGeneral)>1 or np.amax(decoderGeneralInput)>1):
        print('One of the above array in not normalized properly!!!!')
        sys.exit()

    # Since the Test is the last call write the min max at the Test Call
    if(totalTrainOrValStr == testStr):
        # Write the min-max values to the file for later use
        filePath = folderName + '/MinMax.txt'
        with open(filePath, 'x') as fshape:
            fshape.write('maxRealtiveX:' + str(maxRealtiveX) + '\n')
            fshape.write('maxRealtiveY:' + str(maxRealtiveY) + '\n')
            fshape.write('minRealtiveX:' + str(minRealtiveX) + '\n')
            fshape.write('minRealtiveY:' + str(minRealtiveY) + '\n')
            fshape.write('minVelocityX:' + str(minVelocityX) + '\n')      # Commenting out for vanilla
            fshape.write('maxVelocityX:' + str(maxVelocityX) + '\n')      # Commenting out for vanilla
            fshape.write('minVelocityY:' + str(minVelocityY) + '\n')      # Commenting out for vanilla
            fshape.write('maxVelocityY:' + str(maxVelocityY) + '\n')      # Commenting out for vanilla


    # Since now there would be three calls for train, val and test using file mode 'a'
    # Print the shape of the Arrays
    if(totalTrainOrValStr == trainStr):
        filePath = folderName + '/arrayShapes.txt'
        with open(filePath, 'a') as fshape:
            fshape.write('XTrain shape : ' + str(XGeneral.shape) + '\n')
            fshape.write('decoderTrainInput shape : ' + str(decoderGeneralInput.shape) + '\n')
            fshape.write('YClassTrain shape : ' + str(YClassGeneral.shape) + '\n')   
            fshape.write('YVelTrain shape : ' + str(YVelGeneral.shape) + '\n')       
            fshape.write('YPoseTrain shape : ' + str(YPoseGeneral.shape) + '\n')

    elif(totalTrainOrValStr == validationStr):
        filePath = folderName + '/arrayShapes.txt'
        with open(filePath, 'a') as fshape:
            fshape.write('XVal shape : ' + str(XGeneral.shape) + '\n')
            fshape.write('decoderValInput shape : ' + str(decoderGeneralInput.shape) + '\n')
            fshape.write('YClassVal shape : ' + str(YClassGeneral.shape) + '\n')   
            fshape.write('YVelVal shape : ' + str(YVelGeneral.shape) + '\n')       
            fshape.write('YPoseVal shape : ' + str(YPoseGeneral.shape) + '\n')

    elif(totalTrainOrValStr == testStr):
        filePath = folderName + '/arrayShapes.txt'
        with open(filePath, 'a') as fshape:
            fshape.write('XTest shape : ' + str(XGeneral.shape) + '\n')
            fshape.write('decoderTestInput shape : ' + str(decoderGeneralInput.shape) + '\n')
            fshape.write('YClassTest shape : ' + str(YClassGeneral.shape) + '\n')   
            fshape.write('YVelTest shape : ' + str(YVelGeneral.shape) + '\n')       
            fshape.write('YPoseTest shape : ' + str(YPoseGeneral.shape) + '\n')

    else:
        print('Unknown train or validation total str!!!')
        nanVal = 1000/0
        sys.exit()


    # Write the arrays in a generator format
    # Write the Train data list
    print('Writing general samples!!!')
    generalAllArrays = [XGeneral,decoderGeneralInput,YClassGeneral,YVelGeneral,YPoseGeneral]  
    if(totalTrainOrValStr == trainStr):
        GeneretorWriteData(trainFolderName,sampleFileNameList,generalAllArrays)
    elif(totalTrainOrValStr == validationStr):
        GeneretorWriteData(valFolderName,sampleFileNameList,generalAllArrays)
    elif(totalTrainOrValStr == testStr):
        GeneretorWriteData(testFolderName,sampleFileNameList,generalAllArrays)
    else:
        print('Unknown train or validation or test total str!!!')
        nanVal = 1000/0
        sys.exit()


def ModelArch():

    # Saniy check Min max values should be same as -9999 or 99999 after update
    if((minVelocityX == 999) or (maxVelocityX == -999) or (minVelocityY == 999) or (maxVelocityY == -999) or (maxRealtiveX == -9999) or (maxRealtiveY == -9999) or (minRealtiveX == 9999) or (minRealtiveY == 9999)):
        print('Min max values are not porperly updated in the UpdateMinMax function!!!')
        sys.exit()

    import tensorflow as tf

    from keras import backend as K
    from keras.models import Model
    from keras.layers import Input,LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dropout, MaxPooling1D, Concatenate, subtract, Lambda, BatchNormalization, LeakyReLU, ELU
    from keras import optimizers
    from keras.optimizers import RMSprop, Adam, Nadam
    from keras.losses import logcosh

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

    # Common BatchNorm
    # # commonBatchNorm = BatchNormalization()
    # # decoder_outputs = commonBatchNorm(decoder_outputs)

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
    decoder_dense24 = Dense(32, activation='linear')
    # decoder_Leaky24 = ELU(alpha=leakyAlphaValue)
    decoder_output2 = decoder_dense24(decoder_output2)
    # decoder_output2 = batchNorm12(decoder_output2)
    # decoder_output2 = decoder_Leaky24(decoder_output2)
    decoder_dense25 = Dense(2, activation='linear', name='Velcoity')
    velocityOut = decoder_dense25(decoder_output2)

    # Prepeare the slice layers and separate the Vx and Vy
    velocityExtractX = Lambda(lambda x: tf.slice(x, (0, 0, 0), (-1, -1, 1)))
    velocityExtractY = Lambda(lambda x: tf.slice(x, (0, 0, 1), (-1, -1, 1)))

    velocityOutX = velocityExtractX(velocityOut)
    velocityOutY = velocityExtractY(velocityOut)

    # Decoder for position out, This time only extract the first two poses (X,Y) from the decoder input as the decoder input holds the last pose
    # Simply add the predicted motion to the last pose to get the current pose

    # slice the decoder input same as velocity extract to get the first two items. slice(start,size)
    # Prepeare the slice layers and separate the poseX and poseY
    poseExtractX = Lambda(lambda c: tf.slice(c, (0, 0, 0), (-1, -1, 1)))
    poseExtractY = Lambda(lambda c: tf.slice(c, (0, 0, 1), (-1, -1, 1)))

    decoderPoseOutX = poseExtractX(decoder_inputs)
    decoderPoseOutY = poseExtractY(decoder_inputs)

    # As the decoder_input has the normalized pose we need to unnormalize it first to get the real pose
    # Prepeare the unnormalization layers
    minXPoseConst = K.constant(value=minRealtiveX, dtype='float32')
    minXMaxPoseDiffConst = K.constant(value=(maxRealtiveX-minRealtiveX), dtype='float32')

    minYPoseConst = K.constant(value=minRealtiveY, dtype='float32')
    minYMaxPoseDiffConst = K.constant(value=(maxRealtiveY-minRealtiveY), dtype='float32')

    poseXUnNormalizedLayer = Lambda(lambda d: (d*minXMaxPoseDiffConst) + minXPoseConst)
    poseYUnNormalizedLayer = Lambda(lambda d: (d*minYMaxPoseDiffConst) + minYPoseConst)

    # Add the velocity output (we need NOT normalized as this will be directly added) to the not normalized pose
    unNormalizedPoseX = poseXUnNormalizedLayer(decoderPoseOutX)
    addPoseXLayer = Add()
    poseXOut = addPoseXLayer([unNormalizedPoseX,velocityOutX])

    unNormalizedPoseY = poseYUnNormalizedLayer(decoderPoseOutY)
    addPoseYLayer = Add()
    poseYOut = addPoseYLayer([unNormalizedPoseY,velocityOutY])

    # Concat poseX and poseY to formate the final positionOut (X,Y)
    decoder6_concat = Concatenate(name='position')
    positionOut = decoder6_concat([poseXOut,poseYOut])
    
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

    # Common BatchNorm
    # # decoder_outputs = commonBatchNorm(decoder_outputs)

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
    # decoder_output2 = decoder_Leaky24(decoder_output2)
    velocityOut = decoder_dense25(decoder_output2)

    # Inference Decoder Velocity Normalizer
    velocityOutX = velocityExtractX(velocityOut)
    velocityOutY = velocityExtractY(velocityOut)

    velocityConcatX = velocityXNormalized(velocityOutX)
    velocityConcatY = velocityYNormalized(velocityOutY)
    

    #Inference  Decoder for position out
    decoder_output3 = decoder3_concat([decoder_outputs,classOut,velocityConcatX,velocityConcatY])    
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
    # decoder_output3 = decoder_Leaky34(decoder_output3)
    positionOut = decoder_dense35(decoder_output3)

    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [classOut, velocityOut, positionOut] + decoder_states)

    opt = Nadam()     #   RMSprop()

    model.compile(optimizer=opt, loss=['categorical_crossentropy', logcosh, euclidean_distance_loss])

    return model,encoder_model,decoder_model

# Read the data from the file for the generator
def GenReadFromFile(filePath):

    readFile = open(filePath, "r")
    loadedData = readFile.readlines()

    dataFloat = []

    for eachLoadedData in loadedData:
        currentSample = eachLoadedData[1:-2].split(',')   
        currentSampleFloat = [float(i) for i in currentSample]
        dataFloat.append(currentSampleFloat) 

    return dataFloat


# Test the trained model just on the validation data
def ValidationSetPredict(totalSampleCount,valSamples):

    import tensorflow as tf
    from keras.models import load_model
    config = tf.ConfigProto()   # compat.v1.
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    print('Waiting for GPU devices!!!')
    sleep(0.5)
    
    # Get the model architecture
    model,encoder_model,decoder_model = ModelArch()

    # Load the model weights
    print('Loading the models!!!')

    encoder_model.load_weights(encoderModelFilename)
    print('Encoder loaded!!!')
    decoder_model.load_weights(decoderModelFilename)
    print('Decoder loaded!!!')

    print('Models Loaded!!!')

    # Filter parameters
    particleCount = 300
    varianceOffsetX = 8
    varianceOffsetY = 20

    # Image circle properties
    radius = 1
    blueColor = (255,0,0)
    redColor = (0,0,255)
    greenColor = (0,255,0)
    thickness = 1
    textThickness = 3
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = 0.7
    

    # Create the show window
    windowName = 'image'
    cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)


    # Predict sequence
    for eachValidationSample in valSamples:

        # Prepare the path for the validation samples
        validationSamplePath = folderName + valFolderName + '/' + eachValidationSample
        encoderInputPath = validationSamplePath + sampleFileNameList[0]
        decoderInputPath = validationSamplePath + sampleFileNameList[1]
        poseOutputPath = validationSamplePath + sampleFileNameList[4]

        # Read the entire batch
        rawPredictInput = GenReadFromFile(encoderInputPath)
        rawDecoderLocal = GenReadFromFile(decoderInputPath)
        rawGroundTruthPose = GenReadFromFile(poseOutputPath)

        # Split the list based in the history or future temporal length for each sample
        totalEncoderInput = [rawPredictInput[i * historyTemporal:(i + 1) * historyTemporal] for i in range((len(rawPredictInput) + historyTemporal - 1) // historyTemporal )]
        totalDecoderInput = [rawDecoderLocal[i * futureTemporal:(i + 1) * futureTemporal] for i in range((len(rawDecoderLocal) + futureTemporal - 1) // futureTemporal )]
        totalPoseOut = [rawGroundTruthPose[i * futureTemporal:(i + 1) * futureTemporal] for i in range((len(rawGroundTruthPose) + futureTemporal - 1) // futureTemporal )]

        for eachPredInput,decoderLocal,groundTruthPose in zip(totalEncoderInput,totalDecoderInput,totalPoseOut):

            # # # Before each sample create a blank RGB image to show the trajectory and the particles
            # # blankImage = np.zeros((400,200,3), np.uint8)
            # # blankImage.fill(255)

            # # # Put the color codes
            # # blankImage = cv2.putText(blankImage, 'measurement', (90,10), font, fontScale, blueColor, textThickness, cv2.LINE_AA)
            # # blankImage = cv2.putText(blankImage, 'particles', (90,40), font, fontScale, greenColor, textThickness, cv2.LINE_AA)
            # # blankImage = cv2.putText(blankImage, 'PF Pose', (90,70), font, fontScale, redColor, textThickness, cv2.LINE_AA)


            currentPredictInput = np.array(totalEncoderInput[127]).reshape(1,historyTemporal,globalInputFeatures)
            # currentPredictInput = np.array(eachPredInput).reshape(1,historyTemporal,globalInputFeatures)

            # Create a partcile filter object before each sample and use the 30 frames to update
            initialPoseX = (currentPredictInput[0][1][0]*(maxRealtiveX-minRealtiveX)) + minRealtiveX  # First sample (3D), First sample 0. first item poseX, unNormalized
            initialPoseY = (currentPredictInput[0][1][1]*(maxRealtiveY-minRealtiveY)) + minRealtiveY  # First sample (3D), First sample 0. second item poseY, unNormalized

            # Use the initial velocity X and Y as initial variance
            intitalCovarianceX = abs((currentPredictInput[0][1][2]*(maxVelocityX-minVelocityX)) + minVelocityX) + varianceOffsetX
            intitalCovarianceY = abs((currentPredictInput[0][1][3]*(maxVelocityY-minVelocityY)) + minVelocityY) + varianceOffsetY

            # During prediction use both predicted motion and position to update the filter
            motionPositionFilterObj = PFHelper.ParticleFilter(particleCount,[],'Classical',initialPoseX,initialPoseY,intitalCovarianceX,intitalCovarianceY)
            # Duirng prediction only use the predicted position and use the true last motion to update filter
            onlyPositionFilterObj = PFHelper.ParticleFilter(particleCount,[],'Classical',initialPoseX,initialPoseY,intitalCovarianceX,intitalCovarianceY)
            # Duirng prediction do not regenerate the particle and use the current particles
            varNonReGenFilterObj = PFHelper.ParticleFilter(particleCount,[],'Classical',initialPoseX,initialPoseY,intitalCovarianceX,intitalCovarianceY,False)

            # Pass through the input poses and velocitites to update the filter
            # Extra 0 index for the 3D casting of input
            for mdx in range(2,historyTemporal):  

                # Before each sample create a blank RGB image to show the trajectory and the particles
                blankImage = np.zeros((600,600,3), np.uint8)
                blankImage.fill(255)

                # Put the color codes
                blankImage = cv2.putText(blankImage, 'measurement', (400,40), font, fontScale, blueColor, textThickness, cv2.LINE_AA)
                blankImage = cv2.putText(blankImage, 'particles', (400,80), font, fontScale, greenColor, textThickness, cv2.LINE_AA)
                blankImage = cv2.putText(blankImage, 'PF Pose', (400,120), font, fontScale, redColor, textThickness, cv2.LINE_AA)
                blankImage = cv2.putText(blankImage, 'MotionPose,OnlyPose,ContVar', (300,160), font, fontScale, redColor, textThickness, cv2.LINE_AA)

                # Extract pose and unnormalize to use it in the filter 
                normalizedInputPoseX = currentPredictInput[0][mdx][0]         # 0: Only sample mdx time index 0th item poseX
                normalizedInputPoseY = currentPredictInput[0][mdx][1]         # 0: Only sample mdx time index 1st item poseY
                normalizedInputVelocityX = currentPredictInput[0][mdx][2]     # 0: Only sample mdx time index 2nd item velocityX
                normalizedInputVelocityY = currentPredictInput[0][mdx][3]     # 0: Only sample mdx time index 3rd item velocityY

                # Normalize the pose and velocity info
                inputPoseX = (normalizedInputPoseX*(maxRealtiveX-minRealtiveX)) + minRealtiveX
                inputPoseY = (normalizedInputPoseY*(maxRealtiveY-minRealtiveY)) + minRealtiveY 
                inputVelocityX = (normalizedInputVelocityX*(maxVelocityX-minVelocityX)) + minVelocityX 
                inputVelocityY = (normalizedInputVelocityY*(maxVelocityY-minVelocityY)) + minVelocityY 

                # Update the filters
                motionPositionFilterObj.update([inputPoseX,inputPoseY], inputVelocityX, inputVelocityY)
                onlyPositionFilterObj.update([inputPoseX,inputPoseY], inputVelocityX, inputVelocityY)
                varNonReGenFilterObj.update([inputPoseX,inputPoseY], inputVelocityX, inputVelocityY)

                # Filter Object Lists
                filterObjList = [motionPositionFilterObj,onlyPositionFilterObj,varNonReGenFilterObj]
                poseOfset = [40,100,160]

                for eachFilObj,eachPoseOffset in zip(filterObjList,poseOfset):
                    # Draw the pose for different filters (X offset to show in between)
                    blankImage = cv2.circle(blankImage, (int(inputPoseX)+eachPoseOffset,int(inputPoseY)), radius+6, blueColor, thickness)
                    # Draw the particles
                    for eachParticle in eachFilObj.particles:
                        blankImage = cv2.circle(blankImage, (int(eachParticle[0])+eachPoseOffset,int(eachParticle[1])), radius, greenColor, thickness)
                    # Draw the PF position
                    blankImage = cv2.circle(blankImage, (int(eachFilObj.particleMean[0])+eachPoseOffset,int(eachFilObj.particleMean[1])), radius+4, redColor, thickness)

                cv2.imshow(windowName,blankImage)
                cv2.waitKey(0)

            print('Filter position shown!!!')
            sys.exit()

            state = encoder_model.predict(currentPredictInput)

            # First decoderVal entry is the first target sequence
            predDecoderInput = decoderLocal[0]

            target_seq = np.array(predDecoderInput).reshape(1,1,globalDecoderFeatures)
    
            currentError = []                        # Calculate the euclidian error for normal predidcted pose
            currentMotionError = []                  # Calculate the euclidian error for pose estimated using predicted motion
            currentMotionPositionFilterError = []    # Calculate the euclidian error for pose estimated using both predicted motion and position with filter
            currentMotionPositionFilterVar = []      # Variance output from the Filter using both motion and position
            currentOnlyPositionFilterError = []      # Calculate the euclidian error for pose estimated using only predicted position and last true motion with filter
            currentOnlyPositionFilterVar = []        # Variance output from the Filter using only position
            currentNonReGenFilterError = []          # Calculate the euclidian error for pose estimated using both predicted motion and position with filter (without particle regeneration)
            currentNonReGenFilterVar = []            # Variance output from the Filter particle regeneation with low variance

            # Extract the last X/Y pose and convert to unnormalized to maintain the origin
            # 0 only sample, -1 get the last time instance, 0/1 poseX/poseY maxRealtiveY unnormalized
            lastXPose = (currentPredictInput[0][-1][0]*(maxRealtiveX-minRealtiveX)) + minRealtiveX
            lastYPose = (currentPredictInput[0][-1][1]*(maxRealtiveY-minRealtiveY)) + minRealtiveY

            # Extract the last X/Y velocity and convert to unnormalized to use as a last true motion for the onlyPositionFilter
            # 0 only sample, -1 get the last time instance, 2/3 velocityX/velocityY 
            lastTrueVelocityX = (currentPredictInput[0][-1][2]*(maxVelocityX-minVelocityX)) + minVelocityX
            lastTrueVelocityY = (currentPredictInput[0][-1][3]*(maxVelocityY-minVelocityY)) + minVelocityY

            # Perfrom the sequential prediction
            for t in range(futureTemporal):
                # predict next Features
                classPred, velcoityPred, posePred, h1, c1, h2, c2 = decoder_model.predict([target_seq] + state)

                # Extract prediction
                predPoseX = posePred[0][0][0]          # First item PoseX
                predPoseY = posePred[0][0][1]          # Second item poseY
                predVelocityX = velcoityPred[0][0][0]  # First item velocityX
                predVelocityY = velcoityPred[0][0][1]  # Second item velocityY

                # Normalize the predicted velocity and local poses for next instance prediction
                normalizedPredPoseX = (predPoseX-minRealtiveX)/(maxRealtiveX-minRealtiveX)
                normalizedPredPoseY = (predPoseY-minRealtiveY)/(maxRealtiveY-minRealtiveY)
                normalizedPredVelocityX = (predVelocityX-minVelocityX)/(maxVelocityX-minVelocityX)
                normalizedPredVelocityY = (predVelocityY-minVelocityY)/(maxVelocityY-minVelocityY)

                # Another predPoseX,predPoseY using the origin and true motion (CV model)
                predPoseXMotion = lastXPose + lastTrueVelocityX
                predPoseYMotion = lastYPose + lastTrueVelocityY
                # update the intital pose
                lastXPose = predPoseXMotion
                lastYPose = predPoseYMotion

                # Use the updated filter to re-estimate the pose using both predicted motion and pose
                motionPositionFilterObj.update([predPoseX,predPoseY], predVelocityX, predVelocityY)
                motionPositionPoseX = motionPositionFilterObj.particleMean[0]
                motionPositionPoseY = motionPositionFilterObj.particleMean[1]
                motionPositionVarX = motionPositionFilterObj.particleCov[0][0]
                motionPositionVarY = motionPositionFilterObj.particleCov[1][1]

                # Use the updated filter to re-estimate the pose using only predicted pose and last true motion
                onlyPositionFilterObj.update([predPoseX,predPoseY], lastTrueVelocityX, lastTrueVelocityY)
                onlyPositionPoseX = onlyPositionFilterObj.particleMean[0]
                onlyPositionPoseY = onlyPositionFilterObj.particleMean[1]
                onlyPositionVarX = onlyPositionFilterObj.particleCov[0][0]
                onlyPositionVarY = onlyPositionFilterObj.particleCov[1][1]

                # Use the updated filter to re-estimate the pose using both predicted motion and pose (propagated variance while re-generating particles)
                varNonReGenFilterObj.update([predPoseX,predPoseY], predVelocityX, predVelocityY)
                varNonReGenPoseX = varNonReGenFilterObj.particleMean[0]
                varNonReGenPoseY = varNonReGenFilterObj.particleMean[1]
                varNonReGenVarX = varNonReGenFilterObj.particleCov[0][0]
                varNonReGenVarY = varNonReGenFilterObj.particleCov[1][1]

                # Extract the ground truth pose for error calculation
                truePoseX = groundTruthPose[t][0]
                truePoseY = groundTruthPose[t][1]

                # Calculate the Euclidian Error for predicted pose
                euclidianError = math.sqrt(((truePoseX-predPoseX)**2) + ((truePoseY-predPoseY)**2))
                euclidianErrorMeter = euclidianError*0.3048
                currentError.append(euclidianErrorMeter)

                # Calculate the Euclidian Error for estimated pose using motion
                euclidianMotionError = math.sqrt(((truePoseX-predPoseXMotion)**2) + ((truePoseY-predPoseYMotion)**2))
                motionErrorMeter = euclidianMotionError*0.3048
                currentMotionError.append(motionErrorMeter)

                # Calculate the Euclidian Error for estimated pose using both predicted position and motion through filter
                euclidianMotionPositionFilterError = math.sqrt(((truePoseX-motionPositionPoseX)**2) + ((truePoseY-motionPositionPoseY)**2))
                euclidianMotionPositionFilterErrorMeter = euclidianMotionPositionFilterError*0.3048
                currentMotionPositionFilterError.append(euclidianMotionPositionFilterErrorMeter)
                
                # Append the pose variance for using both predicted position and motion through filter
                currentMotionPositionFilterVar.append([motionPositionVarX,motionPositionVarY])


                # Calculate the Euclidian Error for estimated pose using both predicted position and motion through filter
                euclidianOnlyPositionFilterError = math.sqrt(((truePoseX-onlyPositionPoseX)**2) + ((truePoseY-onlyPositionPoseY)**2))
                euclidianOnlyPositionFilterErrorMeter = euclidianOnlyPositionFilterError*0.3048
                currentOnlyPositionFilterError.append(euclidianOnlyPositionFilterErrorMeter)

                # Append the pose variance for using only predicted position 
                currentOnlyPositionFilterVar.append([onlyPositionVarX,onlyPositionVarY])


                # Calculate the Euclidian Error for estimated pose using both predicted position and motion through filter without re-generating particles)
                euclidianNonReGenFilterError = math.sqrt(((truePoseX-varNonReGenPoseX)**2) + ((truePoseY-varNonReGenPoseY)**2))
                eucledianNonReGenFilterErrorMeter = euclidianNonReGenFilterError*0.3048
                currentNonReGenFilterError.append(eucledianNonReGenFilterErrorMeter)

                # Append the pose variance for filter with constant and low particle regeneration variance
                currentNonReGenFilterVar.append([varNonReGenVarX,varNonReGenVarY])


                # Update the states of the encoder and the decoder inputs
                state = [h1, c1, h2, c2]

                # update target sequence
                # Update the target sequence till second last frame. At the last frame no need to update the seq as it will not be used
                if(t<(futureTemporal-1)):
                    # targetDecoder = [normalizedPredPoseX,normalizedPredPoseY,normalizedPredVelocity,classPred[0][0][0],classPred[0][0][1],classPred[0][0][2]]
                    # use the normalizedPredPoseYMotion in the decoder
                    targetDecoder = [normalizedPredPoseX,normalizedPredPoseY,normalizedPredVelocityX,normalizedPredVelocityY,classPred[0][0][0],classPred[0][0][1],classPred[0][0][2]]
                    # Quick check**************************************** t+1 -> t
                    surroundingDecoder = decoderLocal[t+1][decoderFeatureCount-1:] # 6 -> target vehicle features (-1 is beacuse to get the target vehicle distance from junc as it is not possible to calculate here due to absence of absolute/initial distance)
                    # surroundingDecoder = decoderLocal[t][decoderFeatureCount-1:] # 6 -> target vehicle features (-1 is beacuse to get the target vehicle distance from junc as it is not possible to calculate here due to absence of absolute/initial distance)
                    targetDecoder.extend(surroundingDecoder)
                    target_seq = np.array(targetDecoder).reshape(1,1,globalDecoderFeatures)


            # Converted to manager format
            trajErrorList.append(currentError)
            motionErrorList.append(currentMotionError)
            positionMotionFilterError.append(currentMotionPositionFilterError)
            positionMotionFilterVar.append(currentMotionPositionFilterVar)
            onlyPositionFilterError.append(currentOnlyPositionFilterError)
            onlyPositionFilterVar.append(currentOnlyPositionFilterVar)
            nonReGenFilterError.append(currentNonReGenFilterError)
            nonReGenFilterVar.append(currentNonReGenFilterVar)

        # Maintain a counter to check the current progress
        sampleCountList.append(0)
        print('!' + str(len(sampleCountList)) + ' of ' +  str(totalSampleCount) , end='', flush=True )





# Update the global min max for both velocity and relative X/Y
def UpdateMinMax():
    # Declare all the global Velocity max min 
    global minVelocityX, maxVelocityX, minVelocityY, maxVelocityY, maxRealtiveX, maxRealtiveY, minRealtiveX, minRealtiveY

    minMaxFilePath = folderName + '/MinMax.txt'
    minMaxFile = open(minMaxFilePath, "r")
    minMaxData = minMaxFile.readlines()

    for eachLine in minMaxData:
        variableName = eachLine.split(':')[0]
        variableValue = float(eachLine.split(':')[1])
        if(variableName == 'minVelocityX'):
            minVelocityX = variableValue
        elif(variableName == 'maxVelocityX'):
            maxVelocityX = variableValue
        elif(variableName == 'minVelocityY'):
            minVelocityY = variableValue
        elif(variableName == 'maxVelocityY'):
            maxVelocityY = variableValue
        elif(variableName == 'maxRealtiveX'):
            maxRealtiveX = variableValue
        elif(variableName == 'maxRealtiveY'):
            maxRealtiveY = variableValue
        elif(variableName == 'minRealtiveX'):
            minRealtiveX = variableValue
        elif(variableName == 'minRealtiveY'):
            minRealtiveY = variableValue
        else:
            print('Unknwon min max variabel name in file!!!')
            sys.exit()

    # Saniy check Min max values should be same as -9999 or 99999 after update
    if((minVelocityX == 999) or (maxVelocityX == -999) or (minVelocityY == 999) or (maxVelocityY == -999) or (maxRealtiveX == -9999) or (maxRealtiveY == -9999) or (minRealtiveX == 9999) or (minRealtiveY == 9999)):
        print('Min max values are not porperly updated in the UpdateMinMax function!!!')
        sys.exit()


if __name__ == '__main__':


    # Create the folder to save the Train, Val and Test Data
    os.mkdir(folderName)

    # Prepare the Train data
    CreateData(trainTrajFilePath, trainStr)

    # Prepare the Val data
    CreateData(valTrajFilePath, validationStr)

    # Prepare the Test data
    CreateData(testTrajFilePath, testStr)

    print('All data processed!!!')
    sys.exit()

