###################################################################################################################
# The Retrain technique with I-80 data and new maneuver class in the big screen server with the mixed data created
# by FilterNetworkV20.py in this folder (junction). Mixed means usiing two seperate csv for train and test data. This way we can use both TV 
# and all SVs to add in the train data for training and also for the RMSE calculations as well to have better results
# without causing any overlap between the training and testing vehicles. 
# The additonal headway space and headway time is also added for better lane change estimation and eventually better 
# trajectory. The Sample count and vehicle counts are mentione below  
# Train on 843184 samples (1-575 vehicles), validate on 262540 samples (576-695 vehicles) RAM (129/21.6%)
# *** Due to random straight vehicles the above numbers may not match exactly all the time. ****
###################################################################################################################
# Still clear case of over-fitting. Try to change the poistion module from layers to the one currently being used
# "using the predicted motion at each time instance to calculate the position" one. Simple copy from any of the 
# FilterNetworkV*.py file from the FilterNet folder. Theortically the changes should only be in the mdoel and not in the 
# prediction code as the model will still predict the position and motion the only difference is it won't predict inside the model
# rather calculate. So only chage in the model not in the prediction but still double check. 
# Also since the model will be new we have to start it from scratch and not used previous trained models and start from
# the loops directly 
# Now this is first time we will be using the motion based formulation inside the network in the retrain code so we can try
# both reload the model weights and do not load the model weights after each cycle. Try for 3-4 cycles for both the cases. 
# In the second case ("Do not reload the weights" idea) it wasn't atlest over fitting but not improving either. So worth
# a shot. But make sure to run both cases for 3-4 cycles
# Lastly from the position_loss it is clear that it is over-fitted but also from the velocity_loss we can see that it is overfitted as well
# so removing the postion layers and only training velocity will probably won't change anything as currently the velocity is overfitted
# as well but still worth a shot!!!!   
# Motion based position layer from the scratch with reloading the weights after each cycle, jump 30 (for lower memory) -> Crazy bad, close to 30+
# Plus the normalization issue is still there, Fixing it first...
# Normalization issue almost fixed, trying one run..... -> see for any errors......
# Normalization issue almost fixed, after second round decoder val array >1.007.. In decoder this is expected as more it trains
# there are chances it will pick different vehicles and hence sightily higher/lower than the min max estimated. True the model is not being used 
# during the min maz estimatin then why it is model dependent. Cause right now ground truth values are used to estimate the min max, which might 
# not be same when done live via different trained models. The error message was this, expected:
# Second rouund data preperation fininshed!!!
# XVal max :1.0
# decoderValInput max :1.0074062094810272
# We can use the predicted model it get the decoder min max first, the re predict, but might be of not any use as can be a dead-end loop.
# Re predict, check min max and then repredict again the same set, looot effort plus since the min max is not updated the current prediction won't
# be exact same, so of no use, Plus not that higher than 1, like 0.007.  
# Next time try without relaoding the model after each cycle
# If none of these work, it is worth trying replacing the motion based trajectory module to normal dense layer based position module
# inside the network, basically the old style.... Take it from any of the old file. As the raw network is better comapre to the motion
# based one in some recent cases, so maybe it would be worth one try after the normalization issue being fixed.. 
# Trying the relaod the old model weight idea --> Still overfitting and error close to 6.5
# Error file name -> ModifiedDecoderV10Run31.txt, Most likely overfitting after 5-6 runs, but both train and test were going down eventually
# so may be worth trying another round with more runs to see if the test is actually going more down or simple overfitting for train data. 
# Trying the do not relaod the model weight idea (with hope for less overfitting) --> running....
# After one round decoderTrainInput max :1.1404774163141373, most probably due to dicrepency between the ground-truth and predicted entities
# in the decoder input. Currently 'not reload weight' with 'motion based netwrok' is on hold, Try the raw arch with multiple rounds first both
# with reload and not reload weight setting.
###################################################################################################################
# Old architecure with not reload weight setting batch 1024, step 0.001/7.0/0.5 and 0.002/3.0/0.5 (both) -> not good enough
# All same reloading the weight -> Error close to 4.5. Not good enough results and clear overfitting. Error are in the following files.
# resultFileName = '/media/sdd/sap/Junction/results/ModifiedDecoderV13Run5.txt'
# historyFileName = '/media/sdd/sap/Junction/results/ModifiedDecoderV13LossRun5.txt'

# Trying the Navtech Model with both dropout and batchnorm -> Best 4 till now....
# The results are great so far already 4.03 and the trend is still down. Worth keep it running for some more iteration to see 
# how low it can get. The intermediate reuslts are being stored in the following files and still running:
# resultFileName = '/media/sdd/sap/Junction/results/ModifiedDecoderV13Run7.txt'
# historyFileName = '/media/sdd/sap/Junction/results/ModifiedDecoderV13LossRun7.txt' 
# The model number 5

# Navtech model with one more batchnorm + dropout layer set and higher dropout 0.25 insted of 0.2 -> running.. (from start)
# The same above model is also running on the Sen Server.

# Trying with the BatchNorm layers only in maneuver and motion modules to make it generalized -> to be done.. 

# Trying with the BatchNorm layers in all three modules to make it more generalized -> to be done

# Next try with the dropout layer -> to be done.....

# Trying with extra batchnorm and dropout -> runnig...

###################################################################################################################
# To be Done:
# Next time try the motion based arch, and just two blocks, manuevr and motion Vx and Vy together and calculated traj. This will reduce the model
# size so maybe it will fit with 2048 batch size. With bigger data and 2048 batch worth a shot....  
# Try the old architecture, raw layers not motion basd position...
###################################################################################################################

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
trainTrajFilePath = '/media/sdd/sap/Junction/I80RawData/0400pm-0415pm/trajectories-0400-0415.csv'
testTrajFilePath =  '/media/sdd/sap/Junction/I80RawData/0515pm-0530pm/trajectories-0515-0530.csv'
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
folderName = '/media/sdd/sap/FilterNet/data/I80TrainValTestMixed1024V1'    # I804SurroundingMixed1024Small  I804SurroundingMixed1024V1  I804SurroundingMixed2048V1  I80TrainValTestMixed1024V1
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
encoderModelFilename = '/media/sdd/sap/Junction/models/ModV10Encoder13.h5'
decoderModelFilename = '/media/sdd/sap/Junction/models/ModV10Decoder13.h5'
mainModelFileName  = '/media/sdd/sap/Junction/models/ModV10MainModel13.h5'


# Specify the result file to store each sample error last one (ModifiedDecoderV5Run15.txt, ModifiedDecoderV5LossRun1.txt)
resultFileName = '/media/sdd/sap/Junction/results/ModifiedDecoderV13Run17.txt'
historyFileName = '/media/sdd/sap/Junction/results/ModifiedDecoderV13LossRun17.txt'
f = open(resultFileName, 'x')
f.close()
h = open(historyFileName, 'x')
h.close()

# Specify the validation vehicle file name
validationFileName = folderName + '/' + 'validation.txt'

# Specify the training vehicle file name
trainingFileName = folderName + '/' + 'training.txt'

# Train and validation data folder names (DO NOT CHANGE)
trainFolderName = '/TrainData'
valFolderName = '/ValidationData'

# Create the visible window
# cv2.namedWindow('test', cv2.WINDOW_NORMAL)

# Train and Validation process lists
manager = Manager()
trainProcessList = manager.list()
validationProcessList = manager.list()

# To keep count of number of sample processed
countList = manager.list()
minMaxCountList = manager.list()

# To count the manager list
errorManagerList = manager.list()
errorCountList = manager.list()

# Add 20-25 straight vehicles to see the histogram and other distributions
straightVehicles = manager.list()
includedStraightVehicles = 10    

# # To keep the max relative X and Y for later use
# maxRelativeXY = manager.list()

# Model parametrs 
numberOfTrainingLoop = 10 
batchSize = 1024  # 1024    2048  # 1024   #256  128
initialNumberEpochs = 60    #40   #40   #  30   #   30  #30
secondNumberEpochs = 65     # 180  # 80
historyTemporal = 30   #30
futureTemporal = 50   #50
surroudingCarCounts = 4    # 4

inputFeatureCount = 6  # 6 -> [localX,localY,velocityX,velocityY,laneID,movement]  # headwaySpace,HeadwayTime]
globalInputFeatures = (surroudingCarCounts+1)*inputFeatureCount  
globalOutputFeatures = 7                          # 7 -> (poseX,poseY,velocityX,velocityY,Class0,Class1,Class2)
decoderFeatureCount = 7 # output 
globalDecoderFeatures = (surroudingCarCounts+1)*decoderFeatureCount 

# Input and decoder padding for non eligible surrpounding vehicles
inputZeroPadding = [0,0,0,0,0,0]
decoderZeroPadding = [0,0,0,0,0,0,0]


leakyAlphaValue = 0.1      # 0.5
maximumAllowabelJuncDist = 250     #(250 Feet)
maximumSurroundingCarDist = 80   #  80   # 80     #(25 Feet)  # as this is straight road increaed the surroudning dist     40     #(25 Feet)
predictionDistanceThreshold = 250  #(100 Feet )
ignoreFrameCount = 100
classOut = 3
poseOut = 2
velcoityOut = 2
n_units = 256
dropOutFrac = 0.3

# Number of previous and next frames to be considered for the maneuver estimatation
prevNextFrameCount = 20

# Validation vehiles
totalVehileCount = 2000
validationVehicleCount = 300

# Min Max values for normalize or denormalize
minVelocityX = 999
maxVelocityX = -999
minVelocityY = 999
maxVelocityY = -999
maxRealtiveX = -9999
maxRealtiveY = -9999
minRealtiveX = 9999
minRealtiveY = 9999
minHeadwaySpace = 999
maxHeadwaySpace = -999
minHeadwayTime = 999
maxHeadwayTime = -999

# Manager list to hold all the relative poseX\Y plus velocityX\Y which will later be used for min max value
# identification and normalization
relativePoseMotionXYManegerList = manager.list()

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

# File types for each sample
sampleFileNameList = ['/finalX.txt','/finalDecoderInput.txt','/finalYClass.txt','/finalYVel.txt','/finalYPose.txt']

# Unit constants
feetToMeter = 0.3048

# csv feature count
csvFeatureCount = 18

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

# ELU activation with a very small addition to help prevent NaN in loss.
def elu_plus_one_plus_epsilon(x):

    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import ELU, Add

    # ELU activation with a very small addition to help prevent NaN in loss.
    # Add epsilon (1e-7) and 1 to the ELU out
    ELULayer = ELU(alpha=leakyAlphaValue)
    ELUOut = ELULayer(x)

    # # # epsilonOffset = tf.fill(tf.shape(ELUOut),K.epsilon())
    # # # unitOffset = tf.fill(tf.shape(ELUOut),1.0)

    # # # # Add all three
    # # # addLayer = Add()
    # # # offsetELUOut = addLayer([ELUOut,epsilonOffset,unitOffset])

    return ELUOut


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

    # # # # minLocalX = min(datasetArray[:,localXIndex])
    # # # # maxLocalX = max(datasetArray[:,localXIndex])

    # # # # minLocalY = min(datasetArray[:,localYIndex])
    # # # # maxLocalY = max(datasetArray[:,localYIndex])

    # # # # minVel = min(datasetArray[:,velocityIndex])
    # # # # maxVel = max(datasetArray[:,velocityIndex])

    # # # # minAcc = min(datasetArray[:,accIndex])
    # # # # maxAcc = max(datasetArray[:,accIndex])

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

# # # # # Pass the surrounding vechiles and current input list. It will extend the list with surrouding cars info.
# # # # def GetSurroundingCarsInfo(otherVechiles, tempInput, targetVehicleID, inputOrDecoder, localX, localY, initialX, initialY):

# # # #     # remove surrounding vehicles with 0 intersectionID and 0 sectionID #############################################3
# # # #     # popList = []
# # # #     # for odx,eachOtherVechiles in enumerate(otherVechiles):
# # # #     #     section = eachOtherVechiles[sectionIndex]
# # # #     #     intersection = eachOtherVechiles[intersectionIndex]
# # # #     #     if(section==0 and intersection==0):
# # # #     #         popList.append(odx)
    
# # # #     # for eachPopItem in popList:
# # # #     #     otherVechiles.pop(eachPopItem)

# # # #     otherVechilesCount = len(otherVechiles)

# # # #     # Atleast target vehicle shoudl be present in other vehicles
# # # #     if(otherVechilesCount < 1):
# # # #         print('Target vehicle not present in other vehicle')
# # # #         print('otherVechilesCount = ' + str(otherVechilesCount))
# # # #         sys.exit()

# # # #     # Target Vehicle should Present in Other vehicles
# # # #     otherVehicleArray = np.array(otherVechiles).reshape(otherVechilesCount,csvFeatureCount)
# # # #     otherIds = list(otherVehicleArray[:,vechileIDIndex])

# # # #     if(float(targetVehicleID) not in otherIds):
# # # #         print('Target Vehicle not is other list')
# # # #         print('Target Vehicle ID ' + targetVehicleID)
# # # #         print('Other IDs')
# # # #         print(otherIds)

# # # #     paddingCount = surroudingCarCounts + 1 - otherVechilesCount

# # # #     # Vehicle should be removed once due to the presence of target vehicle 
# # # #     removedFlag = 0

# # # #     # If other vehicle count is less than 5 (4 surronding + 1 target as it will present in the frame based list)
# # # #     # append all the vechiles info into input list. 
# # # #     if (otherVechilesCount <= (surroudingCarCounts+1)):
# # # #         # Process the gathered surrounding cars
# # # #         for eachOtherVechiles in otherVechiles:
# # # #             otherVehicleID = str(eachOtherVechiles[vechileIDIndex])
# # # #             # Ignore the target vechile as it is already added
# # # #             if(otherVehicleID == targetVehicleID):
# # # #                 removedFlag = removedFlag + 1
# # # #                 continue

# # # #             otherLocalX = eachOtherVechiles[localXIndex]
# # # #             otherLocalY = eachOtherVechiles[localYIndex]
# # # #             otherVelocity = eachOtherVechiles[velocityIndex]
# # # #             otherLaneID = eachOtherVechiles[laneIDIndex]
# # # #             otherDirection = eachOtherVechiles[directionIndex]
# # # #             otherMovement = eachOtherVechiles[movementIndex]
# # # #             otherRelativeX = abs(otherLocalX - initialX) # Added for the relative position
# # # #             otherRelativeY = abs(otherLocalY - initialY) # Added for the relative position
# # # #             # Nearest junction distance
# # # #             currentSection = eachOtherVechiles[sectionIndex]
# # # #             currentIntersection = eachOtherVechiles[intersectionIndex]
# # # #             juncDist = CalculateNearestJuncLoc(currentSection, currentIntersection, otherLocalX, otherLocalY)

# # # #             # If the other vehicle distance is more that allowable then append zeros
# # # #             otherDist = math.sqrt((((otherLocalX-localX)**2)+((otherLocalY-localY)**2)))

# # # #             if(inputOrDecoder == inputStr):
# # # #                 if(otherDist>maximumSurroundingCarDist):
# # # #                     tempInput.extend([0,0,0,0,0,0,0])
# # # #                 else:
# # # #                     tempInput.extend([otherRelativeX,otherRelativeY,otherVelocity,otherLaneID,otherDirection,otherMovement,juncDist])
# # # #             elif(inputOrDecoder == decoderStr):
# # # #                 if(otherDist>maximumSurroundingCarDist):
# # # #                     tempInput.extend([0,0,0,0,0,0,0])
# # # #                 else:
# # # #                     lastInputClassInfo = MovementToClassForm(otherMovement)
# # # #                     tempInput.extend([otherRelativeX,otherRelativeY,otherVelocity,lastInputClassInfo[0],lastInputClassInfo[1],lastInputClassInfo[2],juncDist])
# # # #             else:
# # # #                 print('Unknown inputOrDecoder string : ' +  inputOrDecoder)
# # # #                 sys.exit()
        
# # # #         # Append the zero padding based on the required padding count calculated using other vechile count and decided surrouning car count
# # # #         # As the number of input features and decoder input/output is same that is why we used the same zero padding width
# # # #         zeroList = []
# # # #         if(inputOrDecoder == inputStr):
# # # #             zeroList = [0,0,0,0,0,0,0]
# # # #         elif(inputOrDecoder == decoderStr):
# # # #             zeroList = [0,0,0,0,0,0,0]
# # # #         else:
# # # #             print('Unknown inputOrDecoder string : ' +  inputOrDecoder)
# # # #             sys.exit()

# # # #         for rdx in range(0,paddingCount):          
# # # #             tempInput.extend(zeroList)

# # # #     # Else the vechile count is more than 4. So select the nearest 4 vechicles.
# # # #     else:
# # # #         # Gather distance of each car from the target car
# # # #         otherCarIndexedDistanceList = []
# # # #         for sdx, eachOtherVechiles in enumerate(otherVechiles):
# # # #             otherVehicleID = str(eachOtherVechiles[vechileIDIndex])
# # # #             # Ignore the target vechile as the distance would be zero
# # # #             if(otherVehicleID == targetVehicleID):
# # # #                 removedFlag = removedFlag + 1
# # # #                 continue

# # # #             otherLocalX = eachOtherVechiles[localXIndex]
# # # #             otherLocalY = eachOtherVechiles[localYIndex]

# # # #             # Calculate distance of each other car append in the list with index value
# # # #             otherDist = math.sqrt((((otherLocalX-localX)**2)+((otherLocalY-localY)**2)))
# # # #             otherCarIndexedDistanceList.append([sdx,otherDist])
        
# # # #         # Sort the list based on distance and gather the lowest indexes
# # # #         otherCarIndexedDistanceList = sorted(otherCarIndexedDistanceList,key=lambda x: x[1])
# # # #         otherCarIndexedDistanceArray = np.array(otherCarIndexedDistanceList)
# # # #         releventOtherIndexes = otherCarIndexedDistanceArray[0:surroudingCarCounts,0:1]

# # # #         # Append other car infos to the temp input based on the decided index
# # # #         for eachReleventIndex in releventOtherIndexes:
# # # #             otherLocalX = otherVechiles[int(eachReleventIndex)][localXIndex]
# # # #             otherLocalY = otherVechiles[int(eachReleventIndex)][localYIndex]
# # # #             otherVelocity = otherVechiles[int(eachReleventIndex)][velocityIndex]
# # # #             otherLaneID = otherVechiles[int(eachReleventIndex)][laneIDIndex]
# # # #             otherDirection = otherVechiles[int(eachReleventIndex)][directionIndex]
# # # #             otherMovement = otherVechiles[int(eachReleventIndex)][movementIndex]
# # # #             otherRelativeX = abs(otherLocalX - initialX) # Added for the relative position
# # # #             otherRelativeY = abs(otherLocalY - initialY) # Added for the relative position
# # # #             # Nearest junction distance
# # # #             currentSection =  otherVechiles[int(eachReleventIndex)][sectionIndex]
# # # #             currentIntersection = otherVechiles[int(eachReleventIndex)][intersectionIndex]
# # # #             juncDist = CalculateNearestJuncLoc(currentSection, currentIntersection, otherLocalX, otherLocalY)

# # # #             # If the other vehicle distance is more than allowable then append zeros
# # # #             otherDist = math.sqrt((((otherLocalX-localX)**2)+((otherLocalY-localY)**2)))
            
# # # #             if(inputOrDecoder == inputStr):
# # # #                 if(otherDist>maximumSurroundingCarDist):
# # # #                     tempInput.extend([0,0,0,0,0,0,0])
# # # #                 else:
# # # #                     tempInput.extend([otherRelativeX,otherRelativeY,otherVelocity,otherLaneID,otherDirection,otherMovement,juncDist])
# # # #             elif(inputOrDecoder == decoderStr):
# # # #                 if(otherDist>maximumSurroundingCarDist):
# # # #                     tempInput.extend([0,0,0,0,0,0,0])
# # # #                 else:
# # # #                     lastInputClassInfo = MovementToClassForm(otherMovement)
# # # #                     tempInput.extend([otherRelativeX,otherRelativeY,otherVelocity,lastInputClassInfo[0],lastInputClassInfo[1],lastInputClassInfo[2],juncDist])

# # # #     if(removedFlag != 1):
# # # #         print('Vehicle not removed propoerly for vehicle ID ' + str(targetVehicleID))
# # # #         print('removedFlag = ' + str(removedFlag))


# # # #     if(inputOrDecoder == inputStr):
# # # #         if(len(tempInput) != globalInputFeatures):
# # # #             print('Unwanted input feature in GetSurroundingCarsInfo is : ' + str(len(tempInput)))
# # # #     elif(inputOrDecoder == decoderStr):
# # # #         if(len(tempInput) != globalDecoderFeatures):
# # # #             print('Unwanted decoder feature in GetSurroundingCarsInfo is : ' + str(len(tempInput)))
# # # #     else:
# # # #         print('Unknonw inputOrDecoder string : ' + str(inputOrDecoder))

# # # #     return tempInput

# # # # def ProcessByVehicle(processItme):

# # # #     currentID = processItme[0]   # Updated key string
# # # #     currentTrainOrValStr = processItme[1] # strig
# # # #     targetVehicleID = processItme[2]   # original key string

# # # #     if(targetVehicleID == None):
# # # #         print('Traget Vehicle ID is none!!!!')

# # # #     currentVehicleList = dictByVehicles[currentID]

# # # #     # Add the check for the side origins and side destination
# # # #     sideOrigin = currentVehicleList[0][originIndex]
# # # #     sideDestination = currentVehicleList[0][destinationIndex]
# # # #     if ((sideOrigin == 111 and sideDestination == 203) or (sideOrigin == 103 and sideDestination == 211) or (sideOrigin == 105 and sideDestination == 210) or (sideOrigin == 110 and sideDestination == 205) or (sideOrigin == 107 and sideDestination == 209) or (sideOrigin == 109 and sideDestination == 207)):
# # # #         return

# # # #     # and straight to straight vehicles
# # # #     if ((sideOrigin == 101 and sideDestination == 208) or (sideOrigin == 108 and sideDestination == 201)):
# # # #         return

# # # #     # Add check of the section and intersection both IDs are zero. Not expected behaviour
# # # #     initialSectionID = currentVehicleList[0][sectionIndex]
# # # #     initialIntersectionID = currentVehicleList[0][intersectionIndex]
# # # #     if(initialSectionID == 0 and initialIntersectionID == 0):
# # # #         return

# # # #     currentVehicleLength = len(currentVehicleList)

# # # #     for idx in range(historyTemporal,currentVehicleLength-futureTemporal):

# # # #         # Get the current vehicles as those are only eligible from prediction point of view 
# # # #         # vehicles appearing in first frame of the target vehicle
# # # #         currentTargetTime = currentVehicleList[idx-historyTemporal][globalTimeIndex]
# # # #         currentOtherVehicles = dictByFrames[str(currentTargetTime)]
# # # #         currentOtherEligibleVehicles = []
# # # #         for eachCurrentOtherVehicles in currentOtherVehicles:
# # # #             currentOtherID = eachCurrentOtherVehicles[vechileIDIndex]
# # # #             # # Update other Id in case it is present in mapper
# # # #             # if(str(currentOtherID) in mapper):
# # # #             #     updatedID = mapper[str(currentOtherID)]
# # # #             #     currentOtherID = updatedID
# # # #             if(currentOtherID == float(targetVehicleID)):
# # # #                 currentOtherEligibleVehicles.append(currentOtherID)
# # # #                 continue
# # # #             # vehicles having history + future temporal frames.
# # # #             currentOtherFrame = eachCurrentOtherVehicles[frameIDIndex]
# # # #             currentOtherTotalFrame = eachCurrentOtherVehicles[totoalFrameIndex]
# # # #             remainingFrames = currentOtherTotalFrame - currentOtherFrame
# # # #             if(remainingFrames>= historyTemporal+futureTemporal):
# # # #                 currentOtherEligibleVehicles.append(currentOtherID)

# # # #         # Prepeare sequential Input Data
# # # #         localXData = []
# # # #         initialLocalX = currentVehicleList[idx-historyTemporal][localXIndex]
# # # #         initialLocalY = currentVehicleList[idx-historyTemporal][localYIndex]
# # # #         for jdx in range(idx-historyTemporal,idx):
# # # #             tempInput = []
# # # #             absoluteX = currentVehicleList[jdx][localXIndex]
# # # #             absoluteY = currentVehicleList[jdx][localYIndex]
# # # #             localX = abs(absoluteX - initialLocalX)
# # # #             localY = abs(absoluteY - initialLocalY)
# # # #             velocity = currentVehicleList[jdx][velocityIndex]
# # # #             laneID = currentVehicleList[jdx][laneIDIndex]
# # # #             direction = currentVehicleList[jdx][directionIndex]
# # # #             movement = currentVehicleList[jdx][movementIndex]

# # # #             # Nearest junction distance
# # # #             currentSection = currentVehicleList[jdx][sectionIndex]
# # # #             currentIntersection = currentVehicleList[jdx][intersectionIndex]
# # # #             juncDist = CalculateNearestJuncLoc(currentSection, currentIntersection, absoluteX, absoluteY)

# # # #             tempInput = [localX,localY,velocity,laneID,direction,movement,juncDist]

# # # #             # Prepare the surrounding cars information
# # # #             # Gather vehicles using the same frame using the Frame Dict
# # # #             currentInputFrame = currentVehicleList[jdx][frameIDIndex]
# # # #             currentInputTime = currentVehicleList[jdx][globalTimeIndex]
# # # #             otherVechiles = dictByFrames[str(currentInputTime)]

# # # #             # Remove the prediction not eligible vehicles
# # # #             eligibleOtherVehicles = []
# # # #             for eachOtherVehicle in otherVechiles:
# # # #                 otherID = eachOtherVehicle[vechileIDIndex]
# # # #                 if (otherID in currentOtherEligibleVehicles):
# # # #                     eligibleOtherVehicles.append(eachOtherVehicle)

# # # #             # Remove vehicles with a different global time which is not possible. Just adding check to be sure
# # # #             for fdx,eachOtherTime in enumerate(otherVechiles):
# # # #                 otherTime = eachOtherTime[globalTimeIndex]
# # # #                 if (otherTime != currentInputTime):
# # # #                     print('Mismatch in input global time..')
# # # #                     print('other Time ' + str(otherTime))
# # # #                     print('Current Time ' + str(currentInputTime))
# # # #                     sys.exit()

# # # #             # Extend the surrounding cars info into the target vehicles input   otherVechiles replaced by  eligibleOtherVehicles
# # # #             tempInput = GetSurroundingCarsInfo(eligibleOtherVehicles, tempInput, targetVehicleID, inputStr, absoluteX, absoluteY, initialLocalX, initialLocalY)

# # # #             if (len(tempInput) != globalInputFeatures):
# # # #                 print('tempInput len is : ' + str(len(tempInput)) + ' instead of ' + str(globalInputFeatures))
# # # #                 sys.exit()

# # # #             # Add the final list of target vehicles and other vehicles info into the local input
# # # #             localXData.append(tempInput)


# # # #         # Prepeare sequential Output Data and decoder input data
# # # #         localYMovementData = []
# # # #         localYVelData = []
# # # #         localYPoseData = []
# # # #         decoderInputData = []

# # # #         # Prepare the First Decoder Input
# # # #         lastInput = localXData[-1]
# # # #         firstDecoderInput = []
# # # #         for tdx in range(0,len(lastInput),inputFeatureCount):
# # # #             lastInputPoseX = lastInput[tdx]
# # # #             lastInputPoseY = lastInput[tdx+1]
# # # #             lastInputVelocity = lastInput[tdx+2]
# # # #             lastInputMovement = lastInput[tdx+5]
# # # #             lastInputClassInfo = MovementToClassForm(lastInputMovement)
# # # #             # Calculate the distance from the junction for the first decoder input 
# # # #             # For section, intersection, absoluteX and absoluteY use the last updated varibale as they hold the info for the last frame.
# # # #             juncDist = CalculateNearestJuncLoc(currentSection, currentIntersection, absoluteX, absoluteY)
# # # #             firstDecoderInput.extend([lastInputPoseX,lastInputPoseY,lastInputVelocity,lastInputClassInfo[0],lastInputClassInfo[1],lastInputClassInfo[2],juncDist])


# # # #         for kdx in range(idx,idx+futureTemporal):

# # # #             # Add the ground truth outputs
# # # #             nextMovement = currentVehicleList[kdx][movementIndex]
# # # #             nextMovementClassData = MovementToClassForm(nextMovement)

# # # #             localYMovementData.append(nextMovementClassData)
            
# # # #             nextVelocity = currentVehicleList[kdx][velocityIndex]
# # # #             deNormalizedNextVelocity = (nextVelocity*(maxVel-minVel))+minVel

# # # #             nextLocalX = currentVehicleList[kdx][localXIndex]
# # # #             nextRelativeX = abs(nextLocalX - initialLocalX)
# # # #             # denormalizedNextLocalX = (nextLocalX*(maxLocalX-minLocalX)+minLocalX) # no need to denormalize as it is not normalized
# # # #             nextLocalY = currentVehicleList[kdx][localYIndex]
# # # #             nextRelativeY = abs(nextLocalY - initialLocalY)
# # # #             # denormalizedNextLocalY = (nextLocalY*(maxLocalY-minLocalY)+minLocalY) # no need to denormalize as it is not normalized

# # # #             localYVelData.append([deNormalizedNextVelocity])
# # # #             localYPoseData.append([nextRelativeX,nextRelativeY])

# # # #             # Add the decoder input
# # # #             # Add the distance from the junc in the decoder as well
# # # #             nextSection = currentVehicleList[kdx][sectionIndex]
# # # #             nextIntersection = currentVehicleList[kdx][intersectionIndex]
# # # #             juncDist = CalculateNearestJuncLoc(nextSection, nextIntersection, nextLocalX, nextLocalY)

# # # #             decoderTemp = [nextRelativeX,nextRelativeY,nextVelocity,nextMovementClassData[0],nextMovementClassData[1],nextMovementClassData[2],juncDist]

# # # #             # Prepare the surrounding cars information for decoder input   # for decoder pass only the vehicles present in the last 30 frames..(not done....)
# # # #             # Gather vehicles using the same frame using the Frame Dict
# # # #             currentInputFrame = currentVehicleList[kdx][frameIDIndex]
# # # #             currentInputTime = currentVehicleList[kdx][globalTimeIndex]
# # # #             otherVechiles = dictByFrames[str(currentInputTime)]

# # # #             # Remove the prediction not eligible vehicles
# # # #             # Identify vehicles not in eligible list
# # # #             eligibleOtherVehicles = []
# # # #             for eachOtherVehicle in otherVechiles:
# # # #                 otherID = eachOtherVehicle[vechileIDIndex]
# # # #                 if (otherID in currentOtherEligibleVehicles):
# # # #                     eligibleOtherVehicles.append(eachOtherVehicle)

# # # #             # Remove vehicles with a different global time. Which is not possible. Just to double check
# # # #             for gdx,eachOtherTime in enumerate(otherVechiles):
# # # #                 otherTime = eachOtherTime[globalTimeIndex]
# # # #                 if (otherTime != currentInputTime):
# # # #                     print('Mismatch in decoder global time..')
# # # #                     print('other Time ' + str(otherTime))
# # # #                     print('Current Time ' + str(currentInputTime))
# # # #                     sys.exit()


# # # #             # Extend the surrounding cars info into the target vehicles decoder input   ##  otherVechiles replacd by eligibleOtherVehicles 
# # # #             decoderTemp = GetSurroundingCarsInfo(eligibleOtherVehicles, decoderTemp, targetVehicleID, decoderStr, nextLocalX, nextLocalY, initialLocalX, initialLocalY)

# # # #             # Check the decoder feature length
# # # #             if (len(decoderTemp) != globalDecoderFeatures):
# # # #                 print('decoderTemp len is : ' + str(len(decoderTemp)) + ' instead of ' + str(globalDecoderFeatures))
# # # #                 sys.exit()

# # # #             # Finally append the target car and surrounding cars info for the current frame into the final decoded input
# # # #             decoderInputData.append(decoderTemp)


# # # #         # Append in the final validation or training set based on decided vehicle ID
# # # #         if(currentTrainOrValStr == 'Validation'):
# # # #             # Shift one time stamp right and append Last input at the beggining 
# # # #             decoderInputData = decoderInputData[:-1]
# # # #             decoderInputData.insert(0,firstDecoderInput)

# # # #             validationProcessList.append([localXData,decoderInputData,localYMovementData,localYVelData,localYPoseData])
# # # #         elif(currentTrainOrValStr == 'Train'):
# # # #             # Shift one time stamp right and append last input at the beggining 
# # # #             decoderInputData = decoderInputData[:-1]
# # # #             decoderInputData.insert(0,firstDecoderInput)

# # # #             trainProcessList.append([localXData,decoderInputData,localYMovementData,localYVelData,localYPoseData])
# # # #         else:
# # # #             print('Unknown Train Val string')
# # # #             sys.exit()
        

# # # #         if((np.array(localXData).shape[0] != historyTemporal) or (np.array(localXData).shape[1] != globalInputFeatures)):
# # # #             print('localXData/Input Array Shape : ')
# # # #             print(np.array(localXData).shape)
# # # #             sys.exit()
        
# # # #         if((np.array(decoderInputData).shape[0] != futureTemporal) or (np.array(decoderInputData).shape[1] != globalDecoderFeatures)):
# # # #             print('decoderInputData Array Shape : ')
# # # #             print(np.array(decoderInputData).shape)
# # # #             sys.exit() 


# # # #     countList.append(0)
# # # #     totalSamplesProcessed = len(countList)
# # # #     print('Finished Processing Sample : ' + str(totalSamplesProcessed))

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

# # # # # Plot all cars trajectory on the global GPS map
# # # # def TrainData(inputFileName):

# # # #     # Load the Vehicle and Frame based Dictionaries
# # # #     global dictByFrames, dictByVehicles, validationVehicles, mapper, trainProcessList, validationProcessList
# # # #     dictByFrames,dictByVehicles,unusedMapperDict = CreateVehicleAndFrameDict(inputFileName)
# # # #     finalVehicleKeys = list(dictByVehicles.keys())
# # # #     finalVehicleKeys.sort(key=float)
# # # #     finalFrameKeys = list(dictByFrames.keys())
# # # #     finalFrameKeys.sort(key=float)

# # # #     # Pin the new process to specified cores
# # # #     os.system("taskset -p -c 1,2,3,4,5,6,7,8,9,10,11 %d" % os.getpid()) 
# # # #     # os.system("taskset -p -c 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 %d" % os.getpid())
# # # #     processes = []
# # # #     numberofCores = 10

# # # #     pool = mp.Pool(numberofCores)

# # # #     print('Total Vehicle : ' + str(len(finalVehicleKeys)))

# # # #     selectedVehilces = random.sample(finalVehicleKeys,totalVehileCount)

# # # #     # Randomly sample 300 validation vehicles from selected vehicles
# # # #     validationList = random.sample(selectedVehilces,validationVehicleCount)

# # # #     # Write the validation vehicles to the data folder
# # # #     validationFileObj = open(validationFileName, 'x')
# # # #     for eachValidationCar in validationList:
# # # #         validationFileObj.write("%s\n" % eachValidationCar)
# # # #     validationFileObj.close()


# # # #     processList = []

# # # #     for eachKey in selectedVehilces:
# # # #         # If the value exists in mapper get the key as original ID of later surrounding car info target vehicle seperation   mydict.keys()[mydict.values().index(16)]
# # # #         originalID = None
# # # #         if(eachKey in unusedMapperDict.values()):
# # # #             valueList = list(unusedMapperDict.values())
# # # #             originalID = list(unusedMapperDict.keys())[valueList.index(eachKey)]
# # # #             # print('Original Key for ' + eachKey + ' is found and the Key is ' + str(originalID))
# # # #         else:
# # # #             originalID = eachKey

# # # #         processStr = ''
# # # #         if eachKey in validationList:
# # # #             processStr = validationStr
# # # #         else:
# # # #             processStr = trainStr
# # # #         processList.append([eachKey,processStr,originalID])

# # # #     pool.map(ProcessByVehicle,processList)
# # # #     # for eachItem in processList:
# # # #     #     ProcessByVehicle(eachItem)

    
# # # #     # Convert the Train manager list to normal list
# # # #     print('Converting the Train Manager list to normal lists.....')
# # # #     normalTrainList = list(trainProcessList)
# # # #     print('List converted!!!')

# # # #     # Prepare the final lists of train and validation data
# # # #     # Train final lists
# # # #     print('Prepering the individual lists')

# # # #     finalXTrain = [x[0] for x in normalTrainList]
# # # #     filePath = folderName + '/finalXTrain.txt'
# # # #     WriteToFile(filePath,finalXTrain)
# # # #     print('Finished XTrain Array!!!')

# # # #     finalTrainDecoderInput = [x[1] for x in normalTrainList]
# # # #     filePath = folderName + '/finalTrainDecoderInput.txt'
# # # #     WriteToFile(filePath,finalTrainDecoderInput)
# # # #     print('Finished decoderTrainInput Array!!!')

# # # #     finalYClassTrain = [x[2] for x in normalTrainList]
# # # #     filePath = folderName + '/finalYClassTrain.txt'
# # # #     WriteToFile(filePath,finalYClassTrain)
# # # #     print('Finished YClassTrain Array!!!')

# # # #     finalYVelTrain = [x[3] for x in normalTrainList]
# # # #     filePath = folderName + '/finalYVelTrain.txt'
# # # #     WriteToFile(filePath,finalYVelTrain)
# # # #     print('Finished finalYVelTrain Array!!!')

# # # #     finalYPoseTrain = [x[4] for x in normalTrainList]
# # # #     filePath = folderName + '/finalYPoseTrain.txt'
# # # #     WriteToFile(filePath,finalYPoseTrain)
# # # #     print('Finished finalYPoseTrain Array!!!')

# # # #     # Convert the Validation manager list to normal list
# # # #     print('Converting the Validation Manager list to normal lists.....')
# # # #     normalValList = list(validationProcessList)
# # # #     print('List converted!!!')

# # # #     # Validation final lists
# # # #     finalXVal = [x[0] for x in normalValList]
# # # #     filePath = folderName + '/finalXVal.txt'
# # # #     WriteToFile(filePath,finalXVal)
# # # #     print('Finished XVal Array!!!')

# # # #     finalValDecoderInput = [x[1] for x in normalValList]
# # # #     filePath = folderName + '/finalValDecoderInput.txt'
# # # #     WriteToFile(filePath,finalValDecoderInput)
# # # #     print('Finished decoderValInput Array!!!')

# # # #     finalYClassVal = [x[2] for x in normalValList]
# # # #     filePath = folderName + '/finalYClassVal.txt'
# # # #     WriteToFile(filePath,finalYClassVal)
# # # #     print('Finished YClassVal Array!!!')

# # # #     finalYVelVal = [x[3] for x in normalValList]
# # # #     filePath = folderName + '/finalYVelVal.txt'
# # # #     WriteToFile(filePath,finalYVelVal)
# # # #     print('Finished YVelVal Array!!!')

# # # #     finalYPoseVal = [x[4] for x in normalValList]
# # # #     filePath = folderName + '/finalYPoseVal.txt'
# # # #     WriteToFile(filePath,finalYPoseVal)
# # # #     print('Finished YPoseVal Array!!!')

# # # #     print('Finished All Array!!!')

# # # #     # Prepare the final Train arrays
# # # #     XTrain = np.array(finalXTrain)
# # # #     decoderTrainInput = np.array(finalTrainDecoderInput)
# # # #     YClassTrain = np.array(finalYClassTrain)
# # # #     YPoseTrain = np.array(finalYPoseTrain)
# # # #     YVelTrain = np.array(finalYVelTrain)

# # # #     # Prepare the final Validation arrays
# # # #     XVal = np.array(finalXVal)
# # # #     decoderValInput = np.array(finalValDecoderInput)
# # # #     YClassVal = np.array(finalYClassVal)
# # # #     YVelVal = np.array(finalYVelVal)
# # # #     YPoseVal = np.array(finalYPoseVal)

# # # #     # Print the shape of the Arrays
# # # #     filePath = folderName + '/arrayShapes.txt'
# # # #     with open(filePath, 'x') as fshape:
# # # #         fshape.write('XTrain shape : ' + str(XTrain.shape) + '\n')
# # # #         fshape.write('decoderTrainInput shape : ' + str(decoderTrainInput.shape) + '\n')
# # # #         fshape.write('YClassTrain shape : ' + str(YClassTrain.shape) + '\n')
# # # #         fshape.write('YPoseTrain shape : ' + str(YPoseTrain.shape) + '\n')
# # # #         fshape.write('YVelTrain shape : ' + str(YVelTrain.shape) + '\n')

# # # #         fshape.write('XVal shape : ' + str(XVal.shape) + '\n')
# # # #         fshape.write('decoderValInput shape : ' + str(decoderValInput.shape) + '\n')
# # # #         fshape.write('YClassVal shape : ' + str(YClassVal.shape) + '\n')
# # # #         fshape.write('YVelVal shape : ' + str(YVelVal.shape) + '\n')
# # # #         fshape.write('YPoseVal shape : ' + str(YPoseVal.shape) + '\n')

# # # #     return XTrain,decoderTrainInput,YClassTrain,YVelTrain,YPoseTrain,XVal,decoderValInput,YClassVal,YVelVal,YPoseVal


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
    initial_lrate = 0.001      # 0.001 for RMSProp and Adma 0.002 for Nadam
    drop = 0.5
    epochs_drop = 4.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    if lrate < 0.00001:
        lrate = 0.00001
    return lrate


# # # Create the model architecture 
# # # Model for used for the Navtech data
def ModelArch():

    # Saniy check Min max values should be same as -9999 or 99999 after update
    if((minVelocityX == 999) or (maxVelocityX == -999) or (minVelocityY == 999) or (maxVelocityY == -999) or (maxRealtiveX == -9999) or (maxRealtiveY == -9999) or (minRealtiveX == 9999) or (minRealtiveY == 9999)):
        print('Min max values are not porperly updated in the UpdateMinMax function!!!')
        sys.exit()

    import tensorflow as tf

    from tensorflow.keras import backend as K
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input,LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dropout, MaxPooling1D, Concatenate, subtract, Lambda, BatchNormalization, LeakyReLU, ELU, Add, Reshape, RepeatVector
    from tensorflow.keras import optimizers
    from tensorflow.keras.optimizers import RMSprop, Adam, Nadam
    from tensorflow.keras.losses import logcosh


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

    # # # Common batch norm
    # # commonBatchNorm = BatchNormalization()
    # # decoder_outputs = commonBatchNorm(decoder_outputs)

    # Decoder for ClassOut
    dropOut11 = Dropout(dropOutFrac)
    batchNorm11 = BatchNormalization()
    decoder_dense10a = Dense(1024)
    decoder_Leaky10a = ELU(alpha=leakyAlphaValue)
    decoder_output1 = decoder_dense10a(decoder_outputs)
    decoder_output1 = batchNorm11(decoder_output1)
    decoder_output1 = decoder_Leaky10a(decoder_output1)
    decoder_output1 = dropOut11(decoder_output1, training=True)

    dropOut12 = Dropout(dropOutFrac)
    batchNorm12 = BatchNormalization()
    decoder_dense10 = Dense(512)
    decoder_Leaky10 = ELU(alpha=leakyAlphaValue)
    decoder_output1 = decoder_dense10(decoder_output1)
    decoder_output1 = batchNorm12(decoder_output1)
    decoder_output1 = decoder_Leaky10(decoder_output1)
    decoder_output1 = dropOut12(decoder_output1, training=True)

    dropOut13 = Dropout(dropOutFrac)
    batchNorm13 = BatchNormalization()
    decoder_dense11 = Dense(256)
    decoder_Leaky11 = ELU(alpha=leakyAlphaValue)
    decoder_output1 = decoder_dense11(decoder_output1)
    decoder_output1 = batchNorm13(decoder_output1)
    decoder_output1 = decoder_Leaky11(decoder_output1)
    decoder_output1 = dropOut13(decoder_output1, training=True)

    dropOut14 = Dropout(dropOutFrac)
    batchNorm14 = BatchNormalization()
    decoder_dense12 = Dense(128)
    decoder_Leaky12 = ELU(alpha=leakyAlphaValue)
    decoder_output1 = decoder_dense12(decoder_output1)
    decoder_output1 = batchNorm14(decoder_output1)
    decoder_output1 = decoder_Leaky12(decoder_output1)
    decoder_output1 = dropOut14(decoder_output1, training=True)

    dropOut15 = Dropout(dropOutFrac)
    batchNorm15 = BatchNormalization()
    decoder_dense13 = Dense(64)
    decoder_Leaky13 = ELU(alpha=leakyAlphaValue)
    decoder_output1 = decoder_dense13(decoder_output1)
    decoder_output1 = batchNorm15(decoder_output1)
    decoder_output1 = decoder_Leaky13(decoder_output1)
    decoder_output1 = dropOut15(decoder_output1, training=True)

    dropOut16 = Dropout(dropOutFrac)
    batchNorm16 = BatchNormalization()
    decoder_dense14 = Dense(32)
    decoder_Leaky14 = ELU(alpha=leakyAlphaValue)
    decoder_output1 = decoder_dense14(decoder_output1)
    decoder_output1 = batchNorm16(decoder_output1)
    decoder_output1 = decoder_Leaky14(decoder_output1)
    decoder_output1 = dropOut16(decoder_output1, training=True)

    # dropOut17 = Dropout(dropOutFrac)
    # batchNorm17 = BatchNormalization()
    decoder_dense15 = Dense(16)
    decoder_Leaky15 = ELU(alpha=leakyAlphaValue)
    decoder_output1 = decoder_dense15(decoder_output1)
    # decoder_output1 = batchNorm17(decoder_output1)
    decoder_output1 = decoder_Leaky15(decoder_output1)
    # decoder_output1 = dropOut17(decoder_output1, training=True)

    # dropOut18 = Dropout(dropOutFrac)
    # batchNorm18 = BatchNormalization()
    decoder_dense16 = Dense(8)
    decoder_Leaky16 = ELU(alpha=leakyAlphaValue)
    decoder_output1 = decoder_dense16(decoder_output1)
    # decoder_output1 = batchNorm18(decoder_output1)
    decoder_output1 = decoder_Leaky16(decoder_output1)
    # decoder_output1 = dropOut18(decoder_output1, training=True)

    decoder_dense17 = Dense(3, activation='softmax', name='Class')
    classOut = decoder_dense17(decoder_output1)

    # Decoder for Velocity Out
    decoder2_concat = Concatenate()
    decoder_output2 = decoder2_concat([decoder_outputs,classOut])

    dropOut21 = Dropout(dropOutFrac)
    batchNorm21 = BatchNormalization()
    decoder_dense20a = Dense(1024)
    decoder_Leaky20a = ELU(alpha=leakyAlphaValue)
    decoder_output2 = decoder_dense20a(decoder_output2)
    decoder_output2 = batchNorm21(decoder_output2)
    decoder_output2 = decoder_Leaky20a(decoder_output2)
    decoder_output2 = dropOut21(decoder_output2, training=True)

    dropOut22 = Dropout(dropOutFrac)
    batchNorm22 = BatchNormalization()
    decoder_dense20 = Dense(512)
    decoder_Leaky20 = ELU(alpha=leakyAlphaValue)
    decoder_output2 = decoder_dense20(decoder_output2)
    decoder_output2 = batchNorm22(decoder_output2)
    decoder_output2 = decoder_Leaky20(decoder_output2)
    decoder_output2 = dropOut22(decoder_output2, training=True)

    dropOut23 = Dropout(dropOutFrac)
    batchNorm23 = BatchNormalization()
    decoder_dense21 = Dense(256)
    decoder_Leaky21 = ELU(alpha=leakyAlphaValue)
    decoder_output2 = decoder_dense21(decoder_output2)
    decoder_output2 = batchNorm23(decoder_output2)
    decoder_output2 = decoder_Leaky21(decoder_output2)
    decoder_output2 = dropOut23(decoder_output2, training=True)

    dropOut24 = Dropout(dropOutFrac)
    batchNorm24 = BatchNormalization()
    decoder_dense22 = Dense(128)
    decoder_Leaky22 = ELU(alpha=leakyAlphaValue)
    decoder_output2 = decoder_dense22(decoder_output2)
    decoder_output2 = batchNorm24(decoder_output2)
    decoder_output2 = decoder_Leaky22(decoder_output2)
    decoder_output2 = dropOut24(decoder_output2, training=True)

    dropOut25 = Dropout(dropOutFrac)
    batchNorm25 = BatchNormalization()
    decoder_dense23 = Dense(64)
    decoder_Leaky23 = ELU(alpha=leakyAlphaValue)
    decoder_output2 = decoder_dense23(decoder_output2)
    decoder_output2 = batchNorm25(decoder_output2)
    decoder_output2 = decoder_Leaky23(decoder_output2)
    decoder_output2 = dropOut25(decoder_output2, training=True)

    dropOut26 = Dropout(dropOutFrac)
    batchNorm26 = BatchNormalization()
    decoder_dense24 = Dense(32)
    decoder_Leaky24 = ELU(alpha=leakyAlphaValue)
    decoder_output2 = decoder_dense24(decoder_output2)
    decoder_output2 = batchNorm26(decoder_output2)
    decoder_output2 = decoder_Leaky24(decoder_output2)
    decoder_output2 = dropOut26(decoder_output2, training=True)

    # dropOut27 = Dropout(dropOutFrac)
    # batchNorm27 = BatchNormalization()
    decoder_dense25 = Dense(16)
    decoder_Leaky25 = ELU(alpha=leakyAlphaValue)
    decoder_output2 = decoder_dense25(decoder_output2)
    # decoder_output2 = batchNorm27(decoder_output2)
    decoder_output2 = decoder_Leaky25(decoder_output2)
    # decoder_output2 = dropOut27(decoder_output2, training=True)

    # dropOut28 = Dropout(dropOutFrac)
    # batchNorm28 = BatchNormalization()
    decoder_dense26 = Dense(8)
    decoder_Leaky26 = ELU(alpha=leakyAlphaValue)
    decoder_output2 = decoder_dense26(decoder_output2)
    # coder_output2 = batchNorm28(decoder_output2)
    decoder_output2 = decoder_Leaky26(decoder_output2)
    # decoder_output2 = dropOut28(decoder_output2, training=True)

    decoder_dense27 = Dense(2, activation='linear', name='Velcoity')
    velocityOut = decoder_dense27(decoder_output2)

    # Normalize output velocity to before concatenate
    # Prepeare the normalization layers
    minXVelConst = K.constant(value=minVelocityX, dtype='float32')
    minXMaxVelDiffConst = K.constant(value=(maxVelocityX-minVelocityX), dtype='float32')

    minYVelConst = K.constant(value=minVelocityY, dtype='float32')
    minYMaxVelDiffConst = K.constant(value=(maxVelocityY-minVelocityY), dtype='float32')

    velocityXNormalized = Lambda(lambda x: (x-minXVelConst)/minXMaxVelDiffConst)
    velocityYNormalized = Lambda(lambda x: (x-minYVelConst)/minYMaxVelDiffConst)

    # Prepeare the slice layers and separate the Vx and Vy
    velocityExtractX = Lambda(lambda x: tf.slice(x, (0, 0, 0), (-1, -1, 1)))
    velocityExtractY = Lambda(lambda x: tf.slice(x, (0, 0, 1), (-1, -1, 1)))

    velocityOutX = velocityExtractX(velocityOut)
    velocityOutY = velocityExtractY(velocityOut)

    # Use the normlize layers to normalize the output
    velocityConcatX = velocityXNormalized(velocityOutX)
    velocityConcatY = velocityYNormalized(velocityOutY)


    # Decoder for position out
    decoder3_concat = Concatenate()
    decoder_output3 = decoder3_concat([decoder_outputs,classOut,velocityConcatX,velocityConcatY])

    dropOut31 = Dropout(dropOutFrac)
    batchNorm31 = BatchNormalization()
    decoder_dense30b = Dense(2048)
    decoder_Leaky30b = ELU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense30b(decoder_output3)
    decoder_output3 = batchNorm31(decoder_output3)
    decoder_output3 = decoder_Leaky30b(decoder_output3)
    decoder_output3 = dropOut31(decoder_output3, training=True)

    dropOut32 = Dropout(dropOutFrac)
    batchNorm32 = BatchNormalization()
    decoder_dense30a = Dense(1024)
    decoder_Leaky30a = ELU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense30a(decoder_output3)
    decoder_output3 = batchNorm32(decoder_output3)
    decoder_output3 = decoder_Leaky30a(decoder_output3)
    decoder_output3 = dropOut32(decoder_output3, training=True)

    dropOut33 = Dropout(dropOutFrac)
    batchNorm33 = BatchNormalization()
    decoder_dense30 = Dense(512)
    decoder_Leaky30 = ELU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense30(decoder_output3)
    decoder_output3 = batchNorm33(decoder_output3)
    decoder_output3 = decoder_Leaky30(decoder_output3)
    decoder_output3 = dropOut33(decoder_output3, training=True)

    dropOut34 = Dropout(dropOutFrac)
    batchNorm34 = BatchNormalization()
    decoder_dense31 = Dense(256)
    decoder_Leaky31 = ELU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense31(decoder_output3)
    decoder_output3 = batchNorm34(decoder_output3)
    decoder_output3 = decoder_Leaky31(decoder_output3)
    decoder_output3 = dropOut34(decoder_output3, training=True)

    dropOut35 = Dropout(dropOutFrac)
    batchNorm35 = BatchNormalization()
    decoder_dense32 = Dense(128)
    decoder_Leaky32 = ELU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense32(decoder_output3)
    decoder_output3 = batchNorm35(decoder_output3)
    decoder_output3 = decoder_Leaky32(decoder_output3)
    decoder_output3 = dropOut35(decoder_output3, training=True)

    # dropOut36 = Dropout(dropOutFrac)
    # batchNorm36 = BatchNormalization()
    decoder_dense33 = Dense(64)
    decoder_Leaky33 = ELU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense33(decoder_output3)
    # decoder_output3 = batchNorm36(decoder_output3)
    decoder_output3 = decoder_Leaky33(decoder_output3)
    # decoder_output3 = dropOut36(decoder_output3, training=True)

    # dropOut37 = Dropout(dropOutFrac)
    # batchNorm37 = BatchNormalization()
    decoder_dense34 = Dense(32)
    decoder_Leaky34 = ELU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense34(decoder_output3)
    # decoder_output3 = batchNorm37(decoder_output3)
    decoder_output3 = decoder_Leaky34(decoder_output3)
    # decoder_output3 = dropOut37(decoder_output3, training=True)

    # dropOut38 = Dropout(dropOutFrac)
    # batchNorm38 = BatchNormalization()
    decoder_dense35 = Dense(16)
    decoder_Leaky35 = ELU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense35(decoder_output3)
    # decoder_output3 = batchNorm38(decoder_output3)
    decoder_output3 = decoder_Leaky35(decoder_output3)
    # decoder_output3 = dropOut38(decoder_output3, training=True)

    # dropOut39 = Dropout(dropOutFrac)
    # batchNorm39 = BatchNormalization()
    decoder_dense36 = Dense(8)
    decoder_Leaky36 = ELU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense36(decoder_output3)
    # decoder_output3 = batchNorm39(decoder_output3)
    decoder_output3 = decoder_Leaky36(decoder_output3)
    # decoder_output3 = dropOut39(decoder_output3, training=True)

    decoder_dense37 = Dense(2, activation='linear', name='Position')
    positionOut = decoder_dense37(decoder_output3)

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

    # # # Common batch norm
    # # decoder_outputs = commonBatchNorm(decoder_outputs)

    # Inference decoder for Class out
    decoder_output1 = decoder_dense10a(decoder_outputs)
    decoder_output1 = batchNorm11(decoder_output1)
    decoder_output1 = decoder_Leaky10a(decoder_output1)
    decoder_output1 = dropOut11(decoder_output1, training=True)

    decoder_output1 = decoder_dense10(decoder_output1)
    decoder_output1 = batchNorm12(decoder_output1)
    decoder_output1 = decoder_Leaky10(decoder_output1)
    decoder_output1 = dropOut12(decoder_output1, training=True)

    decoder_output1 = decoder_dense11(decoder_output1)
    decoder_output1 = batchNorm13(decoder_output1)
    decoder_output1 = decoder_Leaky11(decoder_output1)
    decoder_output1 = dropOut13(decoder_output1, training=True)

    decoder_output1 = decoder_dense12(decoder_output1)
    decoder_output1 = batchNorm14(decoder_output1)
    decoder_output1 = decoder_Leaky12(decoder_output1)
    decoder_output1 = dropOut14(decoder_output1, training=True)

    decoder_output1 = decoder_dense13(decoder_output1)
    decoder_output1 = batchNorm15(decoder_output1)
    decoder_output1 = decoder_Leaky13(decoder_output1)
    decoder_output1 = dropOut15(decoder_output1, training=True)

    decoder_output1 = decoder_dense14(decoder_output1)
    decoder_output1 = batchNorm16(decoder_output1)
    decoder_output1 = decoder_Leaky14(decoder_output1)
    decoder_output1 = dropOut16(decoder_output1, training=True)

    decoder_output1 = decoder_dense15(decoder_output1)
    # decoder_output1 = batchNorm17(decoder_output1)
    decoder_output1 = decoder_Leaky15(decoder_output1)
    # decoder_output1 = dropOut17(decoder_output1, training=True)

    decoder_output1 = decoder_dense16(decoder_output1)
    # decoder_output1 = batchNorm18(decoder_output1)
    decoder_output1 = decoder_Leaky16(decoder_output1)
    # decoder_output1 = dropOut18(decoder_output1, training=True)

    classOut = decoder_dense17(decoder_output1)

    # Inference Decoder for Velocity Out
    decoder_output2 = decoder2_concat([decoder_outputs,classOut])
    decoder_output2 = decoder_dense20a(decoder_output2)
    decoder_output2 = batchNorm21(decoder_output2)
    decoder_output2 = decoder_Leaky20a(decoder_output2)
    decoder_output2 = dropOut21(decoder_output2, training=True)

    decoder_output2 = decoder_dense20(decoder_output2)
    decoder_output2 = batchNorm22(decoder_output2)
    decoder_output2 = decoder_Leaky20(decoder_output2)
    decoder_output2 = dropOut22(decoder_output2, training=True)

    decoder_output2 = decoder_dense21(decoder_output2)
    decoder_output2 = batchNorm23(decoder_output2)
    decoder_output2 = decoder_Leaky21(decoder_output2)
    decoder_output2 = dropOut23(decoder_output2)

    decoder_output2 = decoder_dense22(decoder_output2)
    decoder_output2 = batchNorm24(decoder_output2)
    decoder_output2 = decoder_Leaky22(decoder_output2)
    decoder_output2 = dropOut24(decoder_output2, training=True)

    decoder_output2 = decoder_dense23(decoder_output2)
    decoder_output2 = batchNorm25(decoder_output2)
    decoder_output2 = decoder_Leaky23(decoder_output2)
    decoder_output2 = dropOut25(decoder_output2, training=True)

    decoder_output2 = decoder_dense24(decoder_output2)
    decoder_output2 = batchNorm26(decoder_output2)
    decoder_output2 = decoder_Leaky24(decoder_output2)
    decoder_output2 = dropOut26(decoder_output2, training=True)

    decoder_output2 = decoder_dense25(decoder_output2)
    # decoder_output2 = batchNorm27(decoder_output2)
    decoder_output2 = decoder_Leaky25(decoder_output2)
    # decoder_output2 = dropOut27(decoder_output2, training=True)

    decoder_output2 = decoder_dense26(decoder_output2)
    # decoder_output2 = batchNorm28(decoder_output2)
    decoder_output2 = decoder_Leaky26(decoder_output2)
    # decoder_output2 = dropOut28(decoder_output2, training=True)

    velocityOut = decoder_dense27(decoder_output2)

    # Inference Decoder Velocity Normalizer
    velocityOutX = velocityExtractX(velocityOut)
    velocityOutY = velocityExtractY(velocityOut)

    velocityConcatX = velocityXNormalized(velocityOutX)
    velocityConcatY = velocityYNormalized(velocityOutY)

    #Inference  Decoder for position out
    decoder_output3 = decoder3_concat([decoder_outputs,classOut,velocityConcatX,velocityConcatY])
    decoder_output3 = decoder_dense30b(decoder_output3)
    decoder_output3 = batchNorm31(decoder_output3)
    decoder_output3 = decoder_Leaky30b(decoder_output3)
    decoder_output3 = dropOut31(decoder_output3, training=True)

    decoder_output3 = decoder_dense30a(decoder_output3)
    decoder_output3 = batchNorm32(decoder_output3)
    decoder_output3 = decoder_Leaky30a(decoder_output3)
    decoder_output3 = dropOut32(decoder_output3, training=True)

    decoder_output3 = decoder_dense30(decoder_output3)
    decoder_output3 = batchNorm33(decoder_output3)
    decoder_output3 = decoder_Leaky30(decoder_output3)
    decoder_output3 = dropOut33(decoder_output3, training=True)

    decoder_output3 = decoder_dense31(decoder_output3)
    decoder_output3 = batchNorm34(decoder_output3)
    decoder_output3 = decoder_Leaky31(decoder_output3)
    decoder_output3 = dropOut34(decoder_output3, training=True)

    decoder_output3 = decoder_dense32(decoder_output3)
    decoder_output3 = batchNorm35(decoder_output3)
    decoder_output3 = decoder_Leaky32(decoder_output3)
    decoder_output3 = dropOut35(decoder_output3, training=True)

    decoder_output3 = decoder_dense33(decoder_output3)
    # decoder_output3 = batchNorm36(decoder_output3)
    decoder_output3 = decoder_Leaky33(decoder_output3)
    # decoder_output3 = dropOut36(decoder_output3, training=True)

    decoder_output3 = decoder_dense34(decoder_output3)
    # decoder_output3 = batchNorm37(decoder_output3)
    decoder_output3 = decoder_Leaky34(decoder_output3)
    # decoder_output3 = dropOut37(decoder_output3, training=True)

    decoder_output3 = decoder_dense35(decoder_output3)
    # decoder_output3 = batchNorm38(decoder_output3)
    decoder_output3 = decoder_Leaky35(decoder_output3)
    # decoder_output3 = dropOut38(decoder_output3, training=True)

    decoder_output3 = decoder_dense36(decoder_output3)
    # decoder_output3 = batchNorm39(decoder_output3)
    decoder_output3 = decoder_Leaky36(decoder_output3)
    # decoder_output3 = dropOut39(decoder_output3, training=True)

    positionOut = decoder_dense37(decoder_output3)

    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [classOut, velocityOut, positionOut] + decoder_states)

    opt = Nadam()   # Change to adam

    print('Before compile!!!!')

    # model.compile(optimizer=opt, loss=['categorical_crossentropy', logcosh, euclidean_distance_loss])
    model.compile(optimizer=opt, loss=['categorical_crossentropy', logcosh, euclidean_distance_loss])
    # model.summary()

    # Return the encoder and decoder model
    return model,encoder_model,decoder_model


# Create the model architecture 
# Model for used for the NGSIM data
# # # def ModelArch():

# # #     # Saniy check Min max values should be same as -9999 or 99999 after update
# # #     if((minVelocityX == 999) or (maxVelocityX == -999) or (minVelocityY == 999) or (maxVelocityY == -999) or (maxRealtiveX == -9999) or (maxRealtiveY == -9999) or (minRealtiveX == 9999) or (minRealtiveY == 9999)):
# # #         print('Min max values are not porperly updated in the UpdateMinMax function!!!')
# # #         sys.exit()

# # #     import tensorflow as tf

# # #     from tensorflow.keras import backend as K
# # #     from tensorflow.keras.models import Model
# # #     from tensorflow.keras.layers import Input,LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dropout, MaxPooling1D, Concatenate, subtract, Lambda, BatchNormalization, LeakyReLU, ELU, Add, Reshape, RepeatVector
# # #     from tensorflow.keras import optimizers
# # #     from tensorflow.keras.optimizers import RMSprop, Adam, Nadam
# # #     from tensorflow.keras.losses import logcosh

# # #     # define training encoder
# # #     encoder_inputs = Input(shape=(None, globalInputFeatures))
# # #     # First Encoder LSTM Layer
# # #     encoder1 = LSTM(n_units, return_state=True, return_sequences=True)
# # #     encoder_output, state_h1, state_c1 = encoder1(encoder_inputs)
# # #     encoder_states1 = [state_h1, state_c1]
# # #     # Second Encoder LSTM Layer
# # #     encoder2 = LSTM(n_units, return_state=True)
# # #     encoder_output, state_h2, state_c2 = encoder2(encoder_output)
# # #     encoder_states2 = [state_h2, state_c2]
# # # 	# define training decoder
# # #     decoder_inputs = Input(shape=(None, globalDecoderFeatures)
# # #     # First Decoder LSTM Layer
# # #     decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True)
# # #     decoder_outputs, _, _ = decoder_lstm1(decoder_inputs, initial_state=encoder_states1)
# # #     # Second Decoder LSTM Layer
# # #     decoder_lstm2 = LSTM(n_units, return_sequences=True, return_state=True)
# # #     decoder_outputs, _, _ = decoder_lstm2(decoder_outputs, initial_state=encoder_states2)

# # #     encoder_states = [state_h1, state_c1, state_h2, state_c2]

# # #     # Common BatchNorm
# # #     # # commonBatchNorm = BatchNormalization()
# # #     # # decoder_outputs = commonBatchNorm(decoder_outputs)

# # #     # Decoder for ClassOut        
# # #     batchNorm1 = BatchNormalization()
# # #     decoder_dense10a = Dense(1024)
# # #     decoder_Leaky10a = ELU(alpha=leakyAlphaValue)
# # #     decoder_output1 = decoder_dense10a(decoder_outputs)
# # #     decoder_output1 = batchNorm1(decoder_output1)
# # #     decoder_output1 = decoder_Leaky10a(decoder_output1)
# # #     batchNorm2 = BatchNormalization()
# # #     decoder_dense10 = Dense(512)
# # #     decoder_Leaky10 = ELU(alpha=leakyAlphaValue)
# # #     decoder_output1 = decoder_dense10(decoder_output1)
# # #     decoder_output1 = batchNorm2(decoder_output1)
# # #     decoder_output1 = decoder_Leaky10(decoder_output1)
# # #     batchNorm3 = BatchNormalization()
# # #     decoder_dense11 = Dense(256)
# # #     decoder_Leaky11 = ELU(alpha=leakyAlphaValue)
# # #     decoder_output1 = decoder_dense11(decoder_output1)
# # #     decoder_output1 = batchNorm3(decoder_output1)
# # #     decoder_output1 = decoder_Leaky11(decoder_output1)
# # #     batchNorm4 = BatchNormalization()    
# # #     decoder_dense12 = Dense(128)
# # #     decoder_Leaky12 = ELU(alpha=leakyAlphaValue)
# # #     decoder_output1 = decoder_dense12(decoder_output1)
# # #     decoder_output1 = batchNorm4(decoder_output1)
# # #     decoder_output1 = decoder_Leaky12(decoder_output1)
# # #     # batchNorm5 = BatchNormalization()
# # #     decoder_dense13 = Dense(64)
# # #     decoder_Leaky13 = ELU(alpha=leakyAlphaValue)
# # #     decoder_output1 = decoder_dense13(decoder_output1)
# # #     # decoder_output1 = batchNorm5(decoder_output1)
# # #     decoder_output1 = decoder_Leaky13(decoder_output1)
# # #     # batchNorm6 = BatchNormalization()    
# # #     decoder_dense14 = Dense(32)
# # #     decoder_Leaky14 = ELU(alpha=leakyAlphaValue)
# # #     decoder_output1 = decoder_dense14(decoder_output1)
# # #     # decoder_output1 = batchNorm6(decoder_output1)
# # #     decoder_output1 = decoder_Leaky14(decoder_output1)
# # #     decoder_dense15 = Dense(3, activation='softmax', name='Class')
# # #     classOut = decoder_dense15(decoder_output1)

# # #     # Decoder for Velocity Out
# # #     decoder2_concat = Concatenate()
# # #     decoder_output2 = decoder2_concat([decoder_outputs,classOut])
# # #     batchNorm7 = BatchNormalization()    
# # #     decoder_dense20a = Dense(1024)
# # #     decoder_Leaky20a = ELU(alpha=leakyAlphaValue)
# # #     decoder_output2 = decoder_dense20a(decoder_output2)
# # #     decoder_output2 = batchNorm7(decoder_output2)
# # #     decoder_output2 = decoder_Leaky20a(decoder_output2)
# # #     batchNorm8 = BatchNormalization()    
# # #     decoder_dense20 = Dense(512)
# # #     decoder_Leaky20 = ELU(alpha=leakyAlphaValue)
# # #     decoder_output2 = decoder_dense20(decoder_output2)
# # #     decoder_output2 = batchNorm8(decoder_output2)
# # #     decoder_output2 = decoder_Leaky20(decoder_output2)
# # #     batchNorm9 = BatchNormalization()
# # #     decoder_dense21 = Dense(256)
# # #     decoder_Leaky21 = ELU(alpha=leakyAlphaValue)
# # #     decoder_output2 = decoder_dense21(decoder_output2)
# # #     decoder_output2 = batchNorm9(decoder_output2)
# # #     decoder_output2 = decoder_Leaky21(decoder_output2)
# # #     batchNorm10 = BatchNormalization()    
# # #     decoder_dense22 = Dense(128)
# # #     decoder_Leaky22 = ELU(alpha=leakyAlphaValue)
# # #     decoder_output2 = decoder_dense22(decoder_output2)
# # #     decoder_output2 = batchNorm10(decoder_output2)
# # #     decoder_output2 = decoder_Leaky22(decoder_output2)
# # #     # batchNorm11 = BatchNormalization()
# # #     decoder_dense23 = Dense(64)
# # #     decoder_Leaky23 = ELU(alpha=leakyAlphaValue)
# # #     decoder_output2 = decoder_dense23(decoder_output2)
# # #     # decoder_output2 = batchNorm11(decoder_output2)
# # #     decoder_output2 = decoder_Leaky23(decoder_output2)
# # #     # batchNorm12 = BatchNormalization()    
# # #     decoder_dense24 = Dense(32, activation='linear')
# # #     # decoder_Leaky24 = ELU(alpha=leakyAlphaValue)
# # #     decoder_output2 = decoder_dense24(decoder_output2)
# # #     # decoder_output2 = batchNorm12(decoder_output2)
# # #     # decoder_output2 = decoder_Leaky24(decoder_output2)
# # #     decoder_dense25 = Dense(2, activation='linear', name='Velcoity')
# # #     velocityOut = decoder_dense25(decoder_output2)


# # #     # Prepeare the normalization layers
# # #     minXVelConst = K.constant(value=minVelocityX, dtype='float32')
# # #     minXMaxVelDiffConst = K.constant(value=(maxVelocityX-minVelocityX), dtype='float32')

# # #     minYVelConst = K.constant(value=minVelocityY, dtype='float32')
# # #     minYMaxVelDiffConst = K.constant(value=(maxVelocityY-minVelocityY), dtype='float32')

# # #     velocityXNormalized = Lambda(lambda x: (x-minXVelConst)/minXMaxVelDiffConst)
# # #     velocityYNormalized = Lambda(lambda x: (x-minYVelConst)/minYMaxVelDiffConst)

# # #     # Prepeare the slice layers nad separate the Vx and Vy
# # #     velocityExtractX = Lambda(lambda x: tf.slice(x, (0, 0, 0), (-1, -1, 1)))
# # #     velocityExtractY = Lambda(lambda x: tf.slice(x, (0, 0, 1), (-1, -1, 1)))

# # #     velocityOutX = velocityExtractX(velocityOut)
# # #     velocityOutY = velocityExtractY(velocityOut)

# # #     # Use the normlize layers to normalize the output
# # #     velocityConcatX = velocityXNormalized(velocityOutX)
# # #     velocityConcatY = velocityYNormalized(velocityOutY)

# # #     # Decoder for position out
# # #     decoder3_concat = Concatenate()
# # #     decoder_output3 = decoder3_concat([decoder_outputs,classOut,velocityConcatX,velocityConcatY])
# # #     # batchNorm13 = BatchNormalization()
# # #     decoder_dense30b = Dense(2048)
# # #     decoder_Leaky30b = LeakyReLU(alpha=leakyAlphaValue)
# # #     decoder_output3 = decoder_dense30b(decoder_output3)
# # #     # decoder_output3 = batchNorm13(decoder_output3)
# # #     decoder_output3 = decoder_Leaky30b(decoder_output3)
# # #     # batchNorm14 = BatchNormalization()    
# # #     decoder_dense30a = Dense(1024)
# # #     decoder_Leaky30a = ELU(alpha=leakyAlphaValue)
# # #     decoder_output3 = decoder_dense30a(decoder_output3)
# # #     # decoder_output3 = batchNorm14(decoder_output3)
# # #     decoder_output3 = decoder_Leaky30a(decoder_output3)
# # #     # batchNorm15 = BatchNormalization()
# # #     decoder_dense30 = Dense(512)
# # #     decoder_Leaky30 = ELU(alpha=leakyAlphaValue)
# # #     decoder_output3 = decoder_dense30(decoder_output3)
# # #     # decoder_output3 = batchNorm15(decoder_output3)
# # #     decoder_output3 = decoder_Leaky30(decoder_output3)
# # #     # batchNorm16 = BatchNormalization()    
# # #     decoder_dense31 = Dense(256)
# # #     decoder_Leaky31 = ELU(alpha=leakyAlphaValue)
# # #     decoder_output3 = decoder_dense31(decoder_output3)
# # #     # decoder_output3 = batchNorm16(decoder_output3)
# # #     decoder_output3 = decoder_Leaky31(decoder_output3)
# # #     # batchNorm17 = BatchNormalization()    
# # #     decoder_dense32 = Dense(128)
# # #     decoder_Leaky32 = ELU(alpha=leakyAlphaValue)
# # #     decoder_output3 = decoder_dense32(decoder_output3)
# # #     # decoder_output3 = batchNorm17(decoder_output3)
# # #     decoder_output3 = decoder_Leaky32(decoder_output3)
# # #     # batchNorm18 = BatchNormalization()    
# # #     decoder_dense33 = Dense(64)
# # #     decoder_Leaky33 = ELU(alpha=leakyAlphaValue)
# # #     decoder_output3 = decoder_dense33(decoder_output3)
# # #     # decoder_output3 = batchNorm18(decoder_output3)
# # #     decoder_output3 = decoder_Leaky33(decoder_output3)
# # #     # batchNorm19 = BatchNormalization()    
# # #     decoder_dense34 = Dense(32, activation='linear')
# # #     # decoder_Leaky34 = ELU(alpha=leakyAlphaValue)
# # #     decoder_output3 = decoder_dense34(decoder_output3)
# # #     # decoder_output3 = batchNorm19(decoder_output3)
# # #     # decoder_output3 = decoder_Leaky34(decoder_output3)
# # #     decoder_dense35 = Dense(2, activation='linear', name='Position')
# # #     positionOut = decoder_dense35(decoder_output3)
    
# # #     model = Model([encoder_inputs, decoder_inputs], [classOut, velocityOut, positionOut])

# # # 	# define inference encoder
# # #     encoder_model = Model(encoder_inputs, encoder_states)

# # # 	# define inference decoder
# # #     decoder_state_input_h1 = Input(shape=(n_units,))
# # #     decoder_state_input_c1 = Input(shape=(n_units,))
# # #     decoder_state_input_h2 = Input(shape=(n_units,))
# # #     decoder_state_input_c2 = Input(shape=(n_units,))
# # #     decoder_states_inputs1 = [decoder_state_input_h1, decoder_state_input_c1]
# # #     decoder_states_inputs2 = [decoder_state_input_h2, decoder_state_input_c2]
# # #     decoder_states_inputs = [decoder_state_input_h1, decoder_state_input_c1,decoder_state_input_h2, decoder_state_input_c2]
# # #     decoder_outputs, state_h1, state_c1 = decoder_lstm1(decoder_inputs, initial_state=decoder_states_inputs1)
# # #     decoder_outputs, state_h2, state_c2 = decoder_lstm2(decoder_outputs, initial_state=decoder_states_inputs2)
# # #     decoder_states = [state_h1, state_c1, state_h2, state_c2]

# # #     # Common BatchNorm
# # #     # # decoder_outputs = commonBatchNorm(decoder_outputs)

# # #     # Inference decoder for Class out
# # #     decoder_output1 = decoder_dense10a(decoder_outputs)
# # #     decoder_output1 = batchNorm1(decoder_output1)
# # #     decoder_output1 = decoder_Leaky10a(decoder_output1)
# # #     decoder_output1 = decoder_dense10(decoder_output1)
# # #     decoder_output1 = batchNorm2(decoder_output1)
# # #     decoder_output1 = decoder_Leaky10(decoder_output1)
# # #     decoder_output1 = decoder_dense11(decoder_output1)
# # #     decoder_output1 = batchNorm3(decoder_output1)
# # #     decoder_output1 = decoder_Leaky11(decoder_output1)
# # #     decoder_output1 = decoder_dense12(decoder_output1)
# # #     decoder_output1 = batchNorm4(decoder_output1)
# # #     decoder_output1 = decoder_Leaky12(decoder_output1)
# # #     decoder_output1 = decoder_dense13(decoder_output1)
# # #     # decoder_output1 = batchNorm5(decoder_output1)
# # #     decoder_output1 = decoder_Leaky13(decoder_output1)
# # #     decoder_output1 = decoder_dense14(decoder_output1)
# # #     # decoder_output1 = batchNorm6(decoder_output1)
# # #     decoder_output1 = decoder_Leaky14(decoder_output1)
# # #     classOut = decoder_dense15(decoder_output1)

# # #     # Inference Decoder for Velocity Out
# # #     decoder_output2 = decoder2_concat([decoder_outputs,classOut])
# # #     decoder_output2 = decoder_dense20a(decoder_output2)
# # #     decoder_output2 = batchNorm7(decoder_output2)
# # #     decoder_output2 = decoder_Leaky20a(decoder_output2)
# # #     decoder_output2 = decoder_dense20(decoder_output2)
# # #     decoder_output2 = batchNorm8(decoder_output2)
# # #     decoder_output2 = decoder_Leaky20(decoder_output2)
# # #     decoder_output2 = decoder_dense21(decoder_output2)
# # #     decoder_output2 = batchNorm9(decoder_output2)
# # #     decoder_output2 = decoder_Leaky21(decoder_output2)
# # #     decoder_output2 = decoder_dense22(decoder_output2)
# # #     decoder_output2 = batchNorm10(decoder_output2)
# # #     decoder_output2 = decoder_Leaky22(decoder_output2)
# # #     decoder_output2 = decoder_dense23(decoder_output2)
# # #     # decoder_output2 = batchNorm11(decoder_output2)
# # #     decoder_output2 = decoder_Leaky23(decoder_output2)
# # #     decoder_output2 = decoder_dense24(decoder_output2)
# # #     # decoder_output2 = batchNorm12(decoder_output2)
# # #     # decoder_output2 = decoder_Leaky24(decoder_output2)
# # #     velocityOut = decoder_dense25(decoder_output2)

# # #     # Inference Decoder Velocity Normalizer
# # #     velocityOutX = velocityExtractX(velocityOut)
# # #     velocityOutY = velocityExtractY(velocityOut)

# # #     velocityConcatX = velocityXNormalized(velocityOutX)
# # #     velocityConcatY = velocityYNormalized(velocityOutY)
    

# # #     #Inference  Decoder for position out
# # #     decoder_output3 = decoder3_concat([decoder_outputs,classOut,velocityConcatX,velocityConcatY])    
# # #     decoder_output3 = decoder_dense30b(decoder_output3)
# # #     # decoder_output3 = batchNorm13(decoder_output3)
# # #     decoder_output3 = decoder_Leaky30b(decoder_output3)    
# # #     decoder_output3 = decoder_dense30a(decoder_output3)
# # #     # decoder_output3 = batchNorm14(decoder_output3)
# # #     decoder_output3 = decoder_Leaky30a(decoder_output3)    
# # #     decoder_output3 = decoder_dense30(decoder_output3)
# # #     # decoder_output3 = batchNorm15(decoder_output3)
# # #     decoder_output3 = decoder_Leaky30(decoder_output3)
# # #     decoder_output3 = decoder_dense31(decoder_output3)
# # #     # decoder_output3 = batchNorm16(decoder_output3)
# # #     decoder_output3 = decoder_Leaky31(decoder_output3)    
# # #     decoder_output3 = decoder_dense32(decoder_output3)
# # #     # decoder_output3 = batchNorm17(decoder_output3)
# # #     decoder_output3 = decoder_Leaky32(decoder_output3)
# # #     decoder_output3 = decoder_dense33(decoder_output3)
# # #     # decoder_output3 = batchNorm18(decoder_output3)
# # #     decoder_output3 = decoder_Leaky33(decoder_output3)    
# # #     decoder_output3 = decoder_dense34(decoder_output3)
# # #     # decoder_output3 = batchNorm19(decoder_output3)
# # #     # decoder_output3 = decoder_Leaky34(decoder_output3)
# # #     positionOut = decoder_dense35(decoder_output3)

# # #     decoder_model = Model([decoder_inputs] + decoder_states_inputs, [classOut, velocityOut, positionOut] + decoder_states)

# # #     opt = Nadam()     #   RMSprop()

# # #     model.compile(optimizer=opt, loss=['categorical_crossentropy', logcosh, euclidean_distance_loss])

# # #     return model,encoder_model,decoder_model


# # Create the model architecture 
# # Motion based model to reduce the size and hence 2048 batchsize..
# def ModelArch():

#     # Saniy check Min max values should be same as -9999 or 99999 after update
#     if((minVelocityX == 999) or (maxVelocityX == -999) or (minVelocityY == 999) or (maxVelocityY == -999) or (maxRealtiveX == -9999) or (maxRealtiveY == -9999) or (minRealtiveX == 9999) or (minRealtiveY == 9999)):
#         print('Min max values are not porperly updated in the UpdateMinMax function!!!')
#         sys.exit()

#     import tensorflow as tf

#     from tensorflow.keras import backend as K
#     from tensorflow.keras.models import Model
#     from tensorflow.keras.layers import Input,LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dropout, MaxPooling1D, Concatenate, subtract, Lambda, BatchNormalization, LeakyReLU, ELU, Add, Reshape, RepeatVector
#     from tensorflow.keras import optimizers
#     from tensorflow.keras.optimizers import RMSprop, Adam, Nadam
#     from tensorflow.keras.losses import logcosh


#     # define training encoder
#     encoder_inputs = Input(shape=(None, globalInputFeatures))
#     # First Encoder LSTM Layer
#     encoder1 = LSTM(n_units, return_state=True, return_sequences=True)
#     encoder_output, state_h1, state_c1 = encoder1(encoder_inputs)
#     encoder_states1 = [state_h1, state_c1]
#     # Second Encoder LSTM Layer
#     encoder2 = LSTM(n_units, return_state=True)
#     encoder_output, state_h2, state_c2 = encoder2(encoder_output)
#     encoder_states2 = [state_h2, state_c2]
# 	# define training decoder
#     decoder_inputs = Input(shape=(None, globalDecoderFeatures))
#     # First Decoder LSTM Layer
#     decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True)
#     decoder_outputs, _, _ = decoder_lstm1(decoder_inputs, initial_state=encoder_states1)
#     # Second Decoder LSTM Layer
#     decoder_lstm2 = LSTM(n_units, return_sequences=True, return_state=True)
#     decoder_outputs, _, _ = decoder_lstm2(decoder_outputs, initial_state=encoder_states2)

#     encoder_states = [state_h1, state_c1, state_h2, state_c2]

#     # # # Common batch norm
#     # # commonBatchNorm = BatchNormalization()
#     # # decoder_outputs = commonBatchNorm(decoder_outputs)

#     # Decoder for ClassOut
#     dropOut11 = Dropout(dropOutFrac)
#     batchNorm11 = BatchNormalization()
#     decoder_dense10a = Dense(512)
#     decoder_Leaky10a = ELU(alpha=leakyAlphaValue)
#     decoder_output1 = decoder_dense10a(decoder_outputs)
#     decoder_output1 = batchNorm11(decoder_output1)
#     decoder_output1 = decoder_Leaky10a(decoder_output1)
#     decoder_output1 = dropOut11(decoder_output1, training=True)

#     dropOut12 = Dropout(dropOutFrac)
#     batchNorm12 = BatchNormalization()
#     decoder_dense10 = Dense(256)
#     decoder_Leaky10 = ELU(alpha=leakyAlphaValue)
#     decoder_output1 = decoder_dense10(decoder_output1)
#     decoder_output1 = batchNorm12(decoder_output1)
#     decoder_output1 = decoder_Leaky10(decoder_output1)
#     decoder_output1 = dropOut12(decoder_output1, training=True)

#     dropOut13 = Dropout(dropOutFrac)
#     batchNorm13 = BatchNormalization()
#     decoder_dense11 = Dense(128)
#     decoder_Leaky11 = ELU(alpha=leakyAlphaValue)
#     decoder_output1 = decoder_dense11(decoder_output1)
#     decoder_output1 = batchNorm13(decoder_output1)
#     decoder_output1 = decoder_Leaky11(decoder_output1)
#     decoder_output1 = dropOut13(decoder_output1, training=True)

#     dropOut14 = Dropout(dropOutFrac)
#     batchNorm14 = BatchNormalization()
#     decoder_dense12 = Dense(64)
#     decoder_Leaky12 = ELU(alpha=leakyAlphaValue)
#     decoder_output1 = decoder_dense12(decoder_output1)
#     decoder_output1 = batchNorm14(decoder_output1)
#     decoder_output1 = decoder_Leaky12(decoder_output1)
#     decoder_output1 = dropOut14(decoder_output1, training=True)

#     dropOut15 = Dropout(dropOutFrac)
#     batchNorm15 = BatchNormalization()
#     decoder_dense13 = Dense(32)
#     decoder_Leaky13 = ELU(alpha=leakyAlphaValue)
#     decoder_output1 = decoder_dense13(decoder_output1)
#     decoder_output1 = batchNorm15(decoder_output1)
#     decoder_output1 = decoder_Leaky13(decoder_output1)
#     decoder_output1 = dropOut15(decoder_output1, training=True)

#     # dropOut16 = Dropout(dropOutFrac)
#     # batchNorm16 = BatchNormalization()
#     decoder_dense14 = Dense(16)
#     decoder_Leaky14 = ELU(alpha=leakyAlphaValue)
#     decoder_output1 = decoder_dense14(decoder_output1)
#     # decoder_output1 = batchNorm16(decoder_output1)
#     decoder_output1 = decoder_Leaky14(decoder_output1)
#     # decoder_output1 = dropOut16(decoder_output1, training=True)

#     # dropOut17 = Dropout(dropOutFrac)
#     # batchNorm17 = BatchNormalization()
#     decoder_dense15 = Dense(8)
#     decoder_Leaky15 = ELU(alpha=leakyAlphaValue)
#     decoder_output1 = decoder_dense15(decoder_output1)
#     # decoder_output1 = batchNorm17(decoder_output1)
#     decoder_output1 = decoder_Leaky15(decoder_output1)
#     # decoder_output1 = dropOut17(decoder_output1, training=True)

#     # dropOut18 = Dropout(dropOutFrac)
#     # batchNorm18 = BatchNormalization()
#     decoder_dense16 = Dense(4)
#     decoder_Leaky16 = ELU(alpha=leakyAlphaValue)
#     decoder_output1 = decoder_dense16(decoder_output1)
#     # decoder_output1 = batchNorm18(decoder_output1)
#     decoder_output1 = decoder_Leaky16(decoder_output1)
#     # decoder_output1 = dropOut18(decoder_output1, training=True)

#     decoder_dense17 = Dense(3, activation='softmax', name='Class')
#     classOut = decoder_dense17(decoder_output1)

#     # Decoder for Velocity Out
#     decoder2_concat = Concatenate()
#     decoder_output2 = decoder2_concat([decoder_outputs,classOut])

#     dropOut21 = Dropout(dropOutFrac)
#     batchNorm21 = BatchNormalization()
#     decoder_dense20a = Dense(2048)
#     decoder_Leaky20a = ELU(alpha=leakyAlphaValue)
#     decoder_output2 = decoder_dense20a(decoder_output2)
#     decoder_output2 = batchNorm21(decoder_output2)
#     decoder_output2 = decoder_Leaky20a(decoder_output2)
#     decoder_output2 = dropOut21(decoder_output2, training=True)

#     dropOut22 = Dropout(dropOutFrac)
#     batchNorm22 = BatchNormalization()
#     decoder_dense20 = Dense(1024)
#     decoder_Leaky20 = ELU(alpha=leakyAlphaValue)
#     decoder_output2 = decoder_dense20(decoder_output2)
#     decoder_output2 = batchNorm22(decoder_output2)
#     decoder_output2 = decoder_Leaky20(decoder_output2)
#     decoder_output2 = dropOut22(decoder_output2, training=True)

#     dropOut23 = Dropout(dropOutFrac)
#     batchNorm23 = BatchNormalization()
#     decoder_dense21 = Dense(512)
#     decoder_Leaky21 = ELU(alpha=leakyAlphaValue)
#     decoder_output2 = decoder_dense21(decoder_output2)
#     decoder_output2 = batchNorm23(decoder_output2)
#     decoder_output2 = decoder_Leaky21(decoder_output2)
#     decoder_output2 = dropOut23(decoder_output2, training=True)

#     dropOut24 = Dropout(dropOutFrac)
#     batchNorm24 = BatchNormalization()
#     decoder_dense22 = Dense(256)
#     decoder_Leaky22 = ELU(alpha=leakyAlphaValue)
#     decoder_output2 = decoder_dense22(decoder_output2)
#     decoder_output2 = batchNorm24(decoder_output2)
#     decoder_output2 = decoder_Leaky22(decoder_output2)
#     decoder_output2 = dropOut24(decoder_output2, training=True)

#     dropOut25 = Dropout(dropOutFrac)
#     batchNorm25 = BatchNormalization()
#     decoder_dense23 = Dense(128)
#     decoder_Leaky23 = ELU(alpha=leakyAlphaValue)
#     decoder_output2 = decoder_dense23(decoder_output2)
#     decoder_output2 = batchNorm25(decoder_output2)
#     decoder_output2 = decoder_Leaky23(decoder_output2)
#     decoder_output2 = dropOut25(decoder_output2, training=True)

#     # dropOut26 = Dropout(dropOutFrac)
#     # batchNorm26 = BatchNormalization()
#     decoder_dense24 = Dense(64)
#     decoder_Leaky24 = ELU(alpha=leakyAlphaValue)
#     decoder_output2 = decoder_dense24(decoder_output2)
#     # decoder_output2 = batchNorm26(decoder_output2)
#     decoder_output2 = decoder_Leaky24(decoder_output2)
#     # decoder_output2 = dropOut26(decoder_output2, training=True)

#     # dropOut27 = Dropout(dropOutFrac)
#     # batchNorm27 = BatchNormalization()
#     decoder_dense25 = Dense(32)
#     decoder_Leaky25 = ELU(alpha=leakyAlphaValue)
#     decoder_output2 = decoder_dense25(decoder_output2)
#     # decoder_output2 = batchNorm27(decoder_output2)
#     decoder_output2 = decoder_Leaky25(decoder_output2)
#     # decoder_output2 = dropOut27(decoder_output2, training=True)

#     # dropOut28 = Dropout(dropOutFrac)
#     # batchNorm28 = BatchNormalization()
#     decoder_dense26 = Dense(16)
#     decoder_Leaky26 = ELU(alpha=leakyAlphaValue)
#     decoder_output2 = decoder_dense26(decoder_output2)
#     # coder_output2 = batchNorm28(decoder_output2)
#     decoder_output2 = decoder_Leaky26(decoder_output2)
#     # decoder_output2 = dropOut28(decoder_output2, training=True)

#     # dropOut29 = Dropout(dropOutFrac)
#     # batchNorm29 = BatchNormalization()
#     decoder_dense27 = Dense(8)
#     decoder_Leaky27 = ELU(alpha=leakyAlphaValue)
#     decoder_output2 = decoder_dense27(decoder_output2)
#     # coder_output2 = batchNorm29(decoder_output2)
#     decoder_output2 = decoder_Leaky27(decoder_output2)
#     # decoder_output2 = dropOut29(decoder_output2, training=True)

#     decoder_dense28 = Dense(2, activation='linear', name='Velcoity')
#     velocityOut = decoder_dense28(decoder_output2)


#     # Decoder for position out, This time only extract the first two poses (X,Y) from the decoder input as the decoder input holds the last pose
#     # Simply add the predicted motion to the last pose to get the current pose

#     # Prepeare the slice layers and separate the Vx and Vy
#     velocityExtractX = Lambda(lambda b: tf.slice(b, (0, 0, 0), (-1, -1, 1)))
#     velocityExtractY = Lambda(lambda b: tf.slice(b, (0, 0, 1), (-1, -1, 1)))

#     velocityOutX = velocityExtractX(velocityOut)
#     velocityOutY = velocityExtractY(velocityOut)

#     # slice the decoder input same as velocity extract to get the first two items. slice(start,size)
#     # Prepeare the slice layers and separate the poseX and poseY
#     poseExtractX = Lambda(lambda c: tf.slice(c, (0, 0, 0), (-1, -1, 1)))
#     poseExtractY = Lambda(lambda c: tf.slice(c, (0, 0, 1), (-1, -1, 1)))

#     decoderPoseOutX = poseExtractX(decoder_inputs)
#     decoderPoseOutY = poseExtractY(decoder_inputs)

#     # As the decoder_input has the normalized pose we need to unnormalize it first to get the real pose
#     # Prepeare the unnormalization layers
#     minXPoseConst = K.constant(value=minRealtiveX, dtype='float32')
#     minXMaxPoseDiffConst = K.constant(value=(maxRealtiveX-minRealtiveX), dtype='float32')

#     minYPoseConst = K.constant(value=minRealtiveY, dtype='float32')
#     minYMaxPoseDiffConst = K.constant(value=(maxRealtiveY-minRealtiveY), dtype='float32')

#     poseXUnNormalizedLayer = Lambda(lambda d: (d*minXMaxPoseDiffConst) + minXPoseConst)
#     poseYUnNormalizedLayer = Lambda(lambda d: (d*minYMaxPoseDiffConst) + minYPoseConst)

#     # Add the velocity output (we need NOT normalized as this will be directly added) to the not normalized pose
#     unNormalizedPoseX = poseXUnNormalizedLayer(decoderPoseOutX)
#     addPoseXLayer = Add()
#     poseXOut = addPoseXLayer([unNormalizedPoseX,velocityOutX])

#     unNormalizedPoseY = poseYUnNormalizedLayer(decoderPoseOutY)
#     addPoseYLayer = Add()
#     poseYOut = addPoseYLayer([unNormalizedPoseY,velocityOutY])

#     # Concat poseX and poseY to formate the final positionOut (X,Y)
#     decoder3_concat = Concatenate(name='position')
#     positionOut = decoder3_concat([poseXOut,poseYOut])


#     model = Model([encoder_inputs, decoder_inputs], [classOut, velocityOut, positionOut])

# 	# define inference encoder
#     encoder_model = Model(encoder_inputs, encoder_states)

# 	# define inference decoder
#     decoder_state_input_h1 = Input(shape=(n_units,))
#     decoder_state_input_c1 = Input(shape=(n_units,))
#     decoder_state_input_h2 = Input(shape=(n_units,))
#     decoder_state_input_c2 = Input(shape=(n_units,))
#     decoder_states_inputs1 = [decoder_state_input_h1, decoder_state_input_c1]
#     decoder_states_inputs2 = [decoder_state_input_h2, decoder_state_input_c2]
#     decoder_states_inputs = [decoder_state_input_h1, decoder_state_input_c1,decoder_state_input_h2, decoder_state_input_c2]
#     decoder_outputs, state_h1, state_c1 = decoder_lstm1(decoder_inputs, initial_state=decoder_states_inputs1)
#     decoder_outputs, state_h2, state_c2 = decoder_lstm2(decoder_outputs, initial_state=decoder_states_inputs2)
#     decoder_states = [state_h1, state_c1, state_h2, state_c2]

#     # # # Common batch norm
#     # # decoder_outputs = commonBatchNorm(decoder_outputs)

#     # Inference decoder for Class out
#     decoder_output1 = decoder_dense10a(decoder_outputs)
#     decoder_output1 = batchNorm11(decoder_output1)
#     decoder_output1 = decoder_Leaky10a(decoder_output1)
#     decoder_output1 = dropOut11(decoder_output1, training=True)

#     decoder_output1 = decoder_dense10(decoder_output1)
#     decoder_output1 = batchNorm12(decoder_output1)
#     decoder_output1 = decoder_Leaky10(decoder_output1)
#     decoder_output1 = dropOut12(decoder_output1, training=True)

#     decoder_output1 = decoder_dense11(decoder_output1)
#     decoder_output1 = batchNorm13(decoder_output1)
#     decoder_output1 = decoder_Leaky11(decoder_output1)
#     decoder_output1 = dropOut13(decoder_output1, training=True)

#     decoder_output1 = decoder_dense12(decoder_output1)
#     decoder_output1 = batchNorm14(decoder_output1)
#     decoder_output1 = decoder_Leaky12(decoder_output1)
#     decoder_output1 = dropOut14(decoder_output1, training=True)

#     decoder_output1 = decoder_dense13(decoder_output1)
#     decoder_output1 = batchNorm15(decoder_output1)
#     decoder_output1 = decoder_Leaky13(decoder_output1)
#     decoder_output1 = dropOut15(decoder_output1, training=True)

#     decoder_output1 = decoder_dense14(decoder_output1)
#     # decoder_output1 = batchNorm16(decoder_output1)
#     decoder_output1 = decoder_Leaky14(decoder_output1)
#     # decoder_output1 = dropOut16(decoder_output1, training=True)

#     decoder_output1 = decoder_dense15(decoder_output1)
#     # decoder_output1 = batchNorm17(decoder_output1)
#     decoder_output1 = decoder_Leaky15(decoder_output1)
#     # decoder_output1 = dropOut17(decoder_output1, training=True)

#     decoder_output1 = decoder_dense16(decoder_output1)
#     # decoder_output1 = batchNorm18(decoder_output1)
#     decoder_output1 = decoder_Leaky16(decoder_output1)
#     # decoder_output1 = dropOut18(decoder_output1, training=True)

#     classOut = decoder_dense17(decoder_output1)

#     # Inference Decoder for Velocity Out
#     decoder_output2 = decoder2_concat([decoder_outputs,classOut])
#     decoder_output2 = decoder_dense20a(decoder_output2)
#     decoder_output2 = batchNorm21(decoder_output2)
#     decoder_output2 = decoder_Leaky20a(decoder_output2)
#     decoder_output2 = dropOut21(decoder_output2, training=True)

#     decoder_output2 = decoder_dense20(decoder_output2)
#     decoder_output2 = batchNorm22(decoder_output2)
#     decoder_output2 = decoder_Leaky20(decoder_output2)
#     decoder_output2 = dropOut22(decoder_output2, training=True)

#     decoder_output2 = decoder_dense21(decoder_output2)
#     decoder_output2 = batchNorm23(decoder_output2)
#     decoder_output2 = decoder_Leaky21(decoder_output2)
#     decoder_output2 = dropOut23(decoder_output2, training=True)

#     decoder_output2 = decoder_dense22(decoder_output2)
#     decoder_output2 = batchNorm24(decoder_output2)
#     decoder_output2 = decoder_Leaky22(decoder_output2)
#     decoder_output2 = dropOut24(decoder_output2, training=True)

#     decoder_output2 = decoder_dense23(decoder_output2)
#     decoder_output2 = batchNorm25(decoder_output2)
#     decoder_output2 = decoder_Leaky23(decoder_output2)
#     decoder_output2 = dropOut25(decoder_output2, training=True)

#     decoder_output2 = decoder_dense24(decoder_output2)
#     # decoder_output2 = batchNorm26(decoder_output2)
#     decoder_output2 = decoder_Leaky24(decoder_output2)
#     # decoder_output2 = dropOut26(decoder_output2, training=True)

#     decoder_output2 = decoder_dense25(decoder_output2)
#     # decoder_output2 = batchNorm27(decoder_output2)
#     decoder_output2 = decoder_Leaky25(decoder_output2)
#     # decoder_output2 = dropOut27(decoder_output2, training=True)

#     decoder_output2 = decoder_dense26(decoder_output2)
#     # decoder_output2 = batchNorm28(decoder_output2)
#     decoder_output2 = decoder_Leaky26(decoder_output2)
#     # decoder_output2 = dropOut28(decoder_output2, training=True)

#     decoder_output2 = decoder_dense27(decoder_output2)
#     # decoder_output2 = batchNorm29(decoder_output2)
#     decoder_output2 = decoder_Leaky27(decoder_output2)
#     # decoder_output2 = dropOut29(decoder_output2, training=True)

#     velocityOut = decoder_dense28(decoder_output2)

#     # Decoder for position out, This time only extract the first two poses (X,Y) from the decoder input as the decoder input holds the last pose
#     # Simply add the predicted motion to the last pose to get the current pose
#     velocityOutX = velocityExtractX(velocityOut)
#     velocityOutY = velocityExtractY(velocityOut)

#     # slice the decoder input same as velocity extract to get the first two items. slice(start,size)
#     # Prepeare the slice layers and separate the poseX and poseY
#     decoderPoseOutX = poseExtractX(decoder_inputs)
#     decoderPoseOutY = poseExtractY(decoder_inputs)

#     # Add the velocity output (we need NOT normalized as this will be directly added) to the not normalized pose
#     unNormalizedPoseX = poseXUnNormalizedLayer(decoderPoseOutX)
#     poseXOut = addPoseXLayer([unNormalizedPoseX,velocityOutX])

#     unNormalizedPoseY = poseYUnNormalizedLayer(decoderPoseOutY)
#     poseYOut = addPoseYLayer([unNormalizedPoseY,velocityOutY])

#     # Concat poseX and poseY to formate the final positionOut (X,Y)
#     positionOut = decoder3_concat([poseXOut,poseYOut])

#     decoder_model = Model([decoder_inputs] + decoder_states_inputs, [classOut, velocityOut, positionOut] + decoder_states)

#     opt = Nadam()   # Change to adam

#     print('Before compile!!!!')

#     # model.compile(optimizer=opt, loss=['categorical_crossentropy', logcosh, euclidean_distance_loss])
#     model.compile(optimizer=opt, loss=['categorical_crossentropy', logcosh, euclidean_distance_loss])
#     # model.summary()

#     # Return the encoder and decoder model
#     return model,encoder_model,decoder_model


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


# Class to hold all the relevet vehicle ID specific predicition intermediate information
class PredictionInfos():
    def __init__(self, input = [], decoderInput = [], state = [], output=[], groundTruth = [], initialPose = [], sectionIntersection = [], decoderInputList = []):
        self.input = input
        self.decoderInput = decoderInput
        self.state = state
        self.output = output
        self.groundTruth = groundTruth
        self.initialPose = initialPose
        self.sectionIntersection = sectionIntersection
        self.decoderInputList = decoderInputList


def IntermediatePredictionForTraining(processItemSubList,sublistAssignedGpu):

   # Saniy check Min max values should be same as -9999 or 99999 after update
    if((minVelocityX == 999) or (maxVelocityX == -999) or (minVelocityY == 999) or (maxVelocityY == -999) or (maxRealtiveX == -9999) or (maxRealtiveY == -9999) or (minRealtiveX == 9999) or (minRealtiveY == 9999)):
        print('Min max values are not porperly updated in the UpdateMinMax function!!!')
        sys.exit()

    randWait = random.randint(2,8)
    sleep(randWait)

    # Load the model to the specified GPU id
    import tensorflow as tf

    # This is for tensorflow version 2.2 or above
    # Extract the GPU list
    allGpus = tf.config.experimental.list_physical_devices('GPU')
    # Extract the current index, basically from '/gpu:0 to 0
    gpuIndex = int(sublistAssignedGpu.split(':')[-1])
    # Make the current GPU visible only as diff memory growth is not allowed
    tf.config.set_visible_devices(allGpus[gpuIndex], 'GPU')
    # Put the memry limit on GPU
    tf.config.experimental.set_memory_growth(allGpus[gpuIndex], True)

    # # # from tensorflow.keras import backend as K
    # # #K.set_learning_phase(0)  # For the batch normalization error (0 = test, 1 = train)
    with tf.device(sublistAssignedGpu):

        # This is for tensorflow version 2.1 or less
        # # config = tf.compat.v1.ConfigProto() 
        # # config.gpu_options.allow_growth = True
        # # session = tf.compat.v1.Session(config=config)

        # # # This is for tensorflow version 2.2 or above
        # # # Extract the GPU list
        # # allGpus = tf.config.experimental.list_physical_devices('GPU')
        # # # Extract the current index, basically from '/gpu:0 to 0
        # # gpuIndex = int(sublistAssignedGpu.split(':')[-1])
        # # # Put the memry limit on GPU
        # # tf.config.experimental.set_memory_growth(allGpus[gpuIndex], True)


        print('Waiting for GPU devices!!!')

        sleep(randWait)

        # Compile the model and load the weights
        model,encoder_model,decoder_model = ModelArch()

        encoder_model.load_weights(encoderModelFilename)
        print('Encoder loaded!!!')
        decoder_model.load_weights(decoderModelFilename)
        print('Decoder loaded!!!')
    
    for eachProcessItem in processItemSubList:
        # Retrive the Process item
        eachRelevenatVehicle = eachProcessItem[0]   # string
        currentTrainOrValStr = eachProcessItem[1]   # string

        currentReleventVehicleList = dictByVehicles[eachRelevenatVehicle]
        currentReleventVehicleLength = len(currentReleventVehicleList)


        # loop through the vehicle list to check if there is any single change of laneID. If not means zero lane changes measn ignore
        # Get the intitial lane first to compare
        changeFlag = False
        initialLaneID = currentReleventVehicleList[0][laneIDIndex]
        for eachTargetItem in currentReleventVehicleList:
            currentLaneID = eachTargetItem[laneIDIndex]
            if(currentLaneID != initialLaneID):
                changeFlag = True
                break

        
        # Check if any lane chnage happend, false means no lane chnage happend
        if(changeFlag == False):
            straightVehicles.append(0)
            if(len(straightVehicles) > includedStraightVehicles):
                continue


        # Get the current target vehicle ID
        targetUpdatedID = eachRelevenatVehicle


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
        for idx in range(historyTemporal+10+prevNextFrameCount,currentReleventVehicleLength-futureTemporal-10-prevNextFrameCount,60):   #  15

            # Prepare the trakcer Dict
            trackerDict = dict()
            trackerDict[targetUpdatedID] = []

            # for input
            for jdx in range(idx-historyTemporal,idx+futureTemporal):

                currentVechicleID = currentReleventVehicleList[jdx][vechileIDIndex]
                currentLocalX = currentReleventVehicleList[jdx][localXIndex]
                currentLocalY = currentReleventVehicleList[jdx][localYIndex]
                # # # currentVelocity = currentReleventVehicleList[jdx][velocityIndex]
                currentLaneID = currentReleventVehicleList[jdx][laneIDIndex]
                # # # currentDirection = currentReleventVehicleList[jdx][directionIndex]
                # # # currentMovement = currentReleventVehicleList[jdx][movementIndex]
                currentTime = currentReleventVehicleList[jdx][globalTimeIndex]
                currentFrame = currentReleventVehicleList[jdx][frameIDIndex]
                # # # currentSection = currentReleventVehicleList[jdx][sectionIndex]
                # # # currentIntersection = currentReleventVehicleList[jdx][intersectionIndex]
                # # currentHeadwaySpace = currentReleventVehicleList[jdx][headwaySpaceIndex]
                # # currentHeadwayTime = currentReleventVehicleList[jdx][headwayTimeIndex]

                # Get the last 3 and next 3 lane ids to estimate the lane change maneuver 
                lastThreeLaneIds = np.array(currentReleventVehicleList[jdx-prevNextFrameCount:jdx])[:,laneIDIndex]
                nextThreeLaneIds = np.array(currentReleventVehicleList[jdx:jdx+prevNextFrameCount])[:,laneIDIndex]
                totoalLaneIDs = list(lastThreeLaneIds)
                totoalLaneIDs.extend(list(nextThreeLaneIds))

                # Estmiate the lane change maneuver
                currentMovement = TargetLaneChanageManeuver(totoalLaneIDs)       #  currentVehicleList[jdx][movementIndex]

                # Estimate velocityX and VelocityY
                prevAbsoluteX = currentReleventVehicleList[jdx-1][localXIndex]
                prevAbsoluteY = currentReleventVehicleList[jdx-1][localYIndex]

                # For I80/US101 the vehicles are always moving forward
                currentVelocityX = currentLocalX-prevAbsoluteX        
                currentVelocityY = currentLocalY-prevAbsoluteY


                # Prepeare the target vehicle current input and append and at the end of the tracker dict list
                # tempInput = [localX,localY,velocityX,velocityY,laneID,movement]
                # Without headway time and space
                dictInput = [currentLocalX,currentLocalY,currentVelocityX,currentVelocityY,currentLaneID,currentMovement,currentTime,currentFrame]
                # With headway time and space
                # dictInput = [currentLocalX,currentLocalY,currentVelocityX,currentVelocityY,currentLaneID,currentMovement,currentHeadwaySpace,currentHeadwayTime,currentTime,currentFrame]
                trackerDict[targetUpdatedID].append(dictInput)


                # Get the surrounding cars
                otherVehicles = dictByFrames[str(currentTime)]

                # Get the previous timestamp (-100) other vehicles velcotiy calculations
                prevOtherVehicles = dictByFrames[str(currentTime-100)]
                # Get the next timestamp (+100) also in case the vehicle is not present in prev timestamp
                nextOtherVehicles = dictByFrames[str(currentTime+100)]

                # Target vehicle removal flag 
                targetRemovedFlag = 0

                for eachOtherVehicle in otherVehicles:
                    currentVechicleID = eachOtherVehicle[vechileIDIndex]
                    otherVehicleID = str(eachOtherVehicle[vechileIDIndex])

                    if(str(currentVechicleID) == targetOriginalID):
                        targetRemovedFlag = 1
                        continue

                    currentLocalX = eachOtherVehicle[localXIndex]
                    currentLocalY = eachOtherVehicle[localYIndex]
                    # # # currentVelocity = eachOtherVehicle[velocityIndex]
                    currentLaneID = eachOtherVehicle[laneIDIndex]
                    # # # #currentDirection = eachOtherVehicle[directionIndex]
                    # # #currentMovement = eachOtherVehicle[movementIndex]
                    currentTime = eachOtherVehicle[globalTimeIndex]
                    currentFrame = eachOtherVehicle[frameIDIndex]
                    # # currentSection = eachOtherVehicle[sectionIndex]
                    # # currentIntersection = eachOtherVehicle[intersectionIndex]
                    # Headway space and time for other vehicles
                    otherHeadwaySpace = eachOtherVehicle[headwaySpaceIndex]
                    otherHeadwayTime = eachOtherVehicle[headwayTimeIndex]

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
                            otherVelocityX = currentLocalX-prevOtherLocalPoseX          
                            otherVelocityY = currentLocalY-prevOtherLocalPoseY
                            # Find the lane ID of the same othher vehicle for previous frame
                            otherPrevLaneID = eachOtherItem[laneIDIndex]
                            # Estimate the lane change maneuver for the surroudning vehice
                            otherLaneIDList = [otherPrevLaneID,currentLaneID]
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
                                otherVelocityX = nextOtherLocalPoseX-currentLocalX       
                                otherVelocityY = nextOtherLocalPoseY-currentLocalY
                                # Find the lane ID of the same other vehicle for next frame
                                otherNextLaneID = eachOtherItem[laneIDIndex]
                                # Estimate the lane change maneuver for the surroudning vehice
                                otherLaneIDList = [currentLaneID,otherNextLaneID]
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

                    # tempInput = [localX,localY,velocityX,velocityY,laneID,movement]
                    # Without headway time and space
                    dictInput = [currentLocalX,currentLocalY,otherVelocityX,otherVelocityY,currentLaneID,currentMovement,currentTime,currentFrame]
                    # With headway time and space
                    # # dictInput = [currentLocalX,currentLocalY,otherVelocityX,otherVelocityY,currentLaneID,currentMovement,otherHeadwaySpace,otherHeadwayTime,currentTime,currentFrame]


                    # append the surrounding car info in the trakcer dict
                    # Check if the vehicle ID exist in mapper dict
                    # if yes use the updated key to avoid duplication
                    # Vehicle Birth in tracker Dict
                    if (str(currentVechicleID) not in trackerDict.keys()):
                        trackerDict[str(currentVechicleID)] = []
                        trackerDict[str(currentVechicleID)].append(dictInput)
                    else:
                        # Check the diff of last frame and last time with current frame and current time to avide duplicate vehicle IDs
                        lastTime = trackerDict[str(currentVechicleID)][-1][-2]  # -1 for last item and second last is time index in TrakcerDict list
                        lastFrame = trackerDict[str(currentVechicleID)][-1][-1] # -1 for last item and last is frame index in TrakcerDict list
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

            # print('Tracker populated!!! going for predition')

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
                predictionInfoObj = PredictionInfos([],[],[],[],[],[],[])
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
                    # # # targetSection = eachInputInfo[8]  # 8 is section index in trakcer dict list
                    # # # targetIntersection = eachInputInfo[9]  # 9 is intersection index in trakcer dict list

                    # Add check for time
                    targetTime = eachInputInfo[-2]  # second last is time index in trakcer dict list


                    tempPredictionInput = eachInputInfo.copy()[:-2] # Ignore the last two items (FrameID and Time) for the input
                    # convert the absolute position to normalized relative position
                    tempPredictionInput[0] = ((tempPredictionInput[0]-intitalX)-minRealtiveX)/(maxRealtiveX-minRealtiveX)  # 0 poseX
                    tempPredictionInput[1] = ((tempPredictionInput[1]-intitalY)-minRealtiveY)/(maxRealtiveY-minRealtiveY)   # 1 poseY

                    # Normalize the velcoityX and velocityY
                    tempPredictionInput[2] = (tempPredictionInput[2]-minVelocityX)/(maxVelocityX-minVelocityX)  # 2 velX
                    tempPredictionInput[3] = (tempPredictionInput[3]-minVelocityY)/(maxVelocityY-minVelocityY)

                    # # # # # All entries in the tempPredictionInput should be less than one. If not print the index and fail the execution
                    # # # # for tempIdx,eachTempPredVal in enumerate(tempPredictionInput):
                    # # # #     if(eachTempPredVal > 1.0):
                    # # # #         print('The ' + str(tempIdx) + ' value of the tempPredictionInput is more than 1.0!!!')
                    # # # #         print('The value is ' + str(eachTempPredVal) + ' !!!')
                    # # # #         print('The RAW is ' + str(tempPredictionInput[tempIdx]) + ' !!!')
                    # # # #         nanVal = 10000/0
                    # # # #         sys.exit()

                    # # # # Calculate Nearest junction distance and extend to the input temporary row list
                    # # # juncDist = CalculateNearestJuncLoc(targetSection, targetIntersection, targetLocalX, targetLocalY)
                    # # # tempPredictionInput.insert(len(tempPredictionInput),juncDist)

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
                            surroundingCarTime = trackerDict[eachSurroundingCarID][udx][-2]  # udx for coresponnding Frame and second last is time index in TrakcerDict list
                            if(surroundingCarTime!=targetTime):
                                print('Surrounding vehicle time mismatch during prediction!!!')
                                print('Surrounding vehicle time : ' + str(surroundingCarTime))
                                print('Target vehicle time : ' + str(targetTime))
                                sys.exit()

                            # tempInput = [localX,localY,velocityX,velocityY,laneID,movement]
                            # Extract the absolute pose and convert to normalized relative pose
                            surroundingCarAbsoluteX = trackerDict[eachSurroundingCarID][udx][0]  # udx for coresponnding Frame and 0 is poseX index in TrakcerDict list
                            surroundingCarAbsoluteY = trackerDict[eachSurroundingCarID][udx][1]  # udx for coresponnding Frame and 1 is poseY index in TrakcerDict list
                            surroundingCarLocalX = ((surroundingCarAbsoluteX - intitalX)-minRealtiveX)/(maxRealtiveX-minRealtiveX)
                            surroundingCarLocalY = ((surroundingCarAbsoluteY - intitalY)-minRealtiveY)/(maxRealtiveY-minRealtiveY)
                            # Extract rest of the features
                            surroundingCarVelocityX = (trackerDict[eachSurroundingCarID][udx][2]-minVelocityX)/(maxVelocityX-minVelocityX)  # udx for coresponnding Frame and 2 is velocityX index in TrakcerDict list
                            surroundingCarVelocityY = (trackerDict[eachSurroundingCarID][udx][3]-minVelocityY)/(maxVelocityY-minVelocityY)  # udx for coresponnding Frame and 3 is velocityY index in TrakcerDict list
                            surroundingCarLaneID = trackerDict[eachSurroundingCarID][udx][4]  # udx for coresponnding Frame and 4 is lane ID index in TrakcerDict list
                            surroundingCarMovement = trackerDict[eachSurroundingCarID][udx][5]  # udx for coresponnding Frame and 5 is Movement index in TrakcerDict list
                            surroundingCarHeadwaySpace = trackerDict[eachSurroundingCarID][udx][6]  # udx for coresponnding Frame and 6 is headway space index in TrakcerDict list
                            surroundingCarHeadwayTime = trackerDict[eachSurroundingCarID][udx][7]  # udx for coresponnding Frame and 7 is headway time index in TrakcerDict list

                            # # # # Extract distance from the nearest junction
                            # # # surroundingCarSection = trackerDict[eachSurroundingCarID][udx][8]  # udx for coresponnding Frame and 8 is section index in TrakcerDict list
                            # # # surroundingCarIntersection = trackerDict[eachSurroundingCarID][udx][9]  # udx for coresponnding Frame and 9 is intersection index in TrakcerDict list
                            # # # juncDist = CalculateNearestJuncLoc(surroundingCarSection, surroundingCarIntersection, surroundingCarAbsoluteX, surroundingCarAbsoluteY)

                            # If the other vehicle distance is more that allowable then append zeros
                            otherDist = math.sqrt((((surroundingCarAbsoluteX-targetLocalX)**2)+((surroundingCarAbsoluteY-targetLocalY)**2)))

                            # # # # # surrounding current temp list to check thhe more than one value
                            # # # # surroudingOneCheck = [surroundingCarLocalX,surroundingCarLocalY,surroundingCarVelocityX,surroundingCarVelocityY,surroundingCarLaneID,surroundingCarMovement,surroundingCarHeadwaySpace,surroundingCarHeadwayTime]
                            # # # # for tempIdx,eachTempPredVal in enumerate(surroudingOneCheck):
                            # # # #     if(eachTempPredVal > 1.0):
                            # # # #         print('The ' + str(tempIdx) + ' value of the surrounding is more than 1.0!!!')
                            # # # #         print('The value is ' + str(eachTempPredVal) + ' !!!')
                            # # # #         print('The RAW is ' + str(surroudingOneCheck[tempIdx]) + ' !!!')
                            # # # #         nanVal = 10000/0
                            # # # #         sys.exit()
                            
                            if(otherDist>maximumSurroundingCarDist):
                                tempPredictionInput.extend(inputZeroPadding)
                            else:
                                # tempInput = [localX,localY,velocityX,velocityY,laneID,movement]
                                # With the headway and space
                                # # tempPredictionInput.extend([surroundingCarLocalX,surroundingCarLocalY,surroundingCarVelocityX,surroundingCarVelocityY,surroundingCarLaneID,surroundingCarMovement,surroundingCarHeadwaySpace,surroundingCarHeadwayTime])
                                # Without the headway and space
                                tempPredictionInput.extend([surroundingCarLocalX,surroundingCarLocalY,surroundingCarVelocityX,surroundingCarVelocityY,surroundingCarLaneID,surroundingCarMovement])

                        for adx in range(0,predictionPaddingCount):
                            tempPredictionInput.extend(inputZeroPadding)

                    # Else the surrouding car count is more than the decided surrouding car count then select the nearest 4 cars
                    else:
                        # Get the surrounding car's coresponding frame position and calculate distance
                        surroundingCarDistanceList = []
                        for eachSurroundingCarID in surroudingCarIds:
                            # Check for time
                            surroundingCarTime = trackerDict[eachSurroundingCarID][udx][-2]  # udx for coresponnding Frame and second last is time index in TrakcerDict list
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
                            surroundingCarLocalX = ((surroundingCarAbsoluteX - intitalX)-minRealtiveX)/(maxRealtiveX-minRealtiveX)
                            surroundingCarLocalY = ((surroundingCarAbsoluteY - intitalY)-minRealtiveY)/(maxRealtiveY-minRealtiveY)
                            # Extract rest of the features
                            # tempInput = [localX,localY,velocityX,velocityY,laneID,movement]
                            surroundingCarVelocityX = (trackerDict[eachReleventSurroundingID[0]][udx][2]-minVelocityX)/(maxVelocityX-minVelocityX)  # udx for coresponnding Frame and 2 is velocityX index in TrakcerDict list
                            surroundingCarVelocityY = (trackerDict[eachReleventSurroundingID[0]][udx][3]-minVelocityY)/(maxVelocityY-minVelocityY)  # udx for coresponnding Frame and 3 is velocityY index in TrakcerDict list
                            surroundingCarLaneID = trackerDict[eachReleventSurroundingID[0]][udx][4]  # udx for coresponnding Frame and 4 is Lane index in TrakcerDict list
                            surroundingCarMovement = trackerDict[eachReleventSurroundingID[0]][udx][5]  # udx for coresponnding Frame and 5 is Movement index in TrakcerDict list
                            surroundingCarHeadwaySpace = trackerDict[eachReleventSurroundingID[0]][udx][6]  # udx for coresponnding Frame and 6 is Headway Space index in TrakcerDict list
                            surroundingCarHeadwayTime = trackerDict[eachReleventSurroundingID[0]][udx][7]  # udx for coresponnding Frame and 7 is Headway Time index in TrakcerDict list
                            # Extract distance from the nearest junction
                            # # surroundingCarSection = trackerDict[eachReleventSurroundingID[0]][udx][8]  # udx for coresponnding Frame and 8 is section index in TrakcerDict list
                            # # surroundingCarIntersection = trackerDict[eachReleventSurroundingID[0]][udx][9]  # udx for coresponnding Frame and 9 is intersection index in TrakcerDict list
                            # # juncDist = CalculateNearestJuncLoc(surroundingCarSection, surroundingCarIntersection, surroundingCarAbsoluteX, surroundingCarAbsoluteY)

                            # # # # surrounding current temp list to check thhe more than one value
                            # # # surroudingOneCheck = [surroundingCarLocalX,surroundingCarLocalY,surroundingCarVelocityX,surroundingCarVelocityY,surroundingCarLaneID,surroundingCarMovement,surroundingCarHeadwaySpace,surroundingCarHeadwayTime]
                            # # # for tempIdx,eachTempPredVal in enumerate(surroudingOneCheck):
                            # # #     if(eachTempPredVal > 1.0):
                            # # #         print('The ' + str(tempIdx) + ' value of the surrounding second is more than 1.0!!!')
                            # # #         print('The value is ' + str(eachTempPredVal) + ' !!!')
                            # # #         print('The RAW is ' + str(surroudingOneCheck[tempIdx]) + ' !!!')
                            # # #         nanVal = 10000/0
                            # # #         sys.exit()

                            # If the other vehicle distance is more that allowable then append zeros
                            otherDist = math.sqrt((((surroundingCarAbsoluteX-targetLocalX)**2)+((surroundingCarAbsoluteY-targetLocalY)**2)))
                            if(otherDist>maximumSurroundingCarDist):
                                tempPredictionInput.extend(inputZeroPadding)
                            else:
                                # With the headway space and time
                                # # tempPredictionInput.extend([surroundingCarLocalX,surroundingCarLocalY,surroundingCarVelocityX,surroundingCarVelocityY,surroundingCarLaneID,surroundingCarMovement,surroundingCarHeadwaySpace,surroundingCarHeadwayTime])
                                # Without the headway time and space
                                tempPredictionInput.extend([surroundingCarLocalX,surroundingCarLocalY,surroundingCarVelocityX,surroundingCarVelocityY,surroundingCarLaneID,surroundingCarMovement])
                    
                    # Add the current frame input info in the input list
                    predicitionInputList.append(tempPredictionInput)

                # Add the current Vehicles all history frame input to the prediction dict object (input field)
                predictionDict[eachEligibleKey].input = predicitionInputList

                # print('Prediction input preped!!!!')

                # Add the ground truth output pose in to the prediction dict object for error calculation
                # tempInput = [localX,localY,velocityX,velocityY,laneID,movement]
                outputInfo = totalInfo[historyTemporal:historyTemporal+futureTemporal]
                #tempGroundTruthPoseList = []
                for eachOutputInfo in outputInfo:
                    groundTruthPoseX = eachOutputInfo[0] - intitalX    # 0 is poseX index in trakcer dict list
                    groundTruthPoseY = eachOutputInfo[1] - intitalY    # 1 is poseY index in trakcer dict list
                    groundTruthVelocityX = eachOutputInfo[2]                # 2 is velocityX index in trakcer dict list
                    groundTruthVelocityY = eachOutputInfo[3]                # 3 is velocityY index in trakcer dict list
                    trueMovement = eachOutputInfo[5]                        # 5 is movement index in trakcer dict list
                    nextMovementClassData = MovementToClassForm(trueMovement)


                    # # # # Add the section intersetion values for decoder distane from junc calculation
                    # # # truthSection = eachOutputInfo[8]  # 8 is section index in trakcer dict list
                    # # # truthIntersection = eachOutputInfo[9]  # 9 is intersection index in trakcer dict list

                    # Denormalize poseX and poseY as the traker dict is for input and it is normalized
                    # No need to denormalize the poses are absolute and not normalized
                    # denormPoseX = (groundTruthPoseX*(maxLocalX-minLocalX)+minLocalX)
                    # denormPoseY = (groundTruthPoseY*(maxLocalY-minLocalY)+minLocalY)

                    predictionDict[eachEligibleKey].groundTruth.append([nextMovementClassData[0],nextMovementClassData[1],nextMovementClassData[2],groundTruthVelocityX,groundTruthVelocityY,groundTruthPoseX,groundTruthPoseY])
                    # # # predictionDict[eachEligibleKey].sectionIntersection.append([truthSection,truthIntersection])
            
            # No need to maintain decoder for the target vehicle as we will store for all eligible vehicles
            # # # Initialize decoderInputData list to hold the target decoder input data        
            # # decoderInputData = []
            # Add the decoder inputs in the prediction dict against each vehicle
            # Predict the encoder state for each vehicle and update the prediction dict state values
            for eachPredDictKey in predictionDict.keys():
                lastInput = predictionDict[eachPredDictKey].input[-1]
                predDecoderInput = []
                for bdx in range(0,len(lastInput),inputFeatureCount):  # inputFeatureCount 6 is number of input features for each car
                    lastInputPoseX = lastInput[bdx]
                    lastInputPoseY = lastInput[bdx+1]
                    lastInputVelocityX = lastInput[bdx+2]
                    lastInputVelocityY = lastInput[bdx+3]
                    lastInputMovement = lastInput[bdx+5]
                    lastInputClassInfo = MovementToClassForm(lastInputMovement)
                    # # #lastDistFromJunc = lastInput[bdx+6]
                    predDecoderInput.extend([lastInputPoseX,lastInputPoseY,lastInputVelocityX,lastInputVelocityY,lastInputClassInfo[0],lastInputClassInfo[1],lastInputClassInfo[2]])
                # Add the prepered decoder input in the prediction dict object (decoder input field)
                predictionDict[eachPredDictKey].decoderInput = predDecoderInput[:]

                # Each predDecoderInput should be length of globalDecoderFeatures
                if(len(predDecoderInput) != globalDecoderFeatures):
                    print('Global decoder feature count is not eas expected!!!')
                    print('Length should be ' + str(globalDecoderFeatures))
                    print('Length actula be ' + str(len(predDecoderInput)))
                    nanVal = 1000/0

                # No need to maintain decoder for the target vehicle as we will store for all eligible vehicles
                # # # If the predictKey is targetID then add the first decoder Input in the decoderInputData
                # # if(eachPredDictKey == targetUpdatedID):
                # #     decoderInputData.append(predDecoderInput[:])

                # Add the current vehicle decoder input to the list for later intermediate data preperation
                # predictionDict[eachPredDictKey].decoderInputList.append(predDecoderInput[:])
                predictionDict[eachPredDictKey].decoderInputList = [predDecoderInput[:]]

                # Get the input for the current vehicle to predict the encoder state
                currentPredInput = np.array(predictionDict[eachPredDictKey].input).reshape(1,historyTemporal,globalInputFeatures)

                # Predict the Encoder state for that specific vehicle and update the prediction dict
                currentState = encoder_model.predict(currentPredInput)
                predictionDict[eachPredDictKey].state = currentState

                # print('state predicted!!!!')
            
            # Predict till the decided future temporal
            for cdx in range(futureTemporal):

                # # #print('Processing time : ' + str(cdx))

                # Predict the next frame for each vechicle in the prediction dict
                for eachPredDictKey in predictionDict.keys():
                    # Prepare the target seq and state for the current Vehicle
                    target_seq = np.array(predictionDict[eachPredDictKey].decoderInput).reshape(1,1,globalDecoderFeatures)
                    predState = predictionDict[eachPredDictKey].state

                    # This is for seperate Vx and Vy module
                    # # # Predict the next frame for the current vehicle
                    # # classPred, velcoityPredX, velcoityPredY, posePred, h1, c1, h2, c2 = decoder_model.predict([target_seq] + predState)
                    # # # Store the prediction in the prediction dict w.r.t the coresponding vehicle
                    # # finalOutput = [posePred[0][0][0],posePred[0][0][1],velcoityPredX[0][0][0],velcoityPredY[0][0][0],classPred[0][0][0],classPred[0][0][1],classPred[0][0][2]]

                    # This is for single Vx and Vy module
                    # Predict the next frame for the current vehicle
                    classPred, velcoityPred, posePred, h1, c1, h2, c2 = decoder_model.predict([target_seq] + predState)
                    # Store the prediction in the prediction dict w.r.t the coresponding vehicle
                    finalOutput = [posePred[0][0][0],posePred[0][0][1],velcoityPred[0][0][0],velcoityPred[0][0][1],classPred[0][0][0],classPred[0][0][1],classPred[0][0][2]]
                    
                    
                    # Finally add it to the prediction dict
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
                    lastOutputVelocityX = lastOutput[2]   # 2 is velocityX index in output list of prediction dict
                    lastOutputVelocityY = lastOutput[3]   # 3 is velocityY index in output list of prediction dict
                    lastClassOutput0 = lastOutput[4]   # 4 is 0 class info index in output list of prediction dict
                    lastClassOutput1 = lastOutput[5]   # 5 is 1 class info index in output list of prediction dict
                    lastClassOutput2 = lastOutput[6]   # 5#6 is 2 class info index in output list of prediction dict

                    # Calculate the absolute position using initial pose and relative predicted pose to estimate the surrounding car dist
                    targetInitialPose = predictionDict[eachPredDictKey].initialPose
                    targetAbsolutePoseX = targetInitialPose[0] + lastOutputPoseX   # 0 is the index for poseX in prediction object initialPose field
                    targetAbsolutePoseY = targetInitialPose[1] + lastOutputPoseY   # 1 is the index for poseY in prediction object initialPose field

                    # Normalize poseX, poseY and velocity before adding to the decoder input
                    normalizedPredPoseX = (lastOutputPoseX-minRealtiveX)/(maxRealtiveX-minRealtiveX)
                    normalizedPredPoseY = (lastOutputPoseY-minRealtiveY)/(maxRealtiveY-minRealtiveY)
                    normalizedPredVelocityX = (lastOutputVelocityX-minVelocityX)/(maxVelocityX-minVelocityX)
                    normalizedPredVelocityY = (lastOutputVelocityY-minVelocityY)/(maxVelocityY-minVelocityY)

                    # # # # # Calculate the distance from the nearest junction using the section intersection from prediction dict object and converted absolute values using predicted local values
                    # # # # currentTargetSection = predictionDict[eachPredDictKey].sectionIntersection[cdx][0] # cdx is current future time which is same as the index in the sectionIntersection list in pred object and 0 is for first item is Section
                    # # # # currentTargetIntersection = predictionDict[eachPredDictKey].sectionIntersection[cdx][1] # cdx is current future time which is same as the index in the sectionIntersection list in pred object and 1 is for second item is Intersection
                    # # # # targetDistFromJunc = CalculateNearestJuncLoc(currentTargetSection, currentTargetIntersection, targetAbsolutePoseX, targetAbsolutePoseY)

                    # Finally add the normalized values into the temp decoder input
                    tempDecoderInput = [normalizedPredPoseX,normalizedPredPoseY,normalizedPredVelocityX,normalizedPredVelocityY,lastClassOutput0,lastClassOutput1,lastClassOutput2]

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
                            lastOutputVelocityX = lastOutput[2]   # 2 is velocityX index in output list of prediction dict
                            lastOutputVelocityY = lastOutput[3]   # 3 is velocityY index in output list of prediction dict
                            lastClassOutput0 = lastOutput[4]   # 4 is 0 class info index in output list of prediction dict
                            lastClassOutput1 = lastOutput[5]   # 5 is 1 class info index in output list of prediction dict
                            lastClassOutput2 = lastOutput[6]   # 6 is 2 class info index in output list of prediction dict

                            # Normalize poseX, poseY and velocity before adding to the decoder input
                            normalizedPredPoseX = (lastOutputPoseX-minRealtiveX)/(maxRealtiveX-minRealtiveX)
                            normalizedPredPoseY = (lastOutputPoseY-minRealtiveY)/(maxRealtiveY-minRealtiveY)
                            normalizedPredVelocityX = (lastOutputVelocityX-minVelocityX)/(maxVelocityX-minVelocityX)
                            normalizedPredVelocityY = (lastOutputVelocityY-minVelocityY)/(maxVelocityY-minVelocityY)

                            # Calculate the absolute position using initial pose and relative predicted pose to estimate its distance from target car
                            surroundingInitialPose = predictionDict[eachDecoderSurroundingCarID].initialPose
                            surroundingAbsoluteX = surroundingInitialPose[0] + lastOutputPoseX # 0 is the index for poseX in prediction object initialPose field
                            surroundingAbsoluteY = surroundingInitialPose[1] + lastOutputPoseY # 1 is the index for poseY in prediction object initialPose field

                            # # # # # Calculate the distance from the nearest junction using the section intersection from prediction dict object and converted absolute values using predicted local values
                            # # # # currentSurroundingSection = predictionDict[eachDecoderSurroundingCarID].sectionIntersection[cdx][0] # cdx is current future time which is same as the index in the sectionIntersection list in pred object and 0 is for first item is Section
                            # # # # currentSurroundingIntersection = predictionDict[eachDecoderSurroundingCarID].sectionIntersection[cdx][1] # cdx is current future time which is same as the index in the sectionIntersection list in pred object and 1 is for first item is Intersection
                            # # # # surroundingDistFromJunc = CalculateNearestJuncLoc(currentSurroundingSection, currentSurroundingIntersection, surroundingAbsoluteX, surroundingAbsoluteY)

                            # If the other vehicle distance is more that allowable then append zeros
                            otherDist = math.sqrt((((surroundingAbsoluteX-targetAbsolutePoseX)**2)+((surroundingAbsoluteY-targetAbsolutePoseY)**2)))

                            # Finally add the normalized values into the temp decoder input
                            if(otherDist>maximumSurroundingCarDist):
                                tempDecoderInput.extend(decoderZeroPadding)
                            else:
                                tempDecoderInput.extend([normalizedPredPoseX,normalizedPredPoseY,normalizedPredVelocityX,normalizedPredVelocityY,lastClassOutput0,lastClassOutput1,lastClassOutput2])
                        
                        predZeroPadList = decoderZeroPadding
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
                            lastOutputVelocityX = lastSurroundingOutput[2]   # 2 is velocity index in output list of prediction dict
                            lastOutputVelocityY = lastSurroundingOutput[3]   # 3 is velocity index in output list of prediction dict
                            lastClassOutput0 = lastSurroundingOutput[4]   # 3 is 0 class info index in output list of prediction dict
                            lastClassOutput1 = lastSurroundingOutput[5]   # 4 is 1 class info index in output list of prediction dict
                            lastClassOutput2 = lastSurroundingOutput[6]   # 5 is 2 class info index in output list of prediction dict

                            # Normalize poseX, poseY and velocity before adding to the decoder input
                            normalizedPredPoseX = (lastOutputPoseX-minRealtiveX)/(maxRealtiveX-minRealtiveX)
                            normalizedPredPoseY = (lastOutputPoseY-minRealtiveY)/(maxRealtiveY-minRealtiveY)
                            normalizedPredVelocityX = (lastOutputVelocityX-minVelocityX)/(maxVelocityX-minVelocityX)
                            normalizedPredVelocityY = (lastOutputVelocityY-minVelocityY)/(maxVelocityY-minVelocityY)

                            # Calculate the absolute position using initial pose and relative predicted pose to estimate its distance from target car
                            surroundingInitialPose = predictionDict[eachDecoderReleventSurroundingID[0]].initialPose
                            surroundingAbsoluteX = surroundingInitialPose[0] + lastOutputPoseX # 0 is the index for poseX in prediction object initialPose field
                            surroundingAbsoluteY = surroundingInitialPose[1] + lastOutputPoseY # 1 is the index for poseY in prediction object initialPose field

                            # If the other vehicle distance is more that allowable then append zeros
                            otherDist = math.sqrt((((surroundingAbsoluteX-targetAbsolutePoseX)**2)+((surroundingAbsoluteY-targetAbsolutePoseY)**2)))

                            # # # # # Calculate the distance from the nearest junction using the section intersection from prediction dict object and converted absolute values using predicted local values
                            # # # # currentSurroundingSection = predictionDict[eachDecoderReleventSurroundingID[0]].sectionIntersection[cdx][0] # cdx is current future time which is same as the index in the sectionIntersection list in pred object and 0 is for first item is Section
                            # # # # currentSurroundingIntersection = predictionDict[eachDecoderReleventSurroundingID[0]].sectionIntersection[cdx][1] # cdx is current future time which is same as the index in the sectionIntersection list in pred object and 1 is for first item is Intersection
                            # # # # surroundingDistFromJunc = CalculateNearestJuncLoc(currentSurroundingSection, currentSurroundingIntersection, surroundingAbsoluteX, surroundingAbsoluteY)

                            if(otherDist>maximumSurroundingCarDist):
                                tempDecoderInput.extend(decoderZeroPadding)
                            else:
                                # Finally add the normalized values into the temp decoder input
                                tempDecoderInput.extend([normalizedPredPoseX,normalizedPredPoseY,normalizedPredVelocityX,normalizedPredVelocityY,lastClassOutput0,lastClassOutput1,lastClassOutput2])
                    
                    # Finally update the decoder input in the prediction dict
                    predictionDict[eachPredDictKey].decoderInput = tempDecoderInput

                    # No need to maintain decoder for the target vehicle as we will store for all eligible vehicles
                    # # # Add the current decoder input to the localDecoderInput if the key is targetVehicleID
                    # # if(eachPredDictKey == targetUpdatedID):
                    # #     decoderInputData.append(tempDecoderInput)

                    # No need to maintain decoder for the target vehicle as we will store for all eligible vehicles
                    predictionDict[eachPredDictKey].decoderInputList.append(tempDecoderInput)

            # No need to maintain decoder for the target vehicle as we will store for all eligible vehicles
            # # # # # DecoderInputData is populated already. Extract rest of the input/output from the prediction dict
            # # # # # For decoder input data remove the last item as that is extra due to the last step prediction
            # # # # decoderInputData.pop(-1)

            # Create samples using all the predicted vehicles
            for eachPredDictKey in predictionDict.keys():
                # Creating new list for the decoder input for each vehicle
                # Pop the last item due to the additonal cycle during prediction
                # The first instance is already added using the last input
                predictionDict[eachPredDictKey].decoderInputList.pop(-1)
                localDecoderInput = predictionDict[eachPredDictKey].decoderInputList
                localXData = predictionDict[eachPredDictKey].input
                totalGroundTruthOutputList = predictionDict[eachPredDictKey].groundTruth
                localYMovementData = []
                localYVelData = []
                localYPoseData = []
                for eachGroundTurthOutput in totalGroundTruthOutputList:
                    localYMovementData.append([eachGroundTurthOutput[0],eachGroundTurthOutput[1],eachGroundTurthOutput[2]])   # GroundTruth class0, class1, clas2 index 0,1,2
                    localYVelData.append([eachGroundTurthOutput[3],eachGroundTurthOutput[4]])   # GroundTruth velocityX and velocityY index 3,4
                    localYPoseData.append([eachGroundTurthOutput[5],eachGroundTurthOutput[6]])   # GroundTruth PoseX and PoseY index 5,6

                # Calculate the current error for each sample and append to the global manager list for intermediate error calculation
                # Get the predicted and ground truth poses from the predict dict object
                predictedIntermediatePose = predictionDict[eachPredDictKey].output

                # Length of both these lists should be equal
                if(len(totalGroundTruthOutputList) != len(predictedIntermediatePose)):
                    print('Ground truth and predicted pose lists are not equal while intermediate error calculation')
                    sys.exit()

                localErrorList = []
                for errorIdx,eachPose in enumerate(predictedIntermediatePose):
                    predX = predictedIntermediatePose[errorIdx][0] # 0 is poseX index in output list of prediction dict 
                    predY = predictedIntermediatePose[errorIdx][1] # 1 is poseY index in output list of prediction dict
                    trueX = totalGroundTruthOutputList[errorIdx][5] # 5 is poseX index in Ground Truth list of prediction dict 
                    trueY = totalGroundTruthOutputList[errorIdx][6] # 6 is poseY index in Ground Truth list of prediction dict
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
                decoderInputDataLength = len(localDecoderInput)
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
                    validationProcessList.append([localXData,localDecoderInput,localYMovementData,localYVelData,localYPoseData])
                elif(currentTrainOrValStr == trainStr):
                    trainProcessList.append([localXData,localDecoderInput,localYMovementData,localYVelData,localYPoseData])
                else:
                    print('Unknown Train Val string')
                    sys.exit()

            

            # # # # # Calculate the current error for only target vehicle and append to the global manager list for intermediate error calculation
            # # # # # Get the predicted and ground truth poses from the predict dict object
            # # # # predictedIntermediatePoseTarget = predictionDict[targetUpdatedID].output

            # # # # # Extract the ground truth for the target vehicle only
            # # # # totalGroundTruthOutputListTarget = predictionDict[targetUpdatedID].groundTruth

            # # # # # Length of both these lists should be equal
            # # # # if(len(totalGroundTruthOutputListTarget) != len(predictedIntermediatePoseTarget)):
            # # # #     print('Ground truth and predicted pose lists are not equal while intermediate error calculation')
            # # # #     sys.exit()

            # # # # localErrorList = []
            # # # # for errorIdx,eachPose in enumerate(predictedIntermediatePoseTarget):
            # # # #     predX = predictedIntermediatePoseTarget[errorIdx][0] # 0 is poseX index in output list of prediction dict 
            # # # #     predY = predictedIntermediatePoseTarget[errorIdx][1] # 1 is poseY index in output list of prediction dict
            # # # #     trueX = totalGroundTruthOutputListTarget[errorIdx][5] # 5 is poseX index in Ground Truth list of prediction dict 
            # # # #     trueY = totalGroundTruthOutputListTarget[errorIdx][6] # 6 is poseY index in Ground Truth list of prediction dict
            # # # #     euclidianError = math.sqrt(((predX-trueX)**2) + ((predY-trueY)**2)) * feetToMeter
            # # # #     localErrorList.append(euclidianError)

            # # # # # Check the error list should be of lenght futureTemporal
            # # # # errorListLen = len(localErrorList)
            # # # # if(errorListLen != futureTemporal):
            # # # #     print('Error list is of not expected length!!!')
            # # # #     print('Expected error list length : ' + str(futureTemporal))
            # # # #     print('Received error list length : ' + str(errorListLen))
            # # # #     sys.exit()

            # # # # # Append the current local list to the main manager list
            # # # # errorManagerList.append(localErrorList)
            # # # # errorCountList.append(0)


        countList.append(0)
        totalSamplesProcessed = len(countList)
        print('Finished Processing Sample : ' + str(totalSamplesProcessed))


# Function to go through all the expected samples manually to identify the new min max for both pose and motion
def IntermediateMinMaxPopulate(processItemSubList):

    # Start going through all the vehicles ID to create virtual samples and estimate the min max and update the global min max
    # for both pose and motion.  

    
    for eachProcessItem in processItemSubList:
        # Retrive the Process item
        eachRelevenatVehicle = eachProcessItem[0]   # string
        currentTrainOrValStr = eachProcessItem[1]   # string

        currentReleventVehicleList = dictByVehicles[eachRelevenatVehicle]
        currentReleventVehicleLength = len(currentReleventVehicleList)

        # loop through the vehicle list to check if there is any single change of laneID. If not means zero lane changes measn ignore
        # Get the intitial lane first to compare
        changeFlag = False
        initialLaneID = currentReleventVehicleList[0][laneIDIndex]
        for eachTargetItem in currentReleventVehicleList:
            currentLaneID = eachTargetItem[laneIDIndex]
            if(currentLaneID != initialLaneID):
                changeFlag = True
                break
        
        # Check if any lane chnage happend, false means no lane chnage happend
        if(changeFlag == False):
            straightVehicles.append(0)
            if(len(straightVehicles) > includedStraightVehicles):
                continue

        # Get the current target vehicle ID
        targetUpdatedID = eachRelevenatVehicle


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
        for idx in range(historyTemporal+10+prevNextFrameCount,currentReleventVehicleLength-futureTemporal-10-prevNextFrameCount,30):   #  15

            # Prepare the trakcer Dict
            trackerDict = dict()
            trackerDict[targetUpdatedID] = []

            # for input
            for jdx in range(idx-historyTemporal,idx+futureTemporal):

                currentVechicleID = currentReleventVehicleList[jdx][vechileIDIndex]
                currentLocalX = currentReleventVehicleList[jdx][localXIndex]
                currentLocalY = currentReleventVehicleList[jdx][localYIndex]
                # # # currentVelocity = currentReleventVehicleList[jdx][velocityIndex]
                currentLaneID = currentReleventVehicleList[jdx][laneIDIndex]
                # # # currentDirection = currentReleventVehicleList[jdx][directionIndex]
                # # # currentMovement = currentReleventVehicleList[jdx][movementIndex]
                currentTime = currentReleventVehicleList[jdx][globalTimeIndex]
                currentFrame = currentReleventVehicleList[jdx][frameIDIndex]
                # # # currentSection = currentReleventVehicleList[jdx][sectionIndex]
                # # # currentIntersection = currentReleventVehicleList[jdx][intersectionIndex]
                currentHeadwaySpace = currentReleventVehicleList[jdx][headwaySpaceIndex]
                currentHeadwayTime = currentReleventVehicleList[jdx][headwayTimeIndex]

                # Get the last 3 and next 3 lane ids to estimate the lane change maneuver 
                lastThreeLaneIds = np.array(currentReleventVehicleList[jdx-prevNextFrameCount:jdx])[:,laneIDIndex]
                nextThreeLaneIds = np.array(currentReleventVehicleList[jdx:jdx+prevNextFrameCount])[:,laneIDIndex]
                totoalLaneIDs = list(lastThreeLaneIds)
                totoalLaneIDs.extend(list(nextThreeLaneIds))

                # Estmiate the lane change maneuver
                currentMovement = TargetLaneChanageManeuver(totoalLaneIDs)       #  currentVehicleList[jdx][movementIndex]

                # Estimate velocityX and VelocityY
                prevAbsoluteX = currentReleventVehicleList[jdx-1][localXIndex]
                prevAbsoluteY = currentReleventVehicleList[jdx-1][localYIndex]

                # For I80/US101 the vehicles are always moving forward
                currentVelocityX = currentLocalX-prevAbsoluteX        
                currentVelocityY = currentLocalY-prevAbsoluteY


                # Prepeare the target vehicle current input and append and at the end of the tracker dict list
                # tempInput = [localX,localY,velocityX,velocityY,laneID,movement]
                # Without headway time and space
                # # dictInput = [currentLocalX,currentLocalY,currentVelocityX,currentVelocityY,currentLaneID,currentMovement,currentTime,currentFrame]
                # With headway time and space
                dictInput = [currentLocalX,currentLocalY,currentVelocityX,currentVelocityY,currentLaneID,currentMovement,currentHeadwaySpace,currentHeadwayTime,currentTime,currentFrame]
                trackerDict[targetUpdatedID].append(dictInput)


                # Get the surrounding cars
                otherVehicles = dictByFrames[str(currentTime)]

                # Get the previous timestamp (-100) other vehicles velcotiy calculations
                prevOtherVehicles = dictByFrames[str(currentTime-100)]
                # Get the next timestamp (+100) also in case the vehicle is not present in prev timestamp
                nextOtherVehicles = dictByFrames[str(currentTime+100)]

                # Target vehicle removal flag 
                targetRemovedFlag = 0

                for eachOtherVehicle in otherVehicles:
                    currentVechicleID = eachOtherVehicle[vechileIDIndex]
                    otherVehicleID = str(eachOtherVehicle[vechileIDIndex])

                    if(str(currentVechicleID) == targetOriginalID):
                        targetRemovedFlag = 1
                        continue

                    currentLocalX = eachOtherVehicle[localXIndex]
                    currentLocalY = eachOtherVehicle[localYIndex]
                    # # # currentVelocity = eachOtherVehicle[velocityIndex]
                    currentLaneID = eachOtherVehicle[laneIDIndex]
                    # # # #currentDirection = eachOtherVehicle[directionIndex]
                    # # #currentMovement = eachOtherVehicle[movementIndex]
                    currentTime = eachOtherVehicle[globalTimeIndex]
                    currentFrame = eachOtherVehicle[frameIDIndex]
                    # # currentSection = eachOtherVehicle[sectionIndex]
                    # # currentIntersection = eachOtherVehicle[intersectionIndex]
                    # Headway space and time for other vehicles
                    otherHeadwaySpace = eachOtherVehicle[headwaySpaceIndex]
                    otherHeadwayTime = eachOtherVehicle[headwayTimeIndex]

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
                            otherVelocityX = currentLocalX-prevOtherLocalPoseX          
                            otherVelocityY = currentLocalY-prevOtherLocalPoseY
                            # Find the lane ID of the same othher vehicle for previous frame
                            otherPrevLaneID = eachOtherItem[laneIDIndex]
                            # Estimate the lane change maneuver for the surroudning vehice
                            otherLaneIDList = [otherPrevLaneID,currentLaneID]
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
                                otherVelocityX = nextOtherLocalPoseX-currentLocalX       
                                otherVelocityY = nextOtherLocalPoseY-currentLocalY
                                # Find the lane ID of the same other vehicle for next frame
                                otherNextLaneID = eachOtherItem[laneIDIndex]
                                # Estimate the lane change maneuver for the surroudning vehice
                                otherLaneIDList = [currentLaneID,otherNextLaneID]
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

                    # tempInput = [localX,localY,velocityX,velocityY,laneID,movement]
                    # Without headway time and space
                    # # dictInput = [currentLocalX,currentLocalY,otherVelocityX,otherVelocityY,currentLaneID,currentMovement,currentTime,currentFrame]
                    # With headway time and space
                    dictInput = [currentLocalX,currentLocalY,otherVelocityX,otherVelocityY,currentLaneID,currentMovement,otherHeadwaySpace,otherHeadwayTime,currentTime,currentFrame]


                    # append the surrounding car info in the trakcer dict
                    # Check if the vehicle ID exist in mapper dict
                    # if yes use the updated key to avoid duplication
                    # Vehicle Birth in tracker Dict
                    if (str(currentVechicleID) not in trackerDict.keys()):
                        trackerDict[str(currentVechicleID)] = []
                        trackerDict[str(currentVechicleID)].append(dictInput)
                    else:
                        # Check the diff of last frame and last time with current frame and current time to avide duplicate vehicle IDs
                        lastTime = trackerDict[str(currentVechicleID)][-1][-2]  # -1 for last item and second last is time index in TrakcerDict list
                        lastFrame = trackerDict[str(currentVechicleID)][-1][-1] # -1 for last item and last is frame index in TrakcerDict list
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

            # print('Tracker populated!!! going for predition')

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

            
            # Create a prediction dictionary to hold the relevan information regarding each target vehicle 
            predictionDict = dict()


            # Populate all the input/gournd truth infos in the prediction dictionary  
            ########################## eligibleVehicleKeys replaced by modifiedEligibleKeys (for faster)    ######################################
            for eachEligibleKey in modifiedEligibleKeys:
                predictionInfoObj = PredictionInfos([],[],[],[],[],[],[])
                predictionDict[eachEligibleKey] = predictionInfoObj
                # Get all the input infos from the traker dict for that specific vehicle
                totalInfo = trackerDict[eachEligibleKey].copy()

                # Go through the entire list treasted as input as this loop is only for min max estimation.
                # The surroudning selection and relative pose estimation is same for both input and output
                # except for output the predicted values will be used. For min max estimation the ground truth values can be used  
                inputOutputInfoList = totalInfo[0:historyTemporal+futureTemporal]

                # Get the first element for relative movement calculation and add in the prediction object
                intitalX = inputOutputInfoList[0][0]  # 0 for first item and 0 for poseX index is 0 in trakcer dict
                intitalY = inputOutputInfoList[0][1]  # 0 for first item and 1 for poseY index is 0 in trakcer dict
                predictionDict[eachEligibleKey].initialPose = [intitalX,intitalY]
                poseXList = []
                poseYList = []
                velXList = []
                velYList = []
                for udx, eachInputInfo in enumerate(inputOutputInfoList):

                    # Add check for time
                    targetTime = eachInputInfo[-2]  # second last is time index in trakcer dict list

                    # Add the converted values to the list for min max estimation
                    targetLocalX = eachInputInfo[0]  # 0 is poseX index in trakcer dict list
                    targetLocalY = eachInputInfo[1]  # 1 is poseY index in trakcer dict list


                    poseXList.append(targetLocalX-intitalX) # Do not chane the targetLocalX as it was used for maxSurroundingDist calculation 
                    poseYList.append(targetLocalY-intitalY)  # Do not chane the targetLocalX as it was used for maxSurroundingDist calculation 

                    # Add the converted motion to the list for min max calculation
                    targetVelX = eachInputInfo[2] # 2 is for Vel X in the tracker dict
                    velXList.append(targetVelX)
                    targetVelY = eachInputInfo[3] # 3 is for Vel Y in the tracker dict
                    velYList.append(targetVelY) 


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
                            surroundingCarTime = trackerDict[eachSurroundingCarID][udx][-2]  # udx for coresponnding Frame and second last is time index in TrakcerDict list
                            if(surroundingCarTime!=targetTime):
                                print('Surrounding vehicle time mismatch during prediction!!!')
                                print('Surrounding vehicle time : ' + str(surroundingCarTime))
                                print('Target vehicle time : ' + str(targetTime))
                                sys.exit()

                            # tempInput = [localX,localY,velocityX,velocityY,laneID,movement]
                            # Extract the absolute pose and convert to normalized relative pose
                            surroundingCarAbsoluteX = trackerDict[eachSurroundingCarID][udx][0]  # udx for coresponnding Frame and 0 is poseX index in TrakcerDict list
                            surroundingCarAbsoluteY = trackerDict[eachSurroundingCarID][udx][1]  # udx for coresponnding Frame and 1 is poseY index in TrakcerDict list
                           
                            # If the other vehicle distance is more that allowable then append zeros
                            otherDist = math.sqrt((((surroundingCarAbsoluteX-targetLocalX)**2)+((surroundingCarAbsoluteY-targetLocalY)**2)))
                            
                            if(otherDist<maximumSurroundingCarDist):
                                # Add the relative surroudng poseX PoseY for min max estimation only if was selected based on the surrounding dist
                                surroundingCarLocalX = surroundingCarAbsoluteX - intitalX
                                poseXList.append(surroundingCarLocalX)
                                surroundingCarLocalY = surroundingCarAbsoluteY - intitalY
                                poseYList.append(surroundingCarLocalY)
                                # Add the motionX and Y for surroudning cars for min max calculation
                                surroundingCarVelocityX = trackerDict[eachSurroundingCarID][udx][2]  # udx for coresponnding Frame and 2 is velocityX index in TrakcerDict list
                                velXList.append(surroundingCarVelocityX)
                                surroundingCarVelocityY = trackerDict[eachSurroundingCarID][udx][3]  # udx for coresponnding Frame and 3 is velocityY index in TrakcerDict list
                                velYList.append(surroundingCarVelocityY)


                    # Else the surrouding car count is more than the decided surrouding car count then select the nearest 4 cars
                    else:
                        # Get the surrounding car's coresponding frame position and calculate distance
                        surroundingCarDistanceList = []
                        for eachSurroundingCarID in surroudingCarIds:
                            # Check for time
                            surroundingCarTime = trackerDict[eachSurroundingCarID][udx][-2]  # udx for coresponnding Frame and second last is time index in TrakcerDict list
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

                            # If the other vehicle distance is more that allowable then append zeros
                            otherDist = math.sqrt((((surroundingCarAbsoluteX-targetLocalX)**2)+((surroundingCarAbsoluteY-targetLocalY)**2)))
                            if(otherDist<maximumSurroundingCarDist):
                                # If the surroduing dist check passes add it to the min max list for global min max
                                surroundingCarLocalX = surroundingCarAbsoluteX - intitalX
                                poseXList.append(surroundingCarLocalX)
                                surroundingCarLocalY = surroundingCarAbsoluteY - intitalY
                                poseYList.append(surroundingCarLocalY)
                                surroundingCarVelocityX = (trackerDict[eachReleventSurroundingID[0]][udx][2])  # udx for coresponnding Frame and 2 is velocityX index in TrakcerDict list
                                velXList.append(surroundingCarVelocityX)
                                surroundingCarVelocityY = (trackerDict[eachReleventSurroundingID[0]][udx][3])  # udx for coresponnding Frame and 3 is velocityY index in TrakcerDict list
                                velYList.append(surroundingCarVelocityY)


                # Once the history and future temporal processed check the min max for the current local list
                # and add those min max to the corresponding min max list
                minPoseX = min(poseXList)
                maxPoseX = max(poseXList)
                minPoseY = min(poseYList)
                maxPoseY = max(poseYList)
                minVelX = min(velXList)
                maxVelx = max(velXList)
                minVelY = min(velYList)
                maxVelY = max(velYList)

                # Add all the min max to the manager list, which later be converted to array to estimate individual min max
                currentMinMaxValues = [minPoseX,maxPoseX,minPoseY,maxPoseY,minVelX,maxVelx,minVelY,maxVelY]
                relativePoseMotionXYManegerList.append(currentMinMaxValues)


        minMaxCountList.append(0)
        totalSamplesProcessed = len(minMaxCountList)
        print('Finished MinMax Sample : ' + str(totalSamplesProcessed))



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

    for eachVehicle in vehicleLoadedData:
        # Get the vehicle ID
        vehicleIDStr = eachVehicle.rstrip()
        vehicleProcessList.append([vehicleIDStr,trainOrValStr])
    
    # valFileObj.close()
    vehicleFileObj.close()

    # Select the cores
    os.system("taskset -p -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 %d" % os.getpid())

    # numberOfSublist is the number of sublist to be created decided based on the GPU memory
    numberOfSublist = 50     # 4  # 65   # 65
    # Calculate approx number of item in each sublist
    totolItemCount = len(vehicleProcessList)
    n = int(totolItemCount/numberOfSublist)  

    # Change this back to big list....
    splittedList = [vehicleProcessList[i * n:(i + 1) * n] for i in range((len(vehicleProcessList) + n - 1) // n )] 

    # Assign alternative GPU ids to equally distribute model load
    totalGPUCount = 2
    gpuStrZero = '/gpu:0'
    gpuStrOne = '/gpu:1'
    # gpuStrTwo = '/gpu:2'
    gpuCounter = 0

    processes = []

    for eachSplitedList in splittedList:
        # Decide the gpu Id to be used for each sublist
        # Decide the GPU ID str
        selectedGPUId = gpuCounter%totalGPUCount
        if(selectedGPUId == 0):
            gpuSelectStr = gpuStrZero
        elif(selectedGPUId == 1):
            gpuSelectStr = gpuStrOne
        # elif(selectedGPUId == 2):
        #     gpuSelectStr = gpuStrTwo
        else:
            print('Module division with 3 should not give anything other than 0,1,2!!!')
            print('selectedGPUId : ' + str(selectedGPUId))
            sys.exit()

        # Incremeent the GPU counter 
        gpuCounter = gpuCounter + 1

        # Create each process sublist
        p = mp.Process(target=IntermediatePredictionForTraining, args=(eachSplitedList,gpuSelectStr,))
        processes.append(p)
        p.start()

    # Wait for all the current n process to finish. 
    for process in processes:
        process.join()

    # # # IntermediatePredictionForTraining(splittedList[0], gpuStrTwo)


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




# Intermediate process to go through all the samples and update the min max
def IntermediateMinMaxProcess(vehicleFileName,trainOrValStr):


    # Read the vehicle ID list and populate the process list
    vehicleFileObj = open(vehicleFileName, "r")
    vehicleLoadedData = vehicleFileObj.readlines()
    vehicleProcessList = []

    for eachVehicle in vehicleLoadedData:
        # Get the vehicle ID
        vehicleIDStr = eachVehicle.rstrip()
        vehicleProcessList.append([vehicleIDStr,trainOrValStr])
    
    # valFileObj.close()
    vehicleFileObj.close()

    # Select the cores
    os.system("taskset -p -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 %d" % os.getpid())

    # numberOfSublist is the number of sublist to be created decided based on the GPU memory
    numberOfSublist = 65   # 65
    # Calculate approx number of item in each sublist
    totolItemCount = len(vehicleProcessList)
    n = int(totolItemCount/numberOfSublist)  

    # Change this back to big list....
    splittedList = [vehicleProcessList[i * n:(i + 1) * n] for i in range((len(vehicleProcessList) + n - 1) // n )] 


    processes = []

    for eachSplitedList in splittedList:
        # Create each process sublist
        p = mp.Process(target=IntermediateMinMaxPopulate, args=(eachSplitedList,))
        processes.append(p)
        p.start()

    # Wait for all the current n process to finish. 
    for process in processes:
        process.join()

    # # # IntermediatePredictionForTraining(splittedList[0], gpuStrTwo)


def ReUpdateMinMax():

    # Declare all the global Velocity max min 
    global minVelocityX, maxVelocityX, minVelocityY, maxVelocityY, maxRealtiveX, maxRealtiveY, minRealtiveX, minRealtiveY

    # Once all the vehicles are processed convert the manager list to array
    # Hte min max list structure in the min max multi process is 
    # currentMinMaxValues = [minPoseX,maxPoseX,minPoseY,maxPoseY,minVelX,maxVelx,minVelY,maxVelY]
    minMaxValArray = np.array(relativePoseMotionXYManegerList)

    print('MinMax array shape: ' + str(minMaxValArray.shape))

    if(min(minMaxValArray[:,0]) < minRealtiveX):
        minRealtiveX = min(minMaxValArray[:,0])
    if(max(minMaxValArray[:,1]) > maxRealtiveX):
        maxRealtiveX = max(minMaxValArray[:,1])
    if(min(minMaxValArray[:,2]) < minRealtiveY):
        minRealtiveY = min(minMaxValArray[:,2])
    if(max(minMaxValArray[:,3]) > maxRealtiveY):
        maxRealtiveY = max(minMaxValArray[:,3])
    if(min(minMaxValArray[:,4]) < minVelocityX):
        minVelocityX = min(minMaxValArray[:,4])
    if(max(minMaxValArray[:,5]) > maxVelocityX):
        maxVelocityX = max(minMaxValArray[:,5])
    if(min(minMaxValArray[:,6]) < minVelocityY):
        minVelocityY = min(minMaxValArray[:,6])
    if(max(minMaxValArray[:,7]) > maxVelocityY):
        maxVelocityY = max(minMaxValArray[:,7])


def _training_worker_gen():

    import tensorflow as tf
    from tensorflow.keras.callbacks import LearningRateScheduler
    from tensorflow.keras import callbacks
    from tensorflow.keras import backend as K
    from tensorflow.keras.utils import Sequence
    # # # K.set_learning_phase(1)  # For batch normalization layer (0 = test, 1 = train)

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)


    # Compile the model
    print('Before model compile....')
    model,encoder_model,decoder_model = ModelArch()
    print('Model compiled!!!!!')

    # Data Genarator Class
    class CIFAR10Sequence(Sequence):

        def __init__(self, dataList, batch_size, path):
            self.dataList = dataList
            self.batch_size = batch_size
            self.path = path


        def __len__(self):
            # return int(np.ceil(len(self.dataList) / float(self.batch_size)))
            return int(len(self.dataList))

        def __getitem__(self, dxGen):

            # Get the current batch sample path
            currentBatchfilePath = self.dataList[dxGen]
            samplePath = self.path + currentBatchfilePath

            # Prepeare the paths and read the raw file
            encoderInputPath = samplePath + sampleFileNameList[0]
            decoderInputPath = samplePath + sampleFileNameList[1]
            classOutputPath = samplePath + sampleFileNameList[2]
            velOuputPath = samplePath + sampleFileNameList[3]
            poseOutputPath = samplePath + sampleFileNameList[4]

            rawEncoderInput = GenReadFromFile(encoderInputPath)
            rawDecoderInput = GenReadFromFile(decoderInputPath)
            rawClassOut = GenReadFromFile(classOutputPath)
            rawVelOut = GenReadFromFile(velOuputPath)
            rawPoseOut = GenReadFromFile(poseOutputPath)

            # Split the list based in the history or future temporal length for each sample
            totalEncoderInput = [rawEncoderInput[i * historyTemporal:(i + 1) * historyTemporal] for i in range((len(rawEncoderInput) + historyTemporal - 1) // historyTemporal )]
            totalDecoderInput = [rawDecoderInput[i * futureTemporal:(i + 1) * futureTemporal] for i in range((len(rawDecoderInput) + futureTemporal - 1) // futureTemporal )]
            totalClassOut = [rawClassOut[i * futureTemporal:(i + 1) * futureTemporal] for i in range((len(rawClassOut) + futureTemporal - 1) // futureTemporal )]
            totalVelOut = [rawVelOut[i * futureTemporal:(i + 1) * futureTemporal] for i in range((len(rawVelOut) + futureTemporal - 1) // futureTemporal )]
            totalPoseOut = [rawPoseOut[i * futureTemporal:(i + 1) * futureTemporal] for i in range((len(rawPoseOut) + futureTemporal - 1) // futureTemporal )]


            # Input array
            totalEncoderInputArray = np.array(totalEncoderInput).reshape(-1,historyTemporal,globalInputFeatures)
            totalDecoderInputArray = np.array(totalDecoderInput).reshape(-1,futureTemporal,globalDecoderFeatures)

            # Output array
            totalClassOutAray = np.array(totalClassOut).reshape(-1,futureTemporal,classOut)
            totalVelOutArray = np.array(totalVelOut).reshape(-1,futureTemporal,velcoityOut)
            totalPoseOutArray = np.array(totalPoseOut).reshape(-1,futureTemporal,poseOut)

            # Split the velocity array into Vx and Vy arrays for seperate fitting with two logcosh
            # # # totalVelocityVxArray, totalVelocityVyArray = np.split(np.array(totalVelOutArray), 2, -1)

            # # # #return [totalEncoderInputArray,totalDecoderInputArray],[totalClassOutAray,totalVelocityVxArray,totalVelocityVyArray,totalPoseOutArray]

            return [totalEncoderInputArray,totalDecoderInputArray],[totalClassOutAray,totalVelOutArray,totalPoseOutArray]

    # Early stopping callback 
    class EarlyStoppingByLossVal(callbacks.Callback):
        def __init__(self, monitor='val_loss', value=0.25, verbose=0):
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

    # Prepeare the train and val generator objects
    trainFolderPath = folderName + trainFolderName + '/'   # Extra '/' as the path was used this way in the generator
    trainSampleList = sorted(os.listdir(trainFolderPath), key=int)
    numberOfTrainSamples = len(trainSampleList)
    print('Number of Train sample : ' + str(numberOfTrainSamples))

    valFolderPath = folderName + valFolderName + '/'   # Extra '/' as the path was used this way in the generator
    valSampleList = sorted(os.listdir(valFolderPath), key=int)
    numberOfValSamples = len(valSampleList)
    print('Number of Validation sample : ' + str(numberOfValSamples))

    trainGen = CIFAR10Sequence(trainSampleList,batchSize,trainFolderPath)
    valGen = CIFAR10Sequence(valSampleList,batchSize,valFolderPath)
    stepsPerEpoch = numberOfTrainSamples 
    valStepsPerEpoch = numberOfValSamples 


    lrate = LearningRateScheduler(step_decay)

    # create the final callback
    esObj = EarlyStoppingByLossVal()
    callbacks_list = [esObj,lrate]   #[loss_history, lrate]

    print('Starting Model fit!!!!!!!!!!!')

    os.system("taskset -p -c 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25 %d" % os.getpid())

    print('Before model fit!!!!!')

    history = model.fit(trainGen, steps_per_epoch=stepsPerEpoch, epochs=initialNumberEpochs, verbose=1, validation_data=valGen, validation_steps=valStepsPerEpoch, callbacks=callbacks_list, max_queue_size=36, workers=18, use_multiprocessing=True, shuffle=True)

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


def _training_worker(XTrain,decoderTrainInput,YClassTrain,velocityTrain,YPoseTrain,NumberEpochs,XVal,decoderValInput,YClassVal,velocityVal,YPoseVal):

    import tensorflow as tf
    from tensorflow.keras.callbacks import LearningRateScheduler
    from tensorflow.keras import callbacks
    from tensorflow.keras import backend as K
    # # # K.set_learning_phase(1)  # For batch normalization layer (0 = test, 1 = train)

    # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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



    # Load the current model weights
    model.load_weights(mainModelFileName)

    # create the final callback and schedulers
    lrate = LearningRateScheduler(second_step_decay)
    esObj = EarlyStoppingByLossVal()
    callbacks_list = [esObj,lrate]   #[loss_history, lrate]

    # Both the decoder and normal input should be normalized and no values should be >1
    print('XTrain max :' + str(np.amax(XTrain)))
    print('decoderTrainInput max :' + str(np.amax(decoderTrainInput)))

    # Check all the array to see if the max value not exceds 1.0
    if(np.amax(XTrain)>1 or np.amax(decoderTrainInput)>1):
        print('One of the above array in not normalized properly!!!!')

    # Both the decoder and normal input should be normalized and no values should be >1
    print('XVal max :' + str(np.amax(XVal)))
    print('decoderValInput max :' + str(np.amax(decoderValInput)))

    # Check all the array to see if the max value not exceds 1.0
    if(np.amax(XVal)>1 or np.amax(decoderValInput)>1):
        print('One of the above array in not normalized properly!!!!')

    print('Starting Model fit!!!!!!!!!!!')


    history = model.fit([XTrain,decoderTrainInput], [YClassTrain,velocityTrain,YPoseTrain], batch_size=batchSize, epochs=secondNumberEpochs, verbose=1, validation_data=([XVal,decoderValInput],[YClassVal,velocityVal,YPoseVal]), callbacks=callbacks_list, shuffle=True)

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
            maxVelocityY = variableValue  # Inflate the min max a bit to handle the >1.0 problem, quick fix
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

def CleanUpMinMax():
    # Declare all the global Velocity max min 
    global minVelocityX, maxVelocityX, minVelocityY, maxVelocityY, maxRealtiveX, maxRealtiveY, minRealtiveX, minRealtiveY

    # Destroy the curent min max
    minVelocityX = 999
    maxVelocityX = -999
    minVelocityY = 999
    maxVelocityY = -999
    maxRealtiveX = -9999
    maxRealtiveY = -9999
    minRealtiveX = 9999
    minRealtiveY = 9999



if __name__ == '__main__':

    # # # # # Re-Load the Vehicle and Frame based Dictionaries to populate the min max gloab values and global dicts
    # # # # # # # global dictByFrames, dictByVehicles, validationVehicles, mapperDict
    # # # # dictByFrames,dictByVehicles, mapperDict = CreateVehicleAndFrameDict(testTrajFilePath)
    # # # # finalVehicleKeys = list(dictByVehicles.keys())
    # # # # finalVehicleKeys.sort(key=float)
    # # # # finalFrameKeys = list(dictByFrames.keys())
    # # # # finalFrameKeys.sort(key=float)

    # Declare all the global Velocity max min as for later re update of min max it will used to destroy
    # # global minVelocityX, maxVelocityX, minVelocityY, maxVelocityY, maxRealtiveX, maxRealtiveY, minRealtiveX, minRealtiveY

    # Update the global min max values but only once as the min max would be common
    UpdateMinMax()

    ######################################################################
    ###############   First Round Training ###############################
    ######################################################################

    # First Round Training (last param False, no need to load weight)
    training_process = mp.Process(target=_training_worker_gen)
    training_process.start()
    training_process.join()

    print('Waiting for memeory clear!!!')
    sleep(1)

    ######################################################################

    #########################################################################
    # Re estimate the min max as now we will be using TV and SVs as well for 
    # training/validation data. Due to this the original min max values won't
    # work. So we are going through all the potential samples to update the new
    # min max based on the samples. The potential samples will be the same for
    # each retrain loop, so I think it is okay to update the min max only once 
    # at the start.
    ######################################################################

    # First go through all the inputs for all the vehicles (TV and SVs) and all the ground truth outputs (for decoder input) to find 
    # new global min and max 
    # Clean up the old min max values populated through the MinMax.txt file before intital training round
    CleanUpMinMax()

    # This is for trainng data
    # For this re-create the training dictionaries
    # Clean up the dictionaries first 
    dictByFrames = dict()
    dictByVehicles = dict()
    mapperDict = dict()

    # Populate the dictionaries
    dictByFrames,dictByVehicles, mapperDict = CreateVehicleAndFrameDict(trainTrajFilePath)
    finalVehicleKeys = list(dictByVehicles.keys())
    finalVehicleKeys.sort(key=float)
    finalFrameKeys = list(dictByFrames.keys())
    finalFrameKeys.sort(key=float)

    # Now go through all the the training samples using multi process to check and update the min max values
    # Prepeare the intermediate arrays
    data_process = mp.Process(target=IntermediateMinMaxProcess, args = (trainingFileName,trainStr)) 
    data_process.start()
    data_process.join()

    # This is for validation data
    # For this re-create the training dictionaries
    # Clean up the dictionaries first 
    dictByFrames = dict()
    dictByVehicles = dict()
    mapperDict = dict()

    # Populate the dictionaries
    dictByFrames,dictByVehicles, mapperDict = CreateVehicleAndFrameDict(testTrajFilePath)
    finalVehicleKeys = list(dictByVehicles.keys())
    finalVehicleKeys.sort(key=float)
    finalFrameKeys = list(dictByFrames.keys())
    finalFrameKeys.sort(key=float)

    # Now go through all the the validation samples using multi process to check and update the min max values
    # Prepeare the intermediate arrays
    data_process = mp.Process(target=IntermediateMinMaxProcess, args = (validationFileName,validationStr)) 
    data_process.start()
    data_process.join()

    # Re-update the min max values after going through all the potentional samples before intermediate prediction stage of
    # the retrain loop. This is done as a seperate function as the Process function won't update global variable
    ReUpdateMinMax()


    # Print relative X Y max and each normalized array min max
    print('Relative X max :' + str(maxRealtiveX))
    print('Relative Y max :' + str(maxRealtiveY))
    print('Relative X min :' + str(minRealtiveX))
    print('Relative Y min :' + str(minRealtiveY))
    
    print('VelocityX min :' + str(minVelocityX))
    print('VelocityX max :' + str(maxVelocityX))
    print('VelocityY min :' + str(minVelocityY))
    print('VelocityY max :' + str(maxVelocityY))


    ######################################################################
    ###############   Loop based training   ##############################
    ######################################################################

    for eachLoop in range(numberOfTrainingLoop):


        # Re-Load the Vehicle and Frame based Dictionaries to use during the Intermediate prediction based on Train or Test
        # Clean up the dictionaries first 
        dictByFrames = dict()
        dictByVehicles = dict()
        mapperDict = dict()
        # # # global dictByFrames, dictByVehicles, validationVehicles, mapperDict
        # Populate the dictionaries
        dictByFrames,dictByVehicles, mapperDict = CreateVehicleAndFrameDict(trainTrajFilePath)    # testTrajFilePath   trainTrajFilePath
        finalVehicleKeys = list(dictByVehicles.keys())
        finalVehicleKeys.sort(key=float)
        finalFrameKeys = list(dictByFrames.keys())
        finalFrameKeys.sort(key=float)

        # Prepeare the intermediate arrays
        data_process = mp.Process(target=IntermediatePredictionProcess, args = (trainingFileName,trainStr))   # validationFileName  trainingFileName
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
        YVelTrain = YVelTrain.reshape(currntShape[0],currntShape[1],2)

        # No need to split for the old architecture
        # Split the velocity array into Vx and Vy arrays for seperate fitting with two logcosh
        # # # velocityVxTrain, velocityVyTrain = np.split(YVelTrain, 2, -1)

        # Clean up the trainNormalList list
        trainProcessList[:] = []
        trainNormalList[:] = []

        # Both the decoder and normal input should be normalized and no values should be >1
        print('XTrain max :' + str(np.amax(XTrain)))
        print('decoderTrainInput max :' + str(np.amax(decoderTrainInput)))

        # Check all the array to see if the max value not exceds 1.0
        if(np.amax(XTrain)>1.1 or np.amax(decoderTrainInput)>1.1):
            print('One of the above array in not normalized properly!!!!')
            nanVal = 1000/0
            sys.exit()


        # Re-Load the Vehicle and Frame based Dictionaries to use during the Intermediate prediction based on Train or Test
        # Clean up the dictionaries first 
        dictByFrames = dict()
        dictByVehicles = dict()
        mapperDict = dict()
        # # # global dictByFrames, dictByVehicles, validationVehicles, mapperDict
        # Populate the dictionaries
        dictByFrames,dictByVehicles, mapperDict = CreateVehicleAndFrameDict(testTrajFilePath)
        finalVehicleKeys = list(dictByVehicles.keys())
        finalVehicleKeys.sort(key=float)
        finalFrameKeys = list(dictByFrames.keys())
        finalFrameKeys.sort(key=float)

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
        print('Finished Xval Array!!!')
        decoderValInput = np.array([x[1] for x in valNormalList])
        print('Finished decoderValInput Array!!!')
        YClassVal = np.array([x[2] for x in valNormalList])
        print('Finished YClassVal Array!!!')
        YVelVal = np.array([x[3] for x in valNormalList])
        print('Finished YVelVal Array!!!')
        YPoseVal = np.array([x[4] for x in valNormalList])
        print('Finished YPoseVal Array!!!')

        # Reshape the velocity train array
        currntShape = YVelVal.shape
        YVelVal = YVelVal.reshape(currntShape[0],currntShape[1],2)

        # No need to split for the old architecture
        # Split the velocity array into Vx and Vy arrays for seperate fitting with two logcosh
        velocityVxVal, velocityVyVal = np.split(YVelVal, 2, -1)

        print('Second rouund data preperation fininshed!!!')

        # Clean up the validationProcessList list
        validationProcessList[:] = []
        valNormalList[:] = []

        # Both the decoder and normal input should be normalized and no values should be >1
        print('XVal max :' + str(np.amax(XVal)))
        print('decoderValInput max :' + str(np.amax(decoderValInput)))

        # Check all the array to see if the max value not exceds 1.0
        if(np.amax(XVal)>1.1 or np.amax(decoderValInput)>1.1):
            print('One of the above array in not normalized properly!!!!')
            nanVal = 1000/0
            sys.exit()


        # Loop training process (last param True, to load the weights)
        # This is for the new motion based arch with seperate Vx and Vy fitting....
        # # training_process = mp.Process(target=_training_worker, args = (XTrain,decoderTrainInput,YClassTrain,velocityVxTrain,velocityVyTrain,YPoseTrain,secondNumberEpochs,XVal,decoderValInput,YClassVal,velocityVxVal,velocityVyVal,YPoseVal))
        # This is for the old raw arch with single array for Vx and Vy
        training_process = mp.Process(target=_training_worker, args = (XTrain,decoderTrainInput,YClassTrain,YVelTrain,YPoseTrain,secondNumberEpochs,XVal,decoderValInput,YClassVal,YVelVal,YPoseVal))
        training_process.start()
        training_process.join()

        print('Waiting for memeory clear!!!')
        sleep(1)

        # clean up the intermediate arrays after the training
        XTrain = []
        decoderTrainInput = []
        YClassTrain = []
        YVelTrain = []
        YPoseTrain = []
        XVal = []
        decoderValInput = []
        YClassVal = []
        YVelVal = []
        YPoseVal = []


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