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
#from keras.models import Sequential
#from keras.models import Sequential
from keras.models import Model
#from keras.utils import Sequence
from keras.layers import  Input,LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dropout, MaxPooling1D, concatenate, division, subtract, Lambda
import matplotlib.pyplot as plt
from keras import optimizers
from keras.optimizers import RMSprop, Adam
from keras.utils import multi_gpu_model
import random
from keras import backend as K
from keras.callbacks import LearningRateScheduler
from keras import callbacks
from keras.losses import logcosh
import tensorflow as tf

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.67)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


# Specify the test trajectory csv file
testTrajFilePath = '/home/saptarshi/PythonCode/Junction/data/junc.csv'
# testTrajFilePath = '/media/disk1/sap/Junction/data/Lankershim.csv'

# Specify the result file to store each sample error
resultFileName = 'EncoderV6.txt'
f = open(resultFileName, 'x')

# Create the visible window
# cv2.namedWindow('test', cv2.WINDOW_NORMAL)


# Model parametrs 
historyTemporal = 5
futureTemporal = 10
validationVehicleCount = 1
surroudingCarCounts = 4
#validationFileName = 'EncoderDecoderValtest.txt'

# Min Max values for normalize or denormalize
minLocalY = 0
maxLocalY = 0
minLocalX = 0
maxLocalX = 0
minVel = 0
maxVel = 0

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
directionIndex = 18
movementIndex = 19
intersectionIndex = 16 #####################
sectionIndex = 17 #####################

# String Constants 
inputStr = 'Input'
decoderStr = 'Decoder'

# Unit constants
feetToMeter = 0.3048

# Make the frame dictionary global for use during prediction
dictByFrames = dict()

# Make the mapper dict global for original ID retrival during test
mapper = dict()

# Define the junction location distances
juncLocDict = {
  "1.0": 65,
  "2.0": 430,
  "3.0": 1068,
  "4.0": 1560
} #####################


# Custome Loss function
def euclidean_distance_loss(y_true, y_pred):
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

    global mapper

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
    normalizeIndexList = [velocityIndex,laneIDIndex,directionIndex,movementIndex]
    # normalizeIndexList = [localXIndex,velocityIndex,laneIDIndex,directionIndex,movementIndex,localYIndex]

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

    # Create Dictionary with unique Frames
    uniquFrameIds = list(np.unique(datasetArray[:,3]))
    frameKeys = []
    for idx in range(0, len(uniquFrameIds)):
        frameKeys.append(str(uniquFrameIds[idx]))

    dictionaryByFrames = {key : list() for key in frameKeys}

    for jdx in range(0,len(datasetArray)):
        key = str(datasetArray[jdx,3])
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


# Pass the surrounding vechiles and current input list. It will extend the list with surrouding cars info.
def GetSurroundingCarsInfo(otherVechiles, tempInput, targetVehicleID, inputOrDecoder, localX, localY):
    otherVechilesCount = len(otherVechiles)
    paddingCount = surroudingCarCounts + 1 - otherVechilesCount 

    # If other vehicle count is less than 5 (4 surronding + 1 target as it will present in the frame based list)
    # append all the vechiles info into input list. 
    if (otherVechilesCount <= (surroudingCarCounts+1)):
        # Process the gathered surrounding cars
        for eachOtherVechiles in otherVechiles:
            otherVehicleID = eachOtherVechiles[vechileIDIndex]


            # Ignore the target vechile as it is already added
            if(otherVehicleID == targetVehicleID):
                continue

            otherLocalX = eachOtherVechiles[localXIndex]
            otherLocalY = eachOtherVechiles[localYIndex]
            otherVelocity = eachOtherVechiles[velocityIndex]
            otherLaneID = eachOtherVechiles[laneIDIndex]
            otherDirection = eachOtherVechiles[directionIndex]
            otherMovement = eachOtherVechiles[movementIndex]
            if(inputOrDecoder == inputStr):
                tempInput.extend([otherLocalX,otherLocalY,otherVelocity,otherLaneID,otherDirection,otherMovement])
            elif(inputOrDecoder == decoderStr):
                lastInputClassInfo = MovementToClassForm(otherMovement)
                tempInput.extend([otherLocalX,otherLocalY,otherVelocity,lastInputClassInfo[0],lastInputClassInfo[1],lastInputClassInfo[2]])
            else:
                print('Unknow inputOrDecoder string : ' +  inputOrDecoder)
                sys.exit()
        
        # Append the zero padding based on the required padding count calculated using other vechile count and decided surrouning car count
        # As the number of input features and decoder input/output is same that is why we used the same zero padding width
        zeroList = [0,0,0,0,0,0]

        for rdx in range(0,paddingCount):          
            tempInput.extend(zeroList)

    # Else the vechile count is more than 4. So select the nearest 4 vechicles.
    else:
        # Gather distance of each car from the target car
        otherCarIndexedDistanceList = []
        for sdx, eachOtherVechiles in enumerate(otherVechiles):
            otherVehicleID = eachOtherVechiles[vechileIDIndex]
            # Ignore the target vechile as the distance would be zero
            if(otherVehicleID == targetVehicleID):
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
            if(inputOrDecoder == inputStr):
                tempInput.extend([otherLocalX,otherLocalY,otherVelocity,otherLaneID,otherDirection,otherMovement])
            elif(inputOrDecoder == decoderStr):
                lastInputClassInfo = MovementToClassForm(otherMovement)
                tempInput.extend([otherLocalX,otherLocalY,otherVelocity,lastInputClassInfo[0],lastInputClassInfo[1],lastInputClassInfo[2]])

    return tempInput


# Calculate the Y location of the nearest junction
def CalculateNearestJuncLoc(sectionID, intersectionID, poseX, poseY):

    # Initialize Y location and dist
    yLoc = -1
    juncDist = -1
    # Check if the vehicle is at intersection if yes send the Y location of that intersection
    if(intersectionID!=0):
        yLoc = juncLocDict[str(intersectionID)]
        juncDist = abs(poseY-yLoc)
        return juncDist
    # if the vehicle is not at intersection identify which side of the road ans which section the car is
    # PoseX is negetive for vehicles on the left side
    if(poseX<0):
        if (sectionID <= 0):
            print(sectionID)
            sys.exit()
        nearestIntersection = str(sectionID-1)

        yLoc = juncLocDict[nearestIntersection]
        juncDist = abs(poseY-yLoc)
        return juncDist
    
    # PoseX is positive for vehicles on the left side
    if(poseX>=0):
        nearestIntersection = str(sectionID)
        yLoc = juncLocDict[nearestIntersection]
        juncDist = abs(poseY-yLoc)
        return juncDist

    if(yLoc == -1):
        print('Junction distance calculation is not perfromed properly...')
        sys.exit()

# Plot all cars trajectory on the global GPS map
def TrainData(inputFileName):

    # Load the Vehicle and Frame based Dictionaries
    global dictByFrames
    dictByFrames,dictByVehicles = CreateVehicleAndFrameDict(inputFileName)
    finalVehicleKeys = list(dictByVehicles.keys())
    finalVehicleKeys.sort(key=float)
    finalFrameKeys = list(dictByFrames.keys())
    finalFrameKeys.sort(key=float)

    # Randomly sample 300 validation vehicles and write to file for testing
    validationVehicles = random.sample(finalVehicleKeys,validationVehicleCount)

    # As training and testing in same file no need to write validation vehicles in a file
    # with open(validationFileName, 'w') as f:
    #     for item in validationVehicles:
    #         f.write("%s\n" % item)

    # Train final lists
    finalXTrain = []
    finalYClassTrain = []
    finalYVelTrain = []
    finalYPoseTrain = []
    finalTrainDecoderInput = []

    # Validation final lists
    finalXVal = []
    finalYClassVal = []
    finalYVelVal = []
    finalYPoseVal = []
    finalValDecoderInput = []

    for currentVehicle in finalVehicleKeys:
        currentVehicleList = dictByVehicles[str(currentVehicle)]
        # Add the check for the side origins and side destination
        sideOrigin = currentVehicleList[0][14]
        sideDestination = currentVehicleList[0][15]
        if ((sideOrigin == 111 and sideDestination == 203) or (sideOrigin == 103 and sideDestination == 211) or (sideOrigin == 105 and sideDestination == 210) or (sideOrigin == 110 and sideDestination == 205) or (sideOrigin == 107 and sideDestination == 209) or (sideOrigin == 109 and sideDestination == 207)):
            continue

        print('Processing Vehicle : ' + str(currentVehicle))
        currentVehicleLength = len(currentVehicleList)
        targetVehicleID = currentVehicleList[0][vechileIDIndex]


        for idx in range(historyTemporal,currentVehicleLength-futureTemporal):

#############################################################################
            # Get the current vehicles as those are only eligible from prediction point of view 
            # vehicles appearing in first frame of the target vehicle
            currentTargetTime = currentVehicleList[idx-historyTemporal][globalTimeIndex]
            currentOtherVehicles = dictByFrames[str(currentTargetTime)]
            currentOtherEligibleVehicles = []
            for eachCurrentOtherVehicles in currentOtherVehicles:
                currentOtherID = eachCurrentOtherVehicles[vechileIDIndex]
                # vehicles having history + future temporal frames.
                currentOtherFrame = eachCurrentOtherVehicles[frameIDIndex]
                currentOtherTotalFrame = eachCurrentOtherVehicles[totoalFrameIndex]
                remainingFrames = currentOtherTotalFrame - currentOtherFrame
                if(remainingFrames>= historyTemporal+futureTemporal):
                    currentOtherEligibleVehicles.append(currentOtherID)


            # Prepeare sequential Input Data
            localXData = []
            for jdx in range(idx-historyTemporal,idx):
                tempInput = []
                localX = currentVehicleList[jdx][localXIndex]
                localY = currentVehicleList[jdx][localYIndex]
                velocity = currentVehicleList[jdx][velocityIndex]
                laneID = currentVehicleList[jdx][laneIDIndex]
                direction = currentVehicleList[jdx][directionIndex]
                movement = currentVehicleList[jdx][movementIndex]

                # Nearest junction distance
                currentSection = currentVehicleList[jdx][sectionIndex]
                currentIntersection = currentVehicleList[jdx][intersectionIndex]
                juncDist = CalculateNearestJuncLoc(currentSection, currentIntersection, localX, localY)

                tempInput = [localX,localY,velocity,laneID,direction,movement]

                # Prepare the surrounding cars information
                # Gather vehicles using the same frame using the Frame Dict
                currentInputFrame = currentVehicleList[jdx][frameIDIndex]
                currentInputTime = currentVehicleList[jdx][globalTimeIndex]
                otherVechiles = dictByFrames[str(currentInputTime)]
                # Remove vehicles with a different global time to remve duplicate vehicles
                for fdx,eachOtherTime in enumerate(otherVechiles):
                    otherTime = eachOtherTime[globalTimeIndex]
                    if (otherTime != currentInputTime):
                        otherVechiles.pop(fdx)

##################################################################################
                # Remove the prediction not eligible vehicles
                # Identify vehicles not in eligible list
                removeIndexList = []
                for removeIndex,eachOtherVehicle in enumerate(otherVechiles):
                    otherID = eachOtherVehicle[vechileIDIndex]
                    if not (otherID in currentOtherEligibleVehicles):
                        removeIndexList.append(removeIndex)

                # Remove the identified item from the other vehicle based on the removal list (perfrom the pop from last to not affect the index)
                for removeIndexRev in reversed(removeIndexList):
                    otherVechiles.pop(removeIndexRev)
##############################################################

                # Extend the surrounding cars info into the target vehicles input
                tempInput = GetSurroundingCarsInfo(otherVechiles, tempInput, targetVehicleID, inputStr, localX, localY)
                
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
            for tdx in range(0,len(lastInput),6):
                lastInputPoseX = lastInput[tdx]
                lastInputPoseY = lastInput[tdx+1]
                lastInputVelocity = lastInput[tdx+2]
                lastInputMovement = lastInput[tdx+5]
                lastInputClassInfo = MovementToClassForm(lastInputMovement)
                firstDecoderInput.extend([lastInputPoseX,lastInputPoseY,lastInputVelocity,lastInputClassInfo[0],lastInputClassInfo[1],lastInputClassInfo[2]])


            for kdx in range(idx,idx+futureTemporal):

                # Add the ground truth outputs
                nextMovement = currentVehicleList[kdx][movementIndex]
                nextMovementClassData = MovementToClassForm(nextMovement)

                localYMovementData.append(nextMovementClassData)
                
                nextVelocity = currentVehicleList[kdx][velocityIndex]
                deNormalizedNextVelocity = (nextVelocity*(maxVel-minVel))+minVel

                nextLocalX = currentVehicleList[kdx][localXIndex]
                denormalizedNextLocalX = (nextLocalX*(maxLocalX-minLocalX)+minLocalX)
                nextLocalY = currentVehicleList[kdx][localYIndex]
                denormalizedNextLocalY = (nextLocalY*(maxLocalY-minLocalY)+minLocalY)

                localYVelData.append([deNormalizedNextVelocity])
                localYPoseData.append([denormalizedNextLocalX,denormalizedNextLocalY])

                # Add the decoder input
                decoderTemp = [nextLocalX,nextLocalY,nextVelocity,nextMovementClassData[0],nextMovementClassData[1],nextMovementClassData[2]]

                # Prepare the surrounding cars information for decoder input   # for decoder pass only the vehicles present in the last 30 frames..(not done....)
                # Gather vehicles using the same frame using the Frame Dict
                currentInputFrame = currentVehicleList[kdx][frameIDIndex]
                currentInputTime = currentVehicleList[kdx][globalTimeIndex]
                otherVechiles = dictByFrames[str(currentInputTime)]

                ##################################################################################
                # Remove the prediction not eligible vehicles
                # Identify vehicles not in eligible list
                removeIndexList = []
                for removeIndex,eachOtherVehicle in enumerate(otherVechiles):
                    otherID = eachOtherVehicle[vechileIDIndex]
                    if not (otherID in currentOtherEligibleVehicles):
                        removeIndexList.append(removeIndex)

                # Remove the identified item from the other vehicle based on the removal list (perfrom the pop from last to not affect the index)
                for removeIndexRev in reversed(removeIndexList):
                    otherVechiles.pop(removeIndexRev)
##############################################################


                # Remove vehicles with a different global time to remve duplicate vehicles
                for gdx,eachOtherTime in enumerate(otherVechiles):
                    otherTime = eachOtherTime[globalTimeIndex]
                    if (otherTime != currentInputTime):
                        otherVechiles.pop(gdx)

                # Extend the surrounding cars info into the target vehicles decoder input
                decoderTemp = GetSurroundingCarsInfo(otherVechiles, decoderTemp, targetVehicleID, decoderStr, nextLocalX, nextLocalY)

                # Finally append the target car and surrounding cars info for the current frame into the final decoded input
                decoderInputData.append(decoderTemp)


            # Append in the final validation or training set based of decided vehicle ID
            if(currentVehicle in validationVehicles):
                finalXVal.append(localXData)
                finalYClassVal.append(localYMovementData)
                finalYVelVal.append(localYVelData)
                finalYPoseVal.append(localYPoseData)
                # Shift one time stamp right and append Last input at the beggining 
                decoderInputData = decoderInputData[:-1]
                decoderInputData.insert(0,firstDecoderInput)
                finalValDecoderInput.append(decoderInputData)
            else:
                finalXTrain.append(localXData)
                finalYClassTrain.append(localYMovementData)
                finalYPoseTrain.append(localYPoseData)
                finalYVelTrain.append(localYVelData)
                # Shift one time stamp right and append last input at the beggining 
                decoderInputData = decoderInputData[:-1]
                decoderInputData.insert(0,firstDecoderInput)
                finalTrainDecoderInput.append(decoderInputData)


    # Prepare the final Train arrays
    finalXTrainArray = np.array(finalXTrain)
    finalYClassTrainArray = np.array(finalYClassTrain)
    finalYPoseTrainArray = np.array(finalYPoseTrain)
    finalYVelTrainArray = np.array(finalYVelTrain)
    finalTrainDecoderInputArray = np.array(finalTrainDecoderInput)

    # Prepare the final Validation arrays
    finalXValArray = np.array(finalXVal)
    finalYClassValArray = np.array(finalYClassVal)
    finalYVelValArray = np.array(finalYVelVal)
    finalYPoseValArray = np.array(finalYPoseVal)
    finalValDecoderInputArray = np.array(finalValDecoderInput)

    return finalXTrainArray,finalTrainDecoderInputArray,finalYClassTrainArray,finalYVelTrainArray,finalYPoseTrainArray,finalXValArray,finalValDecoderInputArray,finalYClassValArray,finalYVelValArray,finalYPoseValArray

# Define the Custome learing rate decays
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.8
    epochs_drop = 1.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    if lrate < 0.00001:
        lrate = 0.00001
    return lrate

# class LossHistory(callbacks.Callback):

#     def on_train_begin(self, logs={}):
#         self.losses = []
#         #self.vallosses = []
#         self.lr = []
 
#     def on_epoch_end(self, batch, logs={}):
#         self.losses.append(logs.get('loss'))
#         #self.vallosses.append(logs.get('val_loss'))
#         self.lr.append(step_decay(len(self.losses)))
#         #print('\n Current lr = ' + str(self.lr[-1]))
#         #print('\n Current Val Loss = ' + str(self.vallosses[-1]))

class EarlyStoppingByLossVal(callbacks.Callback):
    def __init__(self, monitor='val_loss', value=5, verbose=0):
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

# Class to hold all the relevet vehicle ID specific predicition intermediate information
class PredictionInfos():
    def __init__(self, input = [], decoderInput = [], state = [], output=[], groundTruth = [], initTime=0.0):
        self.input = input
        self.decoderInput = decoderInput
        self.state = state
        self.output = output
        self.groundTruth = groundTruth
        self.initTime = initTime


if __name__ == '__main__':

    XTrain,decoderTrainInput,YClassTrain,YVelTrain,YPoseTrain,XVal,decoderValInput,YClassVal,YVelVal,YPoseVal= TrainData(testTrajFilePath)
    sampleCount = XTrain.shape[0]
    temporal = XTrain.shape[1]
    features = XTrain.shape[2]
    outputFeatures = 6
    decoderFeatures = (outputFeatures*(surroudingCarCounts+1))
    classOut = 3
    poseOut = 2
    velcoityOut = 1
    n_units = 256


    #inp = Input((temporal,features))


    # define training encoder
    encoder_inputs = Input(shape=(None, features))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
	# define training decoder
    decoder_inputs = Input(shape=(None, decoderFeatures))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    # Decoder for ClassOut
    decoder_dense11 = Dense(256, activation='relu')
    decoder_output1 = decoder_dense11(decoder_outputs)
    decoder_dense12 = Dense(128, activation='relu')
    decoder_output1 = decoder_dense12(decoder_output1)
    decoder_dense13 = Dense(64, activation='relu')
    decoder_output1 = decoder_dense13(decoder_output1)
    decoder_dense14 = Dense(32, activation='relu')
    decoder_output1 = decoder_dense14(decoder_output1)
    decoder_dense15 = Dense(3, activation='softmax', name='Class')
    decoder_output1 = decoder_dense15(decoder_output1)

    # Decoder for Velocity Out
    decoder_dense21 = Dense(256, activation='relu')
    decoder_output2 = decoder_dense21(decoder_outputs)
    decoder_dense22 = Dense(128, activation='relu')
    decoder_output2 = decoder_dense22(decoder_output2)
    decoder_dense23 = Dense(64, activation='relu')
    decoder_output2 = decoder_dense23(decoder_output2)
    decoder_dense24 = Dense(32, activation='relu')
    decoder_output2 = decoder_dense24(decoder_output2)
    decoder_dense25 = Dense(1, activation='linear', name='Velcoity')
    decoder_output2 = decoder_dense25(decoder_output2)

    # Decoder for position out
    decoder_dense31 = Dense(256, activation='relu')
    decoder_output3 = decoder_dense31(decoder_outputs)
    decoder_dense32 = Dense(128, activation='relu')
    decoder_output3 = decoder_dense32(decoder_output3)
    decoder_dense33 = Dense(64, activation='relu')
    decoder_output3 = decoder_dense33(decoder_output3)
    decoder_dense34 = Dense(32, activation='relu')
    decoder_output3 = decoder_dense34(decoder_output3)
    decoder_dense35 = Dense(2, activation='linear', name='Position')
    decoder_output3 = decoder_dense35(decoder_output3)
    
    model = Model([encoder_inputs, decoder_inputs], [decoder_output1, decoder_output2, decoder_output3])

	# define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)

	# define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    # Inference decoder for Class out
    decoder_output1 = decoder_dense11(decoder_outputs)
    decoder_output1 = decoder_dense12(decoder_output1)
    decoder_output1 = decoder_dense13(decoder_output1)
    decoder_output1 = decoder_dense14(decoder_output1)
    decoder_output1 = decoder_dense15(decoder_output1)

    # Inference Decoder for Velocity Out
    decoder_output2 = decoder_dense21(decoder_outputs)
    decoder_output2 = decoder_dense22(decoder_output2)
    decoder_output2 = decoder_dense23(decoder_output2)
    decoder_output2 = decoder_dense24(decoder_output2)
    decoder_output2 = decoder_dense25(decoder_output2)

    #Inference  Decoder for position out
    decoder_output3 = decoder_dense31(decoder_outputs)
    decoder_output3 = decoder_dense32(decoder_output3)
    decoder_output3 = decoder_dense33(decoder_output3)
    decoder_output3 = decoder_dense34(decoder_output3)
    decoder_output3 = decoder_dense35(decoder_output3)

    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_output1, decoder_output2, decoder_output3] + decoder_states)

    # Custom decay rates
    # loss_history = LossHistory()
    # lrate = LearningRateScheduler(step_decay)
    esObj = EarlyStoppingByLossVal()
    callbacks_list = [esObj]   #[loss_history, lrate]
    opt = RMSprop()

    model.compile(optimizer=opt, loss=['categorical_crossentropy', logcosh, euclidean_distance_loss])
    model.summary()

    print('XTrain Shape : ' + str(XTrain.shape))
    print('DecoderInput Shape : ' + str(decoderTrainInput.shape))

    model.fit([XTrain,decoderTrainInput], [YClassTrain,YVelTrain,YPoseTrain], batch_size=128, epochs=2, verbose=1, validation_data=([XVal,decoderValInput],[YClassVal,YVelVal,YPoseVal]), callbacks=callbacks_list)


    # Relese the arrays to save memory consumption
    del XTrain,decoderTrainInput,YClassTrain,YVelTrain,YPoseTrain

    # Test the model with the test dataset

    # Intialize the frame based distance error array with sample count as 0
    finalError = np.zeros(futureTemporal)
    count = 0


    # Prepare the trakcer Dict
    trackerDict = dict()

    # Sort the frame keys for proper sequential prediction
    allFrameKeys = sorted(dictByFrames.keys(), key=float)

    for key in allFrameKeys:
        currentVehicles = dictByFrames[key]
        carsInCurrentFrame = []
        for eachCurrentVehicle in currentVehicles:
            currentVechicleID = eachCurrentVehicle[vechileIDIndex]
            currentLocalX = eachCurrentVehicle[localXIndex]
            currentLocalY = eachCurrentVehicle[localYIndex]
            currentVelocity = eachCurrentVehicle[velocityIndex]
            currentLaneID = eachCurrentVehicle[laneIDIndex]
            currentDirection = eachCurrentVehicle[directionIndex]
            currentMovement = eachCurrentVehicle[movementIndex]
            currentTime = eachCurrentVehicle[globalTimeIndex]
            currentFrame = eachCurrentVehicle[frameIDIndex]
            dictInput = [currentLocalX,currentLocalY,currentVelocity,currentLaneID,currentDirection,currentMovement,currentTime,currentFrame]
            # Vehicle Birth in tracker Dict
            if (str(currentVechicleID) not in trackerDict):
                trackerDict[str(currentVechicleID)] = []
                trackerDict[str(currentVechicleID)].append(dictInput)
            else:
                trackerDict[str(currentVechicleID)].append(dictInput)

            # Append the current vehicle IDs to calculate car deaths
            carsInCurrentFrame.append(str(currentVechicleID))

        existingCars = list(trackerDict.keys())
        deathCars = list(set(existingCars) - set(carsInCurrentFrame))
        #delete the cars which are death cars from the trakcer
        for death in deathCars:
            del trackerDict[death]

        # Identify prediction eligible vehicles having 30 frames history
        eligibleVehicleKeys = []
        for trackerKey in trackerDict.keys():
            vehicleFrameLength = len(trackerDict[trackerKey])
            if(vehicleFrameLength == (historyTemporal+futureTemporal)):
                eligibleVehicleKeys.append(trackerKey)
            elif(vehicleFrameLength > (historyTemporal+futureTemporal)):
                print('Tracker Over populated for vehicle: ' + trackerKey)
                print('This is unwanted event....')
            else:
                print('Tracker Under populated for vehicle: ' + trackerKey)
        
        # If no Eligible vehicles move to next frame
        if not eligibleVehicleKeys:
            continue

        # Prepare a dictionary to hold all the prediction relevent information (input, decoderInput, state, predictedOutput and GroundTurthOutput) against each vehicle
        predictionDict = dict()

        # Populate all the input/gournd truth infos in the prediction dictionary
        for eachEligibleKey in eligibleVehicleKeys:
            predictionInfoObj = PredictionInfos()
            predictionDict[eachEligibleKey] = predictionInfoObj
            # Get all the input infos from the traker dict for that specific vehicle
            inputInfo = trackerDict[eachEligibleKey][0:historyTemporal]
            # Get the initial time to remove duplicate vehicles and update the coresponding object
            intialTime = inputInfo[0][6] # 6 is time index in trakcer dict list
            predictionDict[eachEligibleKey].initTime = intialTime
            predicitionInputList = []
            for udx, eachInputInfo in enumerate(inputInfo):
                tempPredictionInput = eachInputInfo.copy()[:-2]
                targetLocalX = eachInputInfo[0]  # 0 is poseX index in trakcer dict list
                targetLocalY = eachInputInfo[1]  # 1 is poseY index in trakcer dict list

                # Get the surrounding cars for the same frame
                # Get the surrounding car IDs by getting all keys and removing the current key
                surroudingCarIds = eligibleVehicleKeys[:]
                surroudingCarIds.remove(eachEligibleKey) ##### add the time check here
                for hdx,eachTimeEligibleKey in enumerate(surroudingCarIds):
                    otherTime = trackerDict[eachTimeEligibleKey][0][6]
                    if(otherTime != intialTime):
                        surroudingCarIds.pop(hdx)

                # If the surrouding car count is less than or equal to the decided surrouding car count then append all the poses and manage by zero padding
                predSurroudingCarCount = len(surroudingCarIds)
                predictionPaddingCount = surroudingCarCounts - predSurroudingCarCount
                if(predSurroudingCarCount <= surroudingCarCounts):
                    for eachSurroundingCarID in surroudingCarIds:
                        surroundingCarLocalX = trackerDict[eachSurroundingCarID][udx][0]  # udx for coresponnding Frame and 0 is poseX index in TrakcerDict list
                        surroundingCarLocalY = trackerDict[eachSurroundingCarID][udx][1]  # udx for coresponnding Frame and 1 is poseY index in TrakcerDict list
                        surroundingCarVelocity = trackerDict[eachSurroundingCarID][udx][2]  # udx for coresponnding Frame and 2 is poseY index in TrakcerDict list
                        surroundingCarLaneID = trackerDict[eachSurroundingCarID][udx][3]  # udx for coresponnding Frame and 3 is poseY index in TrakcerDict list
                        surroundingCarDirection = trackerDict[eachSurroundingCarID][udx][4]  # udx for coresponnding Frame and 4 is poseY index in TrakcerDict list
                        surroundingCarMovement = trackerDict[eachSurroundingCarID][udx][5]  # udx for coresponnding Frame and 5 is poseY index in TrakcerDict list
                        tempPredictionInput.extend([surroundingCarLocalX,surroundingCarLocalY,surroundingCarVelocity,surroundingCarLaneID,surroundingCarDirection,surroundingCarMovement])
                    
                    predZeroPadList = [0,0,0,0,0,0]
                    for adx in range(0,predictionPaddingCount):
                        tempPredictionInput.extend(predZeroPadList)

                # Else the surrouding car count is more than the decided surrouding car count then select the nearest 4 cars
                else:
                    # Get the surrounding car's coresponding frame position and calculate distance
                    surroundingCarDistanceList = []
                    for eachSurroundingCarID in surroudingCarIds:
                        surroundingCarLocalX = trackerDict[eachSurroundingCarID][udx][0]  # udx for coresponnding Frame and 0 is poseX index in dict list
                        surroundingCarLocalY = trackerDict[eachSurroundingCarID][udx][1]  # udx for coresponnding Frame and 1 is poseY index in dict list
                        surroundingDist =  math.sqrt(((surroundingCarLocalX-targetLocalX)**2) + ((surroundingCarLocalY-targetLocalY)**2))
                        surroundingCarDistanceList.append([eachSurroundingCarID,surroundingDist])
                    
                    # Sort the list based on distance and gather the lowest distance car IDs
                    surroundingCarDistanceList = sorted(surroundingCarDistanceList,key=lambda x: x[1])
                    surroundingCarDistanceArray = np.array(surroundingCarDistanceList)
                    releventSurroundingIds = surroundingCarDistanceArray[0:surroudingCarCounts,0:1]

                    # Add the relevent input of nearest cars to temp list
                    for eachReleventSurroundingID in releventSurroundingIds:
                        surroundingCarLocalX = trackerDict[eachReleventSurroundingID][udx][0]  # udx for coresponnding Frame and 0 is poseX index in TrakcerDict list
                        surroundingCarLocalY = trackerDict[eachReleventSurroundingID][udx][1]  # udx for coresponnding Frame and 1 is poseY index in TrakcerDict list
                        surroundingCarVelocity = trackerDict[eachReleventSurroundingID][udx][2]  # udx for coresponnding Frame and 2 is poseY index in TrakcerDict list
                        surroundingCarLaneID = trackerDict[eachReleventSurroundingID][udx][3]  # udx for coresponnding Frame and 3 is poseY index in TrakcerDict list
                        surroundingCarDirection = trackerDict[eachReleventSurroundingID][udx][4]  # udx for coresponnding Frame and 4 is poseY index in TrakcerDict list
                        surroundingCarMovement = trackerDict[eachReleventSurroundingID][udx][5]  # udx for coresponnding Frame and 5 is poseY index in TrakcerDict list
                        tempPredictionInput.extend([surroundingCarLocalX,surroundingCarLocalY,surroundingCarVelocity,surroundingCarLaneID,surroundingCarDirection,surroundingCarMovement])
                
                # Add the current frame input info in the input list
                predicitionInputList.append(tempPredictionInput)

            # Add the current Vehicles all history frame input to the prediction dict object (input field)
            predictionDict[eachEligibleKey].input = predicitionInputList

        # Add the ground truth output pose in to the prediction dict object for error calculation
        outputInfo = trackerDict[eachEligibleKey][historyTemporal:historyTemporal+futureTemporal]
        tempGroundTruthPoseList = []
        for eachOutputInfo in outputInfo:
            groundTruthPoseX = eachOutputInfo[0]  # 0 is poseX index in trakcer dict list
            groundTruthPoseY = eachOutputInfo[1]  # 1 is poseY index in trakcer dict list
            tempGroundTruthPoseList.append([groundTruthPoseX,groundTruthPoseY])
        predictionDict[eachEligibleKey].groundTruth = tempGroundTruthPoseList
        
        # Add the decoder inputs in the prediction dict against each vehicle
        # Predict the encoder state for each vehicle and update the prediction dict state values
        for eachPredDictKey in predictionDict.keys():
            lastInput = predictionDict[eachPredDictKey].input[-1]
            predDecoderInput = []
            for bdx in range(0,len(lastInput),6):  # 6 is number of input features
                lastInputPoseX = lastInput[bdx]
                lastInputPoseY = lastInput[bdx+1]
                lastInputVelocity = lastInput[bdx+2]
                lastInputMovement = lastInput[bdx+5]
                lastInputClassInfo = MovementToClassForm(lastInputMovement)
                predDecoderInput.extend([lastInputPoseX,lastInputPoseY,lastInputVelocity,lastInputClassInfo[0],lastInputClassInfo[1],lastInputClassInfo[2]])
            # Add the prepered decoder input in the prediction dict object (decoder input field)
            predictionDict[eachPredDictKey].decoderInput = predDecoderInput

            # Get the input for the current vehicle to predict the encoder state
            currentPredInput = np.array(predictionDict[eachPredDictKey].input).reshape(1,historyTemporal,features)

            # Predict the Encoder state for that specific vehicle and update the prediction dict
            currentState = encoder_model.predict(currentPredInput)
            predictionDict[eachPredDictKey].state = currentState
        
        # Predict till the decided future temporal
        for cdx in range(futureTemporal):

            # Predict the next frame for each vechicle in the prediction dict
            for eachPredDictKey in predictionDict.keys():
                # Prepare the target seq and state for the current Vehicle
                target_seq = np.array(predictionDict[eachPredDictKey].decoderInput).reshape(1,1,decoderFeatures)
                predState = predictionDict[eachPredDictKey].state

                # Predict the next frame for the current vehicle
                classPred, velcoityPred, posePred, h, c = decoder_model.predict([target_seq] + predState)

                # Store the prediction in the prediction dict w.r.t the coresponding vehicle
                finalOutput = [posePred[0][0][0],posePred[0][0][1],velcoityPred,classPred[0][0][0],classPred[0][0][1],classPred[0][0][2]]
                predictionDict[eachPredDictKey].output.append(finalOutput)

                # Update the state for each vehicle in the prediction dict
                predictionDict[eachPredDictKey].state = [h, c]
            
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

                # Normalize poseX, poseY and velocity before adding to the decoder input
                normalizedPredPoseX = (lastOutputPoseX-minLocalX)/(maxLocalX-minLocalX)
                normalizedPredPoseY = (lastOutputPoseY-minLocalY)/(maxLocalY-minLocalY)
                normalizedPredVelocity = (lastOutputVelocity-minVel)/(maxVel-minVel)

                # Finally add the normalized values into the temp decoder input
                tempDecoderInput = [normalizedPredPoseX,normalizedPredPoseY,normalizedPredVelocity,lastClassOutput0,lastClassOutput1,lastClassOutput2]

                # Add the nearest vehicle's info into the decoder list
                # Get the surrounding car IDs by getting all keys and removing the current key/target vehicle
                allPredictionKeys = list(predictionDict.keys())
                decoderSurroudingCarIds = allPredictionKeys[:]
                decoderSurroudingCarIds.remove(eachPredDictKey)

                # Remove the vehicles with differnet intial time
                decodeIntialTime = predictionDict[eachPredDictKey].initTime
                for zdx,decodeSurrId in enumerate(decoderSurroudingCarIds):
                    decodeOtherInitialTime = predictionDict[decodeSurrId].initTime
                    if(decodeOtherInitialTime != decodeIntialTime):
                        decoderSurroudingCarIds.pop(zdx)

                
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
                        normalizedPredPoseX = (lastOutputPoseX-minLocalX)/(maxLocalX-minLocalX)
                        normalizedPredPoseY = (lastOutputPoseY-minLocalY)/(maxLocalY-minLocalY)
                        normalizedPredVelocity = (lastOutputVelocity-minVel)/(maxVel-minVel)

                        # Finally add the normalized values into the temp decoder input
                        tempDecoderInput.extend([normalizedPredPoseX,normalizedPredPoseY,normalizedPredVelocity,lastClassOutput0,lastClassOutput1,lastClassOutput2])
                    
                    predZeroPadList = [0,0,0,0,0,0]
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
                        surroundingDist =  math.sqrt(((lastSurroundingOutputPoseX-lastOutputPoseX)**2) + ((lastSurroundingOutputPoseY-lastOutputPoseY)**2))
                        decoderSurroundingCarDistanceList.append([eachDecoderSurroundingCarID,surroundingDist])
                    
                    # Sort the list based on distance and gather the lowest distance car IDs
                    decoderSurroundingCarDistanceList = sorted(decoderSurroundingCarDistanceList,key=lambda x: x[1])
                    decoderSurroundingCarDistanceArray = np.array(decoderSurroundingCarDistanceList)
                    decoderReleventSurroundingIds = decoderSurroundingCarDistanceArray[0:surroudingCarCounts,0:1]

                    # Add the relevent input of nearest cars to temp list
                    for eachDecoderReleventSurroundingID in decoderReleventSurroundingIds:
                        # Add the coresponding surrounding vechicle's last output in the decoder input (after normalization)
                        # Get the coresponding vehicles last output
                        lastSurroundingOutput = predictionDict[eachDecoderReleventSurroundingID].output[-1]
                        lastOutputPoseX = lastSurroundingOutput[0]   # 0 is poseX index in output list of prediction dict
                        lastOutputPoseY = lastSurroundingOutput[1]   # 1 is poseX index in output list of prediction dict
                        lastOutputVelocity = lastSurroundingOutput[2]   # 2 is velocity index in output list of prediction dict
                        lastClassOutput0 = lastSurroundingOutput[3]   # 3 is 0 class info index in output list of prediction dict
                        lastClassOutput1 = lastSurroundingOutput[4]   # 4 is 1 class info index in output list of prediction dict
                        lastClassOutput2 = lastSurroundingOutput[5]   # 5 is 2 class info index in output list of prediction dict

                        # Normalize poseX, poseY and velocity before adding to the decoder input
                        normalizedPredPoseX = (lastOutputPoseX-minLocalX)/(maxLocalX-minLocalX)
                        normalizedPredPoseY = (lastOutputPoseY-minLocalY)/(maxLocalY-minLocalY)
                        normalizedPredVelocity = (lastOutputVelocity-minVel)/(maxVel-minVel)

                        # Finally add the normalized values into the temp decoder input
                        tempDecoderInput.extend([normalizedPredPoseX,normalizedPredPoseY,normalizedPredVelocity,lastClassOutput0,lastClassOutput1,lastClassOutput2])
                
                # Finally update the decoder input in the prediction dict
                predictionDict[eachPredDictKey].decoderInput = tempDecoderInput


        # Calculate error for each Predicted poses in the prediction dict
        for eachErrorPredDictKey in predictionDict.keys():
            # Get the predicted and ground truth poses from the predict dict object
            groundTruthPose = predictionDict[eachErrorPredDictKey].groundTruth
            predictedPose = predictionDict[eachErrorPredDictKey].output

            # Length of both these lists should be equal
            if(len(groundTruthPose) != len(predictedPose)):
                print('Ground truth and predicted pose lists are not equal')
                sys.exit()
            localErrorList = []
            for edx,eachPose in enumerate(predictedPose):
                predX = predictedPose[edx][0] # 0 is poseX index in output list of prediction dict 
                predY = predictedPose[edx][1] # 1 is poseY index in output list of prediction dict
                trueX = groundTruthPose[edx][0] # 0 is poseX index in Ground Truth list of prediction dict 
                trueY = groundTruthPose[edx][1] # 1 is poseY index in Ground Truth list of prediction dict
                euclidianError = math.sqrt(((predX-trueX)**2) + ((predY-trueY)**2)) * feetToMeter
                localErrorList.append(euclidianError)
            
            # Keep count for average calculation and display average error
            count = count + 1
            finalError = finalError + np.array(localErrorList)

            # Print in the same line 
            printList = np.around(np.array([finalError[0],finalError[4],finalError[9],finalError[14],finalError[19],finalError[24],finalError[29],finalError[34],finalError[39],finalError[44],finalError[49]])/count, 2)
            print(*printList, end='\r', flush=True)

            # Write the individual error to text file.
            f.write("%s\n" % localErrorList)

        # Remove the first frame info from the tracker dict with vechiles having total history frame as the future trajectory for those vechiles are already predicted
        for eachRemovalKey in eligibleVehicleKeys:
            trackerDict[eachRemovalKey].pop(0)


    f.close()
    print('Final Distance Error')
    print(finalError/count)
    print('All the cars are predcited in the dataset.')
    sys.exit()