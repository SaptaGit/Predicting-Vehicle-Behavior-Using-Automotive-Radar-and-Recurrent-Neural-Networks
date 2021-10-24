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
#from keras.models import Sequential
#from keras.models import Sequential
from keras.models import Model
#from keras.utils import Sequence
from keras.layers import  Input,LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dropout, MaxPooling1D, concatenate, division, subtract, Lambda, BatchNormalization
import matplotlib.pyplot as plt
from keras import optimizers
from keras.optimizers import RMSprop, Adam
# from keras.utils import multi_gpu_model
import random
from keras import backend as K
from keras.callbacks import LearningRateScheduler
from keras import callbacks
from keras.losses import logcosh
import tensorflow as tf
import multiprocessing as mp
from multiprocessing import Process, Manager
import time
import multiprocessing
# multiprocessing.set_start_method('spawn', True)

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Specify the test trajectory csv file
# Path for local folder
# testTrajFilePath = '/home/saptarshi/PythonCode/Junction/data/junc.csv'
# Path for Sen Server trajectory csv file
# testTrajFilePath = '/media/disk1/sap/Junction/data/Lankershim.csv'
# Path for Big Screen Server trajectory csv file
# testTrajFilePath = '/home/sap/Sap/Junction/data/Lankershim.csv'
# Path for small Server trajectory csv file
testTrajFilePath = '/home/sap/Junction/data/Lankershim.csv'

# Specify if process the data or read the processed data
#  'read' -> Read Data  'process' -> Process data
readStr = 'read'
processStr = 'process'
processOrRead = processStr

# Specify the folder name for the sample to read/write based on the above flag
# Path for local folder
# folderName = '/home/saptarshi/PythonCode/Junction/Lankershim8Surrounding'
# Path for Sen Screen Server folder
# folderName ='/media/disk1/sap/Junction/Lankershim8Surrounding'
# Path for Big Screen Server folder
# folderName = '/home/sap/Sap/Junction/Lankershim1'
# Path for small Server folder
folderName = '/home/sap/Junction/Lankershim6Surrounding2'

# Specify the result file to store each sample error
resultFileName = 'data8Suurounding4.txt'
f = open(resultFileName, 'x')

# Specify the validation vehicle file name
validationFileName = folderName + '/' + 'validation.txt'

# Create the visible window
# cv2.namedWindow('test', cv2.WINDOW_NORMAL)

# Train and Validation process lists
manager = Manager()
trainProcessList = manager.list()
validationProcessList = manager.list()

# To keep count of number of sample processed
countList = manager.list()

# Model parametrs 
historyTemporal = 30   #30
futureTemporal = 50   #50
surroudingCarCounts = 6
globalInputFeatures = (surroudingCarCounts+1)*6   # 6 -> (poseX,poseY,velocity, LaneID, Movement, Direction)
globalOutputFeatures = 6                          # 6 -> (poseX,poseY,velocity, Class0, Class1, Class2)
globalDecoderFeatures = (surroudingCarCounts+1)*globalOutputFeatures 

# Validation vehiles
totalVehileCount = 1800
validationVehicleCount = 300

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
globalTimeIndex = 3
localXIndex = 4
localYIndex = 5
velocityIndex = 11
laneIDIndex = 13
originIndex = 14
destinationIndex = 15
directionIndex = 18
movementIndex = 19

# String Constants 
inputStr = 'Input'
decoderStr = 'Decoder'
trainStr = 'Train'
validationStr = 'Validation'

# Unit constants
feetToMeter = 0.3048

# Make the frame dictionary global for use during prediction
dictByFrames = dict()
# Make the Vehicle dictionary global for use multi processing
dictByVehicles = dict()

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

    print('Creating Vehicle and Frame based dictionary')

    #Create Dictionary for Mapper
    mapper = dict()

    loadFile = open(loadFileName, 'r')
    loadReader = csv.reader(loadFile)
    loadDataset = []
    for loadRow in loadReader:
        loadDataset.append(loadRow[0:24])

    loadDataset.pop(0)
    sortedList = sorted(loadDataset, key=lambda x: (float(x[0]), float(x[1])))
    datasetArray = np.array(sortedList, dtype=np.float)

    # Normalize each feature columns
    normalizeIndexList = [localXIndex,velocityIndex,laneIDIndex,directionIndex,movementIndex,localYIndex]

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

    loadFile.close()

    return dictionaryByFrames,dictionaryByVehicles,mapper


# Pass the surrounding vechiles and current input list. It will extend the list with surrouding cars info.
def GetSurroundingCarsInfo(otherVechiles, tempInput, targetVehicleID, inputOrDecoder, localX, localY):

    otherVechilesCount = len(otherVechiles)

    # Target Vehicle should Present in Other vehicles
    otherVehicleArray = np.array(otherVechiles)
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
            if(inputOrDecoder == inputStr):
                tempInput.extend([otherLocalX,otherLocalY,otherVelocity,otherLaneID,otherDirection,otherMovement])
            elif(inputOrDecoder == decoderStr):
                lastInputClassInfo = MovementToClassForm(otherMovement)
                tempInput.extend([otherLocalX,otherLocalY,otherVelocity,lastInputClassInfo[0],lastInputClassInfo[1],lastInputClassInfo[2]])
            else:
                print('Unknown inputOrDecoder string : ' +  inputOrDecoder)
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
            if(inputOrDecoder == inputStr):
                tempInput.extend([otherLocalX,otherLocalY,otherVelocity,otherLaneID,otherDirection,otherMovement])
            elif(inputOrDecoder == decoderStr):
                lastInputClassInfo = MovementToClassForm(otherMovement)
                tempInput.extend([otherLocalX,otherLocalY,otherVelocity,lastInputClassInfo[0],lastInputClassInfo[1],lastInputClassInfo[2]])

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

    currentID = processItme[0]
    currentTrainOrValStr = processItme[1]
    targetVehicleID = processItme[2]

    if(targetVehicleID == None):
        print('Traget Vehicle ID is none!!!!')

    currentVehicleList = dictByVehicles[str(currentID)]

    # Add the check for the side origins and side destination
    sideOrigin = currentVehicleList[0][originIndex]
    sideDestination = currentVehicleList[0][destinationIndex]
    if ((sideOrigin == 111 and sideDestination == 203) or (sideOrigin == 103 and sideDestination == 211) or (sideOrigin == 105 and sideDestination == 210) or (sideOrigin == 110 and sideDestination == 205) or (sideOrigin == 107 and sideDestination == 209) or (sideOrigin == 109 and sideDestination == 207)):
        return

    # and straight to straight vehicles
    if ((sideOrigin == 101 and sideDestination == 208) or (sideOrigin == 108 and sideDestination == 201)):
        return

    currentVehicleLength = len(currentVehicleList)

    for idx in range(historyTemporal,currentVehicleLength-futureTemporal):

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
            tempInput = [localX,localY,velocity,laneID,direction,movement]

            # Prepare the surrounding cars information
            # Gather vehicles using the same frame using the Frame Dict
            currentInputFrame = currentVehicleList[jdx][frameIDIndex]
            currentInputTime = currentVehicleList[jdx][globalTimeIndex]
            otherVechiles = dictByFrames[str(currentInputTime)]

            # Remove vehicles with a different global time which is not possible. Just adding check to be sure
            for fdx,eachOtherTime in enumerate(otherVechiles):
                otherTime = eachOtherTime[globalTimeIndex]
                if (otherTime != currentInputTime):
                    print('Mismatch in input global time..')
                    print('other Time ' + str(otherTime))
                    print('Current Time ' + str(currentInputTime))
                    sys.exit()

            # Extend the surrounding cars info into the target vehicles input
            tempInput = GetSurroundingCarsInfo(otherVechiles, tempInput, targetVehicleID, inputStr, localX, localY)

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

            # Remove vehicles with a different global time. Which is not possible. Just to double check
            for gdx,eachOtherTime in enumerate(otherVechiles):
                otherTime = eachOtherTime[globalTimeIndex]
                if (otherTime != currentInputTime):
                    print('Mismatch in decoder global time..')
                    print('other Time ' + str(otherTime))
                    print('Current Time ' + str(currentInputTime))
                    sys.exit()


            # Extend the surrounding cars info into the target vehicles decoder input
            decoderTemp = GetSurroundingCarsInfo(otherVechiles, decoderTemp, targetVehicleID, decoderStr, nextLocalX, nextLocalY)

            # Check the decoder feature length
            if (len(decoderTemp) != globalDecoderFeatures):
                print('decoderTemp len is : ' + str(len(decoderTemp)) + ' instead of ' + str(globalIglobalDecoderFeaturesnputFeatures))
                sys.exit()

            # Finally append the target car and surrounding cars info for the current frame into the final decoded input
            decoderInputData.append(decoderTemp)


        # Append in the final validation or training set based of decided vehicle ID
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
    def __init__(self, input = [], decoderInput = [], state = [], output=[], groundTruth = []):
        self.input = input
        self.decoderInput = decoderInput
        self.state = state
        self.output = output
        self.groundTruth = groundTruth


if __name__ == '__main__':

    if(processOrRead == processStr):
        os.mkdir(folderName)
        XTrain,decoderTrainInput,YClassTrain,YVelTrain,YPoseTrain,XVal,decoderValInput,YClassVal,YVelVal,YPoseVal = TrainData(testTrajFilePath)
    elif(processOrRead == readStr):
        # Re-Load the Vehicle and Frame based Dictionaries to populate the min max gloab values and global dicts
        # global dictByFrames, dictByVehicles, validationVehicles
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
    else:
        print('Unknown Process Read String')
        sys.exit()

    # Read the validation file 
    valFileObj = open(validationFileName, "r")
    valLoadedData = valFileObj.readlines()
    validationVehicleList = []

    for eachValVehicle in valLoadedData:
        validationVehicleList.append(eachValVehicle.rstrip())
    
    valFileObj.close()

    # Define the array shapes
    sampleCount = XTrain.shape[0]
    temporal = XTrain.shape[1]
    features = XTrain.shape[2]
    outputFeatures = 6
    decoderFeatures = (outputFeatures*(surroudingCarCounts+1))
    classOut = 3
    poseOut = 2
    velcoityOut = 1
    n_units = 256

    # define training encoder
    encoder_inputs = Input(shape=(None, features))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
	# define training decoder
    decoder_inputs = Input(shape=(None, decoderFeatures))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    # Define the BatchNormalization Layer
    batchNorm = BatchNormalization()
    decoder_outputs = batchNorm(decoder_outputs)

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

    # Define inference Batch Norm
    decoder_outputs = batchNorm(decoder_outputs)

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
    print('DecoderInput Shape : ' + str(decoderTrainInput.shape))
    print('YClassTrain Shape : ' + str(YClassTrain.shape))
    print('YVelTrain Shape : ' + str(YVelTrain.shape))
    print('YPoseTrain Shape : ' + str(YPoseTrain.shape))
    print('XVal Shape : ' + str(XVal.shape))
    print('decoderValInput Shape : ' + str(decoderValInput.shape))
    print('YClassVal Shape : ' + str(YClassVal.shape))
    print('YVelVal Shape : ' + str(YVelVal.shape))
    print('YPoseVal Shape : ' + str(YPoseVal.shape))


    model.fit([XTrain,decoderTrainInput], [YClassTrain,YVelTrain,YPoseTrain], batch_size=128, epochs=30, verbose=1, validation_data=([XVal,decoderValInput],[YClassVal,YVelVal,YPoseVal]), callbacks=callbacks_list)

    # Relese the arrays to save memory consumption
    del XTrain,decoderTrainInput,YClassTrain,YVelTrain,YPoseTrain
    del XVal,decoderValInput,YClassVal,YVelVal,YPoseVal

    # Test the model with the test dataset

    # Intialize the frame based distance error array with sample count as 0
    finalError = np.zeros(futureTemporal)
    count = 0


    # Prepare the trakcer Dict
    trackerDict = dict()

    # Sort the frame keys for proper sequential prediction
    allFrameKeys = sorted(dictByFrames.keys(), key=float)

    for key in allFrameKeys:
        print('Current Frame : ' + str(key) + ' ')
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

            # Append the current vehicle IDs to calculate car deaths
            carsInCurrentFrame.append(str(currentVechicleID))

        existingCars = list(trackerDict.keys())
        deathCars = list(set(existingCars) - set(carsInCurrentFrame))
        #delete the cars which are death cars from the trakcer
        for death in deathCars:
            del trackerDict[death]

        # Identify prediction eligible vehicles having 30 frames history and 50 Frames Future
        eligibleVehicleKeys = []
        for trackerKey in trackerDict.keys():
            vehicleFrameLength = len(trackerDict[trackerKey])
            if(vehicleFrameLength == (historyTemporal+futureTemporal)):
                eligibleVehicleKeys.append(trackerKey)
            
            if(vehicleFrameLength > (historyTemporal+futureTemporal)):
                print('Tracker Over populated for vehicle: ' + trackerKey)
                print('Vehicle Lenght is ' + str(vehicleFrameLength))
                print('This is unwanted event....')
                sys.exit()
        
        # If no Eligible vehicles move to next frame
        if not eligibleVehicleKeys:
            continue

        # Check if atleast one validation vehicle preset in the eligible list else update the tracker for eligible vehicles and move to next frame
        # Length of total validation vehicle should be atlst one more than subset of the current vehicles. Means atlst one present in current frame
        # eligibleValidationVehicles > 0
        eligibleValidationVehicles = len(validationVehicleList) - len(list(set(validationVehicleList)-set(eligibleVehicleKeys)))
        if(eligibleValidationVehicles<=0):
            # Update the tracker
            # Remove the first frame info from the tracker dict with vechiles having total history frame as the future trajectory for those vechiles are already predicted
            for eachRemovalKey in eligibleVehicleKeys:
                trackerDict[eachRemovalKey].pop(0)
            continue

        # Prepare a dictionary to hold all the prediction relevent information (input, decoderInput, state, predictedOutput and GroundTurthOutput) against each vehicle
        predictionDict = dict()

        # Populate all the input/gournd truth infos in the prediction dictionary
        for eachEligibleKey in eligibleVehicleKeys:
            predictionInfoObj = PredictionInfos([],[],[],[],[])
            predictionDict[eachEligibleKey] = predictionInfoObj
            # Get all the input infos from the traker dict for that specific vehicle
            totalInfo = trackerDict[eachEligibleKey].copy()
            inputInfo = totalInfo[0:historyTemporal]
            predicitionInputList = []
            for udx, eachInputInfo in enumerate(inputInfo):
                tempPredictionInput = eachInputInfo.copy()[:-2] # Ignore the last two items (FrameID and Time) for the input
                targetLocalX = eachInputInfo[0]  # 0 is poseX index in trakcer dict list
                targetLocalY = eachInputInfo[1]  # 1 is poseY index in trakcer dict list

                # Get the surrounding cars for the same frame
                # Get the surrounding car IDs by getting all keys and removing the current key
                surroudingCarIds = eligibleVehicleKeys[:]
                surroudingCarIds.remove(eachEligibleKey)

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
                        surroundingDist = math.sqrt(((surroundingCarLocalX-targetLocalX)**2) + ((surroundingCarLocalY-targetLocalY)**2))
                        surroundingCarDistanceList.append([eachSurroundingCarID,surroundingDist])
                    
                    # Sort the list based on distance and gather the lowest distance car IDs
                    surroundingCarDistanceList = sorted(surroundingCarDistanceList,key=lambda x: x[1])
                    surroundingCarDistanceArray = np.array(surroundingCarDistanceList)
                    releventSurroundingIds = surroundingCarDistanceArray[0:surroudingCarCounts,0:1]

                    # Add the relevent input of nearest cars to temp list
                    for eachReleventSurroundingID in releventSurroundingIds:
                        surroundingCarLocalX = trackerDict[eachReleventSurroundingID[0]][udx][0]  # udx for coresponnding Frame and 0 is poseX index in TrakcerDict list
                        surroundingCarLocalY = trackerDict[eachReleventSurroundingID[0]][udx][1]  # udx for coresponnding Frame and 1 is poseY index in TrakcerDict list
                        surroundingCarVelocity = trackerDict[eachReleventSurroundingID[0]][udx][2]  # udx for coresponnding Frame and 2 is poseY index in TrakcerDict list
                        surroundingCarLaneID = trackerDict[eachReleventSurroundingID[0]][udx][3]  # udx for coresponnding Frame and 3 is poseY index in TrakcerDict list
                        surroundingCarDirection = trackerDict[eachReleventSurroundingID[0]][udx][4]  # udx for coresponnding Frame and 4 is poseY index in TrakcerDict list
                        surroundingCarMovement = trackerDict[eachReleventSurroundingID[0]][udx][5]  # udx for coresponnding Frame and 5 is poseY index in TrakcerDict list
                        tempPredictionInput.extend([surroundingCarLocalX,surroundingCarLocalY,surroundingCarVelocity,surroundingCarLaneID,surroundingCarDirection,surroundingCarMovement])
                
                # Add the current frame input info in the input list
                predicitionInputList.append(tempPredictionInput)

            # Add the current Vehicles all history frame input to the prediction dict object (input field)
            predictionDict[eachEligibleKey].input = predicitionInputList

            # Add the ground truth output pose in to the prediction dict object for error calculation
            outputInfo = totalInfo[historyTemporal:historyTemporal+futureTemporal]
            #tempGroundTruthPoseList = []
            for eachOutputInfo in outputInfo:
                groundTruthPoseX = eachOutputInfo[0]  # 0 is poseX index in trakcer dict list
                groundTruthPoseY = eachOutputInfo[1]  # 1 is poseY index in trakcer dict list
                # Denormalize poseX and poseY as the traker dict is for input and it is normalized
                denormPoseX = (groundTruthPoseX*(maxLocalX-minLocalX)+minLocalX)
                denormPoseY = (groundTruthPoseY*(maxLocalY-minLocalY)+minLocalY)
                predictionDict[eachEligibleKey].groundTruth.append([denormPoseX,denormPoseY])
        
        # Add the decoder inputs in the prediction dict against each vehicle
        # Predict the encoder state for each vehicle and update the prediction dict state values
        for eachPredDictKey in predictionDict.keys():
            lastInput = predictionDict[eachPredDictKey].input[-1]
            predDecoderInput = []
            for bdx in range(0,len(lastInput),6):  # 6 is number of input features for each car
                lastInputPoseX = lastInput[bdx]
                lastInputPoseY = lastInput[bdx+1]
                lastInputVelocity = lastInput[bdx+2]
                lastInputMovement = lastInput[bdx+5]
                lastInputClassInfo = MovementToClassForm(lastInputMovement)
                predDecoderInput.extend([lastInputPoseX,lastInputPoseY,lastInputVelocity,lastInputClassInfo[0],lastInputClassInfo[1],lastInputClassInfo[2]])
            # Add the prepered decoder input in the prediction dict object (decoder input field)
            predictionDict[eachPredDictKey].decoderInput = predDecoderInput[:]

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
                finalOutput = [posePred[0][0][0],posePred[0][0][1],velcoityPred[0][0][0],classPred[0][0][0],classPred[0][0][1],classPred[0][0][2]]
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
                        lastSurroundingOutput = predictionDict[eachDecoderReleventSurroundingID[0]].output[-1]
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
            # Ignore the vehicle if it is not in the validation vehicle list
            if (eachErrorPredDictKey not in validationVehicleList):
                continue
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
            print(printList)
            # print(*printList, end='\r', flush=True)

            # Write the individual error to text file.
            f.write("%s\n" % localErrorList)

        # Remove the first frame info from the tracker dict with vechiles having total history frame as the future trajectory for those vechiles are already predicted
        for eachRemovalKey in eligibleVehicleKeys:
            trackerDict[eachRemovalKey].pop(0)


    print('Final Distance Error')
    print(finalError/count)
    print('All the cars are predcited in the dataset.')
    f.close()
    sys.exit()