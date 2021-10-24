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
#from keras.models import Sequential
from keras.models import Model
#from keras.utils import Sequence
from keras.layers import  Input,LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dropout, MaxPooling1D, concatenate, division, subtract, Lambda
import matplotlib.pyplot as plt
from keras import optimizers
from keras.optimizers import RMSprop, Adam
from keras.utils import multi_gpu_model
from sklearn.preprocessing import normalize
import random
from keras import backend as K
from keras.callbacks import LearningRateScheduler
from keras import callbacks
from keras.losses import logcosh


# Specify the test trajectory csv file
testTrajFilePath = '/home/saptarshi/PythonCode/Junction/data/Lankershim.csv'

# Set the different Occupancy Grid map and scene dimensions

# Create the visible window
# cv2.namedWindow('test', cv2.WINDOW_NORMAL)


# Model parametrs 
historyTemporal = 20
futureTemporal = 1
validationVehicleCount = 300
validationFileName = 'JointNoConcatValidationVehicles.txt'

# Min Max values for normalize or denormalize
minLocalY = 0
maxLocalY = 0
minLocalX = 0
maxLocalX = 0
minVel = 0
maxVel = 0

# Index of different features in the csv file
localXIndex = 4
localYIndex = 5
velocityIndex = 11
laneIDIndex = 13
originIndex = 14
destinationIndex = 15
directionIndex = 18
movementIndex = 19


# Custome Loss function

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
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
    finalYClassTrain = []
    finalYVelTrain = []
    finalYPoseTrain = []
    finalXVal = []
    finalYClassVal = []
    finalYVelVal = []
    finalYPoseVal = []

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
                localX = currentVehicleList[jdx][localXIndex]
                localY = currentVehicleList[jdx][localYIndex]
                velocity = currentVehicleList[jdx][velocityIndex]
                laneID = currentVehicleList[jdx][laneIDIndex]
                direction = currentVehicleList[jdx][directionIndex]
                movement = currentVehicleList[jdx][movementIndex]
                localXData.append([localX,velocity,laneID,direction,movement,localY])

            nextMovement = currentVehicleList[idx + futureTemporal][movementIndex]
            # Next movements are 0, 0.5, 1 due to normalization 1/0 - through (TH), 2/0.5 - left-turn (LT), 3/1 - right-turn.....
            if(nextMovement == 0):
                localYMovementData = [1,0,0]
            elif(nextMovement == 0.5):
                localYMovementData = [0,1,0]
            elif(nextMovement == 1):
                localYMovementData = [0,0,1]
            else:
                print('Unknow movement data!!!')
                sys.exit()
            
            nextVelocity = currentVehicleList[idx + futureTemporal][velocityIndex]
            deNormalizedNextVelocity = (nextVelocity*(maxVel-minVel))+minVel
            nextLocalX = currentVehicleList[idx + futureTemporal][localXIndex]
            denormalizedNextLocalX = (nextLocalX*(maxLocalX-minLocalX)+minLocalX)
            nextLocalY = currentVehicleList[idx + futureTemporal][localYIndex]
            denormalizedNextLocalY = (nextLocalY*(maxLocalY-minLocalY)+minLocalY)

            localYVelData = [deNormalizedNextVelocity]
            localYPoseData = [denormalizedNextLocalX,denormalizedNextLocalY]

            if(currentVehicle in validationVehicles):
                finalXVal.append(localXData)
                finalYClassVal.append(localYMovementData)
                finalYVelVal.append(localYVelData)
                finalYPoseVal.append(localYPoseData)
            else:
                finalXTrain.append(localXData)
                finalYClassTrain.append(localYMovementData)
                finalYPoseTrain.append(localYPoseData)
                finalYVelTrain.append(localYVelData)

    finalXTrainArray = np.array(finalXTrain)
    finalYClassTrainArray = np.array(finalYClassTrain)
    finalYPoseTrainArray = np.array(finalYPoseTrain)
    finalYVelTrainArray = np.array(finalYVelTrain)

    finalXValArray = np.array(finalXVal)
    finalYClassValArray = np.array(finalYClassVal)
    finalYVelValArray = np.array(finalYVelVal)
    finalYPoseValArray = np.array(finalYPoseVal)

    return finalXTrainArray,finalYClassTrainArray,finalYVelTrainArray,finalYPoseTrainArray,finalXValArray,finalYClassValArray,finalYVelValArray,finalYPoseValArray

# Define the Custome learing rate decays
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.8
    epochs_drop = 1.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    if lrate < 0.00001:
        lrate = 0.00001
    return lrate

class LossHistory(callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.losses = []
        #self.vallosses = []
        self.lr = []
 
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        #self.vallosses.append(logs.get('val_loss'))
        self.lr.append(step_decay(len(self.losses)))
        #print('\n Current lr = ' + str(self.lr[-1]))
        #print('\n Current Val Loss = ' + str(self.vallosses[-1]))


if __name__ == '__main__':

    XTrain,YClassTrain,YVelTrain,YPoseTrain,XVal,YClassVal,YVelVal,YPoseVal= TrainData(testTrajFilePath)
    sampleCount = XTrain.shape[0]
    temporal = XTrain.shape[1]
    features = XTrain.shape[2]

    inp = Input((temporal,features))

    LSTMLayer1 = LSTM(256, activation='relu', return_sequences=True)(inp)
    LSTMLayer2 = LSTM(128, activation='relu')(LSTMLayer1)

    commonLayer = Dense(256, activation='relu')(LSTMLayer2)
    commonLayer = Dense(128, activation='relu')(commonLayer)


    classificationLayer = Dense(64, activation='relu')(commonLayer)
    classificationLayer = Dense(32, activation='relu')(classificationLayer)
    classificationLayer = Dense(16, activation='relu')(classificationLayer)
    classificationLayer = Dense(8, activation='relu')(classificationLayer)

    classOut = Dense(3, activation='softmax', name='Class')(classificationLayer)

    velocityLayer = concatenate([commonLayer,classOut])
    velocityLayer = Dense(128, activation='relu')(velocityLayer)
    velocityLayer = Dense(64, activation='relu')(velocityLayer)
    velocityLayer = Dense(32, activation='relu')(velocityLayer)
    velocityLayer = Dense(16, activation='relu')(velocityLayer)
    velocityLayer = Dense(8, activation='relu')(velocityLayer)

    velocityOut = Dense(1, activation='linear', name='Velocity')(velocityLayer)

    minVelConst = K.constant(value=minVel, dtype='float32')
    minMaxVelDiffConst = K.constant(value=(maxVel-minVel), dtype='float32')

    velocityNormalized = Lambda(lambda x: x-minVelConst)(velocityOut)
    velocityNormalized = Lambda(lambda x: x/minMaxVelDiffConst)(velocityNormalized)

    positionLayer = concatenate([commonLayer,classOut,velocityNormalized])
    positionLayer = Dense(256, activation='relu')(commonLayer)
    positionLayer = Dense(128, activation='relu')(commonLayer)
    positionLayer = Dense(64, activation='relu')(commonLayer)
    positionLayer = Dense(32, activation='relu')(positionLayer)
    positionLayer = Dense(16, activation='relu')(positionLayer)
    positionLayer = Dense(8, activation='relu')(positionLayer)

    positionOut = Dense(2, activation='linear', name='Position')(positionLayer)


    model = Model(inp, [classOut,velocityOut,positionOut])

    # Custom decay rates
    loss_history = LossHistory()
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [loss_history, lrate]
    opt = RMSprop()

    model.compile(optimizer=opt, loss=['categorical_crossentropy', logcosh, euclidean_distance_loss])
    model.summary()

    model.fit(XTrain, [YClassTrain,YVelTrain,YPoseTrain], batch_size=128, epochs=10, verbose=1, validation_data=(XVal,[YClassVal,YVelVal,YPoseVal]), callbacks=callbacks_list)

    model.save('./models/JointNoConcat.h5')

    print('All the cars are plotted in the scene.')

    sys.exit()







