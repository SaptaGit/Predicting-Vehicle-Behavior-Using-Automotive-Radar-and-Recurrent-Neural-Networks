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
from sys import stdout


# Specify the test trajectory csv file
#testTrajFilePath = '/home/saptarshi/PythonCode/Junction/data/junc.csv'
testTrajFilePath = '/media/disk1/sap/Junction/data/Lankershim.csv'

# Specify the result file to store each sample error
resultFileName = 'EncoderV1.txt'
f = open(resultFileName, 'x')

# Create the visible window
# cv2.namedWindow('test', cv2.WINDOW_NORMAL)

# Model parametrs 
historyTemporal = 30
futureTemporal = 30
validationVehicleCount = 300
#validationFileName = 'EncoderDecoderVal.txt'

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

# Prepare the traing and validation data
def TrainData(inputFileName):

    # Load the Vehicle and Frame based Dictionaries
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

    # Val final lists
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

        #print('Processing Vehicle : ' + str(currentVehicle))
        currentVehicleLength = len(currentVehicleList)

        for idx in range(historyTemporal,currentVehicleLength-futureTemporal):

            # Prepeare sequential Input Data
            localXData = []
            for jdx in range(idx-historyTemporal,idx):
                localX = currentVehicleList[jdx][localXIndex]
                localY = currentVehicleList[jdx][localYIndex]
                velocity = currentVehicleList[jdx][velocityIndex]
                laneID = currentVehicleList[jdx][laneIDIndex]
                direction = currentVehicleList[jdx][directionIndex]
                movement = currentVehicleList[jdx][movementIndex]
                localXData.append([movement,velocity,localX,localY,laneID,direction])


            # Prepeare sequential Output Data and decoder input data
            localYMovementData = []
            localYVelData = []
            localYPoseData = []
            decoderInputData = []

            for kdx in range(idx,idx+futureTemporal):

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

                decoderInputData.append([nextMovementClassData[0],nextMovementClassData[1],nextMovementClassData[2], nextVelocity,nextLocalX,nextLocalY])


            # Append in the final validation or training set based of decided vehicle ID
            if(currentVehicle in validationVehicles):
                finalXVal.append(localXData)
                finalYClassVal.append(localYMovementData)
                finalYVelVal.append(localYVelData)
                finalYPoseVal.append(localYPoseData)
                # Shift one time stamp right and append zeros at the beggining 
                decoderInputData = decoderInputData[:-1]
                decoderInputData.insert(0,[0,0,0,0,0,0])
                finalValDecoderInput.append(decoderInputData)
            else:
                finalXTrain.append(localXData)
                finalYClassTrain.append(localYMovementData)
                finalYPoseTrain.append(localYPoseData)
                finalYVelTrain.append(localYVelData)
                # Shift one time stamp right and append zeros at the beggining 
                decoderInputData = decoderInputData[:-1]
                decoderInputData.insert(0,[0,0,0,0,0,0])
                finalTrainDecoderInput.append(decoderInputData)


    # Prepare the final train arrays
    finalXTrainArray = np.array(finalXTrain)
    finalYClassTrainArray = np.array(finalYClassTrain)
    finalYPoseTrainArray = np.array(finalYPoseTrain)
    finalYVelTrainArray = np.array(finalYVelTrain)
    finalTrainDecoderInputArray = np.array(finalTrainDecoderInput)

    # Prepare the final validation arrays
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

    XTrain,decoderTrainInput,YClassTrain,YVelTrain,YPoseTrain,XVal,decoderValInput,YClassVal,YVelVal,YPoseVal= TrainData(testTrajFilePath)
    sampleCount = XTrain.shape[0]
    temporal = XTrain.shape[1]
    features = XTrain.shape[2]
    outputFeatures = 6
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
    decoder_inputs = Input(shape=(None, outputFeatures))
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
    loss_history = LossHistory()
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [loss_history, lrate]
    opt = RMSprop()

    model.compile(optimizer=opt, loss=['categorical_crossentropy', logcosh, euclidean_distance_loss])
    model.summary()

    print('XTrain Shape : ' + str(XTrain.shape))
    print('DecoderInput Shape : ' + str(decoderTrainInput.shape))

    model.fit([XTrain,decoderTrainInput], [YClassTrain,YVelTrain,YPoseTrain], batch_size=128, epochs=20, verbose=1, validation_data=([XVal,decoderValInput],[YClassVal,YVelVal,YPoseVal]))

    # Test the model with the test dataset

    # Intialize the frame based distance error array with sample count as 0
    finalError = np.zeros(futureTemporal)
    count = 0

    # Predict sequence
    for pdx,eachXVal in  enumerate(XVal):
        currentPredictInput = np.array(eachXVal).reshape(1,historyTemporal,features)
        groundTruthPose = YPoseVal[pdx]

        state = encoder_model.predict(currentPredictInput)
        target_seq = np.array([0,0,0,0,0,0]).reshape(1,1,outputFeatures)

        outputPose = []

        # Perfrom the sequential prediction
        for t in range(futureTemporal):
            # predict next Features
            classPred, velcoityPred, posePred, h, c = decoder_model.predict([target_seq] + state)

            # store prediction
            outputPose.append([posePred[0][0][0],posePred[0][0][1]])

            # Normalize the predicted velocity for next instance prediction
            normalizedPredVelocity = (velcoityPred-minVel)/(maxVel-minVel)

            # Normalize the predicted local poses for next instance prediction
            normalizedPredPoseX = (posePred[0][0][0]-minLocalX)/(maxLocalX-minLocalX)
            normalizedPredPoseY = (posePred[0][0][1]-minLocalY)/(maxLocalY-minLocalY)

            # update state
            state = [h, c]
            # update target sequence
            target_seq = np.array([classPred[0][0][0],classPred[0][0][1],classPred[0][0][2],normalizedPredVelocity,normalizedPredPoseX,normalizedPredPoseY]).reshape(1,1,outputFeatures)

        
        # Calculate the euclidian error
        currentError = []

        for ndx in range(futureTemporal):
            truePoseX = groundTruthPose[ndx][0]
            truePoseY = groundTruthPose[ndx][1]

            predPoseX = outputPose[ndx][0]
            predPoseY = outputPose[ndx][1]

            euclidianError = math.sqrt(((truePoseX-predPoseX)**2) + ((truePoseY-predPoseY)**2))
            euclidianErrorMeter = euclidianError*0.3048

            currentError.append(euclidianErrorMeter)
        
        # Keep count for average calculation and display average error
        count = count + 1
        finalError = finalError + np.array(currentError)

        # Print in the same line 
        printList = np.around(np.array([finalError[0],finalError[4],finalError[9],finalError[14],finalError[19],finalError[24],finalError[29]])/count, 2)
        print(*printList, end='\r', flush=True)

        # Write the individual error to text file.
        f.write("%s\n" % currentError)

    f.close()
    print('Final Distance Error')
    print(finalError/count)
    print('All the cars are predcited in the dataste.')
    sys.exit()