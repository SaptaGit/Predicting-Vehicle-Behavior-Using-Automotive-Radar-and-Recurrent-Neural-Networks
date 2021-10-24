import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.visualization.visualize_sequences import viz_sequence
import numpy as np
import cv2
import sys
from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt
from keras.layers import  Input,LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dropout, MaxPooling1D, Concatenate, division, subtract, Lambda, BatchNormalization, LeakyReLU
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras import backend as K
import math
import multiprocessing as mp
from multiprocessing import Process, Manager
import time
import multiprocessing
from keras.callbacks import LearningRateScheduler
from keras import callbacks
import datetime

# set root_dir to the correct path to your dataset folder
# rootTrainDir = '/home/saptarshi/PythonCode/Junction/ArgoSample/'
# rootValDir = '/home/saptarshi/PythonCode/Junction/ArgoSample/'

# set original_dir to the correct path to your dataset folder
rootTrainDir = '/home/saptarshi/Downloads/train/data/'
rootValDir = '/home/saptarshi/Downloads/val/data/'

# rootTestDir = '/home/saptarshi/Downloads/test_obs/data/'
# rootSampleDir = '/home/saptarshi/PythonCode/Junction/ArgoSample/'


# Model parametrs
batchSize = 256
nepochs = 80
historyTemporal = 20
futureTemporal = 30
leakyAlphaValue = 0.3

# Train and Validation process lists
manager = Manager()
trainProcessList = manager.list()
validationProcessList = manager.list()

# To keep count of number of sample processed
countList = manager.list()

# Load all the trajectory CSV files as globa to be accesable by process function
aflTrain = ArgoverseForecastingLoader(rootTrainDir)
aflVal = ArgoverseForecastingLoader(rootValDir)
avm = ArgoverseMap()

# Number of samples in the folder
seqCountTrain = len(aflTrain)
seqCountVal = len(aflVal)
print('Number of Train samples in the folder : ' + str(seqCountTrain))
print('Number of Val samples in the folder : ' + str(seqCountVal))

# String Constants
trainStr = 'Train'
valStr = 'Validation'


# Custome Loss function
def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True))


def ProcessSamples(processItem):

    indexItem = processItem[0]
    trainValItem = processItem[1]

    # Get all the time stamps for the current sample
    if(trainValItem == trainStr):
        dataDf = aflTrain[indexItem].seq_df
    elif(trainValItem == valStr):
        dataDf = aflVal[indexItem].seq_df
    else:
        print('Unknown Train validation !!!!')
        sys.exit()

    uniqueTimeStamps = sorted(np.unique(dataDf['TIMESTAMP'].values))

    # List to hold the current agent poses
    localAgentPoseList = []

    # Select the first pose to calculate relative movement
    firstTimeStamp = uniqueTimeStamps[0]
    selectedFirstRows = dataDf.loc[dataDf['TIMESTAMP'] == firstTimeStamp]
    selectedFirstAgentRow = selectedFirstRows.loc[dataDf['OBJECT_TYPE'] == 'AGENT']
    initialX = selectedFirstAgentRow['X'].values[0]
    initialY = selectedFirstAgentRow['Y'].values[0]

    for eachUniqueTime in uniqueTimeStamps:
        selectedCurrentRows = dataDf.loc[dataDf['TIMESTAMP'] == eachUniqueTime]
        selectedCurrentAgenRow = selectedCurrentRows.loc[dataDf['OBJECT_TYPE'] == 'AGENT']
        absoluteX = selectedCurrentAgenRow['X'].values[0]
        absoluteY = selectedCurrentAgenRow['Y'].values[0]
        relativeX = abs(absoluteX-initialX)
        relativeY = abs(absoluteY-initialY)
        localAgentPoseList.append([relativeX,relativeY])
    
    # Each frame should be of length 50
    if(len(localAgentPoseList) != 50):
        print('Agent length is ' + str(len(localAgentPoseList)) +' and not 50!!!')
        sys.exit()

    # Split the input decoderInput and output
    localInput = localAgentPoseList[0:historyTemporal]
    localOutput = localAgentPoseList[historyTemporal:historyTemporal+futureTemporal]
    decoderInput = localAgentPoseList[historyTemporal-1:historyTemporal+futureTemporal-1]

    # Append everthing to the manager list to Train of Val
    if(trainValItem == trainStr):
        trainProcessList.append([localInput,decoderInput,localOutput])
    elif(trainValItem == valStr):
        validationProcessList.append([localInput,decoderInput,localOutput])
    else:
        print('Unknown Train validation String !!!!')
        sys.exit()

    countList.append(0)
    totalSamplesProcessed = len(countList)
    print('Finished Processing Sample : ' + str(totalSamplesProcessed) + '/' + str(seqCountTrain+seqCountVal))


# Load the Data create XTrain decoderTrain and YTrain for training
def LoadArgoData(trainOrVal):

    processIndexList = []

    if(trainOrVal == trainStr):
        # Prepare the index list for Train processes
        for edx in range(0,seqCountTrain):
            processIndexList.append([edx,trainStr])
    elif(trainOrVal == valStr):
        for edx in range(0,seqCountVal):
            processIndexList.append([edx,valStr])
    else:
        print('Unknown Train validation String !!!!')
        sys.exit()

    # Create the process pool and map and pin to the specified cores
    os.system("taskset -p -c 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 %d" % os.getpid())
    #os.system("taskset -p -c 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 %d" % os.getpid())
    processes = []
    numberofCores = 16
    pool = mp.Pool(numberofCores)
    pool.map(ProcessSamples,processIndexList)

    print('All the samples are processed!!!')

    normalList = []

    if(trainOrVal == trainStr):
        # Convert the Train manager list to normal list
        print('Converting the Train Manager list to normal lists.....')
        normalList = list(trainProcessList)
        print('List converted!!!')
    elif(trainOrVal == valStr):
        # Convert the Validation manager list to normal list
        print('Converting the Validation Manager list to normal lists.....')
        normalList = list(validationProcessList)
        print('List converted!!!')
    else:
        print('Unknown Train validation !!!!')
        sys.exit()

    print('Preparing inputArray!!!')    
    inputArray = np.array([x[0] for x in normalList])
    print('Preparing decoderInputArray!!!')    
    decoderInputArray = np.array([x[1] for x in normalList])
    print('Preparing outputArray!!!')    
    outputArray = np.array([x[2] for x in normalList])

    return inputArray,decoderInputArray,outputArray


# Define the Custome learing rate decays
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    if lrate < 0.00001:
        lrate = 0.00001
    return lrate


if __name__ == '__main__':

    #global globalMaxYVal,globalMinYVal

    features = 2 # Only features poseX and poseY 
    decoderFeatures = 2 # Only features poseX and poseY 
    n_units = 256

    startTime = datetime.datetime.now()

    XTrain,decoderTrain,YTrain = LoadArgoData(trainStr)
    XVal,decoderVal,YVal = LoadArgoData(valStr)

    maxRelativeX = 0.0
    maxRelativeY = 0.0

    print('Normalizing the X/Y poses!!!')

    # Identify the max Relative poses X and Y

    for ydx in range(len(XTrain)):
        currentMaxX = max(XTrain[ydx,:,0])
        currentMaxY = max(XTrain[ydx,:,1])
        if(currentMaxX>maxRelativeX):
            maxRelativeX = currentMaxX
        if(currentMaxY>maxRelativeY):
            maxRelativeY = currentMaxY

    for ydx in range(len(decoderTrain)):
        currentMaxX = max(decoderTrain[ydx,:,0])
        currentMaxY = max(decoderTrain[ydx,:,1])
        if(currentMaxX>maxRelativeX):
            maxRelativeX = currentMaxX
        if(currentMaxY>maxRelativeY):
            maxRelativeY = currentMaxY

    # Normalize each array based on the identified max values
    for ydx in range(len(XTrain)):
        XTrain[ydx,:,0] = XTrain[ydx,:,0]/maxRelativeX
        XTrain[ydx,:,1] = XTrain[ydx,:,1]/maxRelativeY
    
    for ydx in range(len(decoderTrain)):
        decoderTrain[ydx,:,0] = decoderTrain[ydx,:,0]/maxRelativeX
        decoderTrain[ydx,:,1] = decoderTrain[ydx,:,1]/maxRelativeY

    for ydx in range(len(XVal)):
        XVal[ydx,:,0] = XVal[ydx,:,0]/maxRelativeX
        XVal[ydx,:,1] = XVal[ydx,:,1]/maxRelativeY

    for ydx in range(len(decoderVal)):
        decoderVal[ydx,:,0] = decoderVal[ydx,:,0]/maxRelativeX
        decoderVal[ydx,:,1] = decoderVal[ydx,:,1]/maxRelativeY

    # Print relative X Y max and each normalized array min max
    print('Relative X max :' + str(maxRelativeX))
    print('Relative Y max :' + str(maxRelativeY))

    print('XTrain max :' + str(np.amax(XTrain)))
    print('decoderTrain max :' + str(np.amax(decoderTrain)))
    print('XVal max :' + str(np.amax(XVal)))
    print('decoderVal max :' + str(np.amax(decoderVal)))

    print('All input output data prepered!!!')

    endTime = datetime.datetime.now()

    print('Data prep time : ' + str(endTime-startTime))


    # Stacked LSTM encoder decoder with connected output    

    # define training encoder
    encoder_inputs = Input(shape=(None, features))
    # First Encoder LSTM Layer
    encoder1 = LSTM(n_units, return_state=True, return_sequences=True)
    encoder_output, state_h1, state_c1 = encoder1(encoder_inputs)
    encoder_states1 = [state_h1, state_c1]
    # Second Encoder LSTM Layer
    encoder2 = LSTM(n_units, return_state=True)
    encoder_output, state_h2, state_c2 = encoder2(encoder_output)
    encoder_states2 = [state_h2, state_c2]

	# define training decoder
    decoder_inputs = Input(shape=(None, decoderFeatures))
    # First Decoder LSTM Layer
    decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm1(decoder_inputs, initial_state=encoder_states1)
    # Second Decoder LSTM Layer
    decoder_lstm2 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm2(decoder_outputs, initial_state=encoder_states2)

    encoder_states = [state_h1, state_c1, state_h2, state_c2]

    decoder_dense30a = Dense(1024)
    decoder_Leaky30a = LeakyReLU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense30a(decoder_outputs)
    decoder_output3 = decoder_Leaky30a(decoder_output3)
    decoder_dense30 = Dense(512)
    decoder_Leaky30 = LeakyReLU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense30(decoder_output3)
    decoder_output3 = decoder_Leaky30(decoder_output3)
    decoder_dense31 = Dense(256)
    decoder_Leaky31 = LeakyReLU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense31(decoder_output3)
    decoder_output3 = decoder_Leaky31(decoder_output3)
    decoder_dense32 = Dense(128)
    decoder_Leaky32 = LeakyReLU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense32(decoder_output3)
    decoder_output3 = decoder_Leaky32(decoder_output3)
    decoder_dense33 = Dense(64)
    decoder_Leaky33 = LeakyReLU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense33(decoder_output3)
    decoder_output3 = decoder_Leaky33(decoder_output3)
    decoder_dense34 = Dense(32)
    decoder_Leaky34 = LeakyReLU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense34(decoder_output3)
    decoder_output3 = decoder_Leaky34(decoder_output3)
    decoder_dense35 = Dense(2, activation='linear', name='Position')
    positionOut = decoder_dense35(decoder_output3)
    
    model = Model([encoder_inputs, decoder_inputs], [positionOut])

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

    #Inference  Decoder for position out
    decoder_output3 = decoder_dense30a(decoder_outputs)
    decoder_output3 = decoder_Leaky30a(decoder_output3)
    decoder_output3 = decoder_dense30(decoder_output3)
    decoder_output3 = decoder_Leaky30a(decoder_output3)
    decoder_output3 = decoder_dense31(decoder_output3)
    decoder_output3 = decoder_Leaky31(decoder_output3)
    decoder_output3 = decoder_dense32(decoder_output3)
    decoder_output3 = decoder_Leaky32(decoder_output3)
    decoder_output3 = decoder_dense33(decoder_output3)
    decoder_output3 = decoder_Leaky33(decoder_output3)
    decoder_output3 = decoder_dense34(decoder_output3)
    decoder_output3 = decoder_Leaky34(decoder_output3)

    positionOut = decoder_dense35(decoder_output3)

    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [positionOut] + decoder_states)

    # Custom decay rates
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [lrate]
    opt = RMSprop()

    model.compile(optimizer=opt, loss=[euclidean_distance_loss])
    model.summary()

    print('XTrain Shape : ' + str(XTrain.shape))
    print('YTrain Shape : ' + str(YTrain.shape))
    print('decoderTrain Shape : ' + str(decoderTrain.shape))

    model.fit([XTrain,decoderTrain], [YTrain], batch_size=batchSize, epochs=nepochs, validation_data=([XVal,decoderVal], [YVal]), verbose=1, callbacks=callbacks_list)


    # Test the trained model

    # Intialize the frame based distance error array with sample count as 0
    finalError = np.zeros(futureTemporal)
    count = 0

    xValSampleCount = len(XVal)

    # Predict sequence
    for pdx,eachXVal in  enumerate(XVal):
        currentPredictInput = np.array(eachXVal).reshape(1,historyTemporal,features)
        groundTruthPose = YVal[pdx]

        state = encoder_model.predict(currentPredictInput)

        # Prepare the start of target sequence with the last item of the input
        lastMovementForPred = eachXVal[-1]


        target_seq = np.array(lastMovementForPred).reshape(1,1,decoderFeatures)

        outputPose = []

        # Perfrom the sequential prediction
        for t in range(futureTemporal):
            # predict next Features
            posePred, h1, c1, h2, c2 = decoder_model.predict([target_seq] + state)

            # store prediction
            outputPose.append([posePred[0][0][0],posePred[0][0][1]])

            # Normalize the predicted local poses for next instance prediction
            normalizedPredPoseX = posePred[0][0][0]/maxRelativeX
            normalizedPredPoseY = posePred[0][0][1]/maxRelativeY

            # update state
            state = [h1, c1, h2, c2]
            # update target sequence
            target_seq = np.array([normalizedPredPoseX,normalizedPredPoseY]).reshape(1,1,decoderFeatures)

        
        # Calculate the euclidian error
        currentError = []

        for ndx in range(futureTemporal):
            truePoseX = groundTruthPose[ndx][0]
            truePoseY = groundTruthPose[ndx][1]

            predPoseX = outputPose[ndx][0]
            predPoseY = outputPose[ndx][1]

            euclidianError = math.sqrt(((truePoseX-predPoseX)**2) + ((truePoseY-predPoseY)**2))

            currentError.append(euclidianError)
        
        # Keep count for average calculation and display average error
        count = count + 1
        finalError = finalError + np.array(currentError)

        # Print in the same line 
        print('Current Frame : ' + str(pdx) + '/' + str(xValSampleCount) + ' ', end=' ')
        printList = np.around(np.array([finalError[0],finalError[4],finalError[9],finalError[14],finalError[19],finalError[24],finalError[29]])/count, 2)
        print(*printList, end='\r', flush=True)

    print('Final Error!!!')
    print(finalError/count)
