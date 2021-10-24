import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

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
import tensorflow as tf


gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.compat.v1.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# set root_dir to the correct path to your dataset folder
# rootTrainDir = '/home/saptarshi/PythonCode/Junction/ArgoSample/'
# rootValDir = '/home/saptarshi/PythonCode/Junction/ArgoSample/'

# set original_dir to the correct path to your dataset folder
# rootTrainDir = '/home/saptarshi/Downloads/train/data/'
# rootValDir = '/home/saptarshi/Downloads/val/data/'

# set original_dir for big screen server to the correct path to your dataset folder
rootTrainDir = '/home/sap/Sap/Junction/ArgoVerseData/train/data/'
rootValDir = '/home/sap/Sap/Junction/ArgoVerseData/val/data/'

# rootTestDir = '/home/saptarshi/Downloads/test_obs/data/'
# rootSampleDir = '/home/saptarshi/PythonCode/Junction/ArgoSample/'

# Specify if process the data or read the processed data
#  'read' -> Read Data  'process' -> Process data
readStr = 'read'
processStr = 'process'
processOrRead = processStr

# Specify the folder name for the sample to read/write based on the above flag
# Path for local folder
# folderName = '/home/saptarshi/PythonCode/Junction/Argo4Surrounding'
# Path for Sen Server folder
# folderName ='/media/disk1/sap/Junction/Surrounding4RelativeExtraCheck1'
# Path for Big Screen Server folder
folderName = '/home/sap/Sap/Junction/Argo5SurroundingException'
# Path for small Server folder
# folderName = '/home/sap/Junction/Surrounding4RelativeExtraCheck'


# Model parametrs
batchSize = 256
nepochs = 80
historyTemporal = 20
futureTemporal = 30
surroundingCarCount = 5
maximumSurroundingCarDist = 50
inputFeatureCount = 7
totalInputFeature = (surroundingCarCount+1)*inputFeatureCount
decoderFetaureCount = 7
totalDecoderFeature = (surroundingCarCount+1)*decoderFetaureCount
outputFeatureCount = 2
leakyAlphaValue = 0.3

# String constanst
turnDirectionNone = 'NONE'
turnDirectionLeft = 'LEFT'
turnDirectionRight = 'RIGHT'

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
    # return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True))
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

# Extract surrounding car positions
def GetSurroundingCarInfo(dataDf,eachUniqueTime,tempInput,absoluteX,absoluteY,initialX,initialY,cityName,fileName):
    # Add the surrounding car position along with the AV vehicle (nearest surrounding count cars)
    selectedSurroundingRows = dataDf.loc[dataDf['TIMESTAMP'] == eachUniqueTime]
    selectedCurrentSurroundingRow = selectedSurroundingRows.loc[dataDf['OBJECT_TYPE'] == 'OTHERS']
    selectedCurrentAVRow = selectedSurroundingRows.loc[dataDf['OBJECT_TYPE'] == 'AV']
    #Extract the X/Y pose values
    absoluteSurroundingX = list(selectedCurrentSurroundingRow['X'].values)
    absoluteSurroundingX.append(selectedCurrentAVRow['X'].values[0])
    absoluteSurroundingY = list(selectedCurrentSurroundingRow['Y'].values)
    absoluteSurroundingY.append(selectedCurrentAVRow['Y'].values[0])

    # Length of X and Y list lenght must be same
    surroundingXLen = len(absoluteSurroundingX)
    surroundingYLen = len(absoluteSurroundingY)
    if(surroundingXLen != surroundingYLen):
        print(' Uneven surrounding X/Y list length')
        print('Surrounding X Length : ' + str(surroundingXLen))
        print('Surrounding Y Length : ' + str(surroundingYLen))
        sys.exit()

    # Calculate the distance and prepeare the list
    otherCarIndexedDistanceList = []
    for kdx in range(surroundingXLen):

        otherAbsoluteX = absoluteSurroundingX[kdx]
        otherAbsoluteY = absoluteSurroundingY[kdx]

        # Calculate distance of each other car append in the list with index value
        otherDist = math.sqrt((((otherAbsoluteX-absoluteX)**2)+((otherAbsoluteY-absoluteY)**2)))
        otherCarIndexedDistanceList.append([kdx,otherDist])

    # Sort the list based on distance and gather the lowest indexes
    otherCarIndexedDistanceList = sorted(otherCarIndexedDistanceList,key=lambda x: x[1])
    otherCarIndexedDistanceArray = np.array(otherCarIndexedDistanceList)
    releventOtherIndexes = otherCarIndexedDistanceArray[0:surroundingCarCount,0:1]

    # Append other car infos to the temp input based on the decided index
    for eachReleventIndex in releventOtherIndexes:
        otherAbsoluteX = absoluteSurroundingX[int(eachReleventIndex[0])]
        otherAbsoluteY = absoluteSurroundingY[int(eachReleventIndex[0])] 
        otherRelativeX = abs(initialX-otherAbsoluteX)
        otherRelativeY = abs(initialY-otherAbsoluteY)
        # Extract the map features for the current surrounding vehicle
        poseQuery = np.array([otherAbsoluteX,otherAbsoluteY])

        try:
            surroundingMapfeatures = ExtractMapFeatures(poseQuery,cityName)
        except Exception as e:
            print('Exception occured!!!')
            print(e)
            print('The file is :' + str(fileName))
            print('Pose query_x is ' + str(poseQuery[0]))
            print('Pose query_y is ' + str(poseQuery[1]))
            print('city name is ' + str(cityName))
            sys.exit()
            
        # Recalculate distance to eliminate distances higer than the cap
        otherDist = math.sqrt((((otherAbsoluteX-absoluteX)**2)+((otherAbsoluteY-absoluteY)**2)))

        if(otherDist<=maximumSurroundingCarDist):
            tempInput.extend([otherRelativeX,otherRelativeY])
            tempInput.extend(surroundingMapfeatures)
        else:
            tempInput.extend([0,0,0,0,0,0,0]) # 2 zeros for the poses and 5 zeros for the map features
    
    # Calculate padding requirement
    paddingCount = surroundingCarCount - len(releventOtherIndexes)
    for adx in range(paddingCount):
        tempInput.extend([0,0,0,0,0,0,0]) # 2 zeros for the poses and 5 zeros for the map features
    
    return tempInput

# Convert the turn direction String (NONE,LEFT and RIGHT) to normalized value
def TurnStrToVal(turnStr):
    turnVal = -1
    if(turnStr == turnDirectionNone):
        turnVal = 0.0
    elif(turnStr == turnDirectionLeft):
        turnVal = 0.5
    elif(turnStr == turnDirectionRight):
        turnVal = 1.0
    else:
        print('Unknown turn direction string : ' + turnStr)
        sys.exit()
    
    return turnVal

# Convert TRUE/FALSE value to normalized numeric value
def TrueFalseToVal(trueFalseVal):
    retVal = -1
    if(trueFalseVal == True):
        retVal = 1.0
    elif(trueFalseVal == False):
        retVal = 0.0
    else:
        print('Unknown turn direction string : ' + turnStr)
    
    return retVal

# Extract Map related features
def ExtractMapFeatures(poseQuery,cityName):
    laneDirectionTuple = avm.get_lane_direction(poseQuery,cityName)
    if(laneDirectionTuple!= None):
        laneDirection = laneDirectionTuple[0]
        if((type(laneDirection) is np.ndarray) and (len(laneDirection)== 2)):
            laneDirectionVal = laneDirection[0]
            laneDirectionAng = laneDirection[1]
        else:
            laneDirectionVal = 0.0
            laneDirectionAng = 0.0
    else:
        laneDirectionVal = 0.0
        laneDirectionAng = 0.0

    nearestLane = avm.get_nearest_centerline(poseQuery,cityName)
    if(nearestLane!= None):
        laneObj = nearestLane[0]
        turnDir = TurnStrToVal(laneObj.turn_direction)
        isIntersection = TrueFalseToVal(laneObj.is_intersection)
        hasTrafficControl = TrueFalseToVal(laneObj.has_traffic_control)
    else:
        turnDir = 0.0
        isIntersection = 0.0
        hasTrafficControl = 0.0

    mapFetaureList = [laneDirectionVal,laneDirectionAng,turnDir,isIntersection,hasTrafficControl]
    return mapFetaureList

# Process sample using multiple cores
def ProcessSamples(processItem):

    indexItem = processItem[0]
    trainValItem = processItem[1]

    # Get all the time stamps for the current sample
    if(trainValItem == trainStr):
        dfItem = aflTrain[indexItem]
        dataDf = dfItem.seq_df
        fileName = dfItem.current_seq.name
        
    elif(trainValItem == valStr):
        dfItem = aflVal[indexItem]
        dataDf = dfItem.seq_df
        fileName = dfItem.current_seq.name
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
    cityName = str(dataDf['CITY_NAME'].values[0])

    if(cityName != 'PIT'):
        return

    for eachUniqueTime in uniqueTimeStamps:
        # Add the Agent/Traget vehicle position
        selectedCurrentRows = dataDf.loc[dataDf['TIMESTAMP'] == eachUniqueTime]
        selectedCurrentAgenRow = selectedCurrentRows.loc[dataDf['OBJECT_TYPE'] == 'AGENT']
        absoluteX = selectedCurrentAgenRow['X'].values[0]
        absoluteY = selectedCurrentAgenRow['Y'].values[0]
        relativeX = abs(absoluteX-initialX)
        relativeY = abs(absoluteY-initialY)
        tempInput = [relativeX,relativeY]
        # Extract the map features for the target vehicle
        poseQuery = np.array([absoluteX,absoluteY])
        
        try:
            targetMapfeatures = ExtractMapFeatures(poseQuery,cityName)
        except Exception as e:
            print('Exception occured for target!!!')
            print(e)
            print('The file is :' + str(fileName))
            sys.exit()

        tempInput.extend(targetMapfeatures)
        # Add the surrounding car positons
        tempInput = GetSurroundingCarInfo(dataDf,eachUniqueTime,tempInput,absoluteX,absoluteY,initialX,initialY,cityName,fileName)
        # Check Temp input length
        if(len(tempInput) != totalInputFeature):
            print('temp input length is ' + str(len(tempInput)) + ' instead of ' + str(totalInputFeature))
            sys.exit()
        # Add the target vehicle and surrounding vehicle info in the final list
        localAgentPoseList.append(tempInput)
    
    # Each sample should be of 50 frame Length
    if(len(localAgentPoseList) != 50):
        print('Agent length is ' + str(len(localAgentPoseList)) +' and not 50!!!')
        sys.exit()

    # Split the input decoderInput and output
    localInput = localAgentPoseList[0:historyTemporal]
    localOutputArray =  np.array(localAgentPoseList[historyTemporal:historyTemporal+futureTemporal])[:,0:2] # First two columns are target car
    localOutput = []
    # Convert it to normal list
    for eachLocalOutput in localOutputArray:
        outputPoseX = eachLocalOutput[0]
        outputPoseY = eachLocalOutput[1]
        localOutput.append([outputPoseX,outputPoseY])
    decoderInput = localAgentPoseList[historyTemporal-1:historyTemporal+futureTemporal-1]
    localInfoList = [initialX,initialY,cityName]

    # Append everthing to the manager list to Train of Val
    if(trainValItem == trainStr):
        trainProcessList.append([localInput,decoderInput,localOutput,localInfoList])
    elif(trainValItem == valStr):
        validationProcessList.append([localInput,decoderInput,localOutput,localInfoList])
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
    #os.system("taskset -p -c 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 %d" % os.getpid())
    os.system("taskset -p -c 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 %d" % os.getpid())
    processes = []
    numberofCores = 20
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

    # Format the file name based on the train and validation str
    if(trainOrVal == trainStr):
        filePathInput = folderName + '/finalInputTrain.txt'
        filePathDecoder = folderName + '/finalDecoderTrain.txt'
        filePathOutput = folderName + '/finalOutputTrain.txt'
        filePathInfo = folderName + '/finalInfoTrain.txt'
    elif(trainOrVal == valStr):
        filePathInput = folderName + '/finalInputVal.txt'
        filePathDecoder = folderName + '/finalDecoderVal.txt'
        filePathOutput = folderName + '/finalOutputVal.txt'
        filePathInfo = folderName + '/finalInfoVal.txt'
    else:
        print('Unknown Train validation !!!!')
        sys.exit()

    print('Preparing inputArray!!!')
    inputList = [x[0] for x in normalList]
    WriteToFile(filePathInput,inputList)
    inputArray = np.array(inputList)

    print('Preparing decoderInputArray!!!')    
    decoderInputList = [x[1] for x in normalList]
    WriteToFile(filePathDecoder,decoderInputList)
    decoderInputArray = np.array(decoderInputList)

    print('Preparing outputArray!!!')    
    outputList = [x[2] for x in normalList]
    WriteToFile(filePathOutput,outputList)
    outputArray = np.array(outputList)

    print('Preparing infoArray!!!')    
    infoList = [x[3] for x in normalList]
    WriteInfoToFile(filePathInfo,infoList)
    infoArray = np.array(infoList)

    return inputArray,decoderInputArray,outputArray,infoArray


# Write the processed data to a file
def WriteToFile(writeFileName,samples):

    fsample = open(writeFileName, 'x')
    for eachSample in samples:
        for eachTemporal in eachSample:
            fsample.write("%s\n" % eachTemporal)
    fsample.close()

# Write the processed info data to a file (Seperate function due to the reduced dimension)
def WriteInfoToFile(writeFileName,samples):

    fsample = open(writeFileName, 'x')
    for eachSample in samples:
        fsample.write("%s\n" % eachSample)
    fsample.close()


# Read the processed data from a folder
def ReadFromFile(readFileName, temporal, seperator, trimIndex):

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
            currentSample = eachLoadedSample[1:trimIndex].split(seperator)   
            currentSampleFloat = [float(i) for i in currentSample]
            sampleList.append(currentSampleFloat) 
        dataList.append(sampleList)

    loadedDataArray = np.array(dataList)
    print(readFileName + ' Array Shape : ' + str(loadedDataArray.shape))

    return loadedDataArray


# Read the processed data from a folder
def ReadFromInfoFile(readFileName, temporal):

    readFilePath = folderName + '/' + readFileName + '.txt'
    readFile = open(readFilePath, "r")
    loadedData = readFile.readlines()

    dataList = []

    for eachLine in loadedData:
        infoTrimedLine = eachLine[1:-2].split(',')   
        infoPoseX = float(infoTrimedLine[0])
        infoPoseY = float(infoTrimedLine[1])
        cityName = infoTrimedLine[2][2:-1]
        dataList.append([infoPoseX,infoPoseY,cityName])

    loadedDataArray = np.array(dataList)
    print(readFileName + ' Array Shape : ' + str(loadedDataArray.shape))

    return loadedDataArray

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

    n_units = 256

    if(processOrRead == processStr):
        os.mkdir(folderName)
        startTime = datetime.datetime.now()
        XTrain,decoderTrain,YTrain,infoTrain = LoadArgoData(trainStr)
        XVal,decoderVal,YVal,infoVal = LoadArgoData(valStr)
        endTime = datetime.datetime.now()
        print('Data prep time : ' + str(endTime-startTime))
        sys.exit()
    elif(processOrRead == readStr):
        # Read the files and populate the arrays
        # Prepare the final lists of train and validation data
        # Train final lists
        sepeartorChar = ','
        trimIndex = -2
        XTrain = ReadFromFile('finalInputTrain', historyTemporal,sepeartorChar,trimIndex)
        print('Finished finalInputTrain Array!!!')
        decoderTrain = ReadFromFile('finalDecoderTrain', futureTemporal,sepeartorChar,trimIndex)
        print('Finished finalDecoderTrain Array!!!')
        sepeartorChar = ','
        trimIndex = -2
        YTrain = ReadFromFile('finalOutputTrain', futureTemporal,sepeartorChar,trimIndex)
        print('Finished finalOutputTrain Array!!!')
        infoTrain = ReadFromInfoFile('finalInfoTrain', futureTemporal)
        print('Finished finalInfoTrain Array!!!')

        # Validation final lists
        sepeartorChar = ','
        trimIndex = -2
        XVal = ReadFromFile('finalInputVal', historyTemporal,sepeartorChar,trimIndex)
        print('Finished finalInputVal Array!!!')
        decoderVal = ReadFromFile('finalDecoderVal', futureTemporal,sepeartorChar,trimIndex)
        print('Finished finalDecoderVal Array!!!')
        sepeartorChar = ','
        trimIndex = -2
        YVal = ReadFromFile('finalOutputVal', futureTemporal,sepeartorChar,trimIndex)
        print('Finished finalOutputVal Array!!!')
        infoVal = ReadFromInfoFile('finalInfoVal', futureTemporal)
        print('Finished finalInfoVal Array!!!')

        print('Finished All Array!!!')

    else:
        print('Unknown Process Read String')
        sys.exit()

    maxRelativeX = 0.0
    maxRelativeY = 0.0

    print('Normalizing the X/Y poses!!!')

    # Identify the max Relative poses X and Y

    for ydx in range(len(XTrain)):
        for zdx in range(0,totalInputFeature,inputFeatureCount):
            currentMaxX = max(XTrain[ydx,:,zdx])
            currentMaxY = max(XTrain[ydx,:,zdx+1])
            if(currentMaxX>maxRelativeX):
                maxRelativeX = currentMaxX
            if(currentMaxY>maxRelativeY):
                maxRelativeY = currentMaxY

    for ydx in range(len(decoderTrain)):
        for zdx in range(0,totalDecoderFeature,decoderFetaureCount):
            currentMaxX = max(decoderTrain[ydx,:,zdx])
            currentMaxY = max(decoderTrain[ydx,:,zdx+1])
            if(currentMaxX>maxRelativeX):
                maxRelativeX = currentMaxX
            if(currentMaxY>maxRelativeY):
                maxRelativeY = currentMaxY

    # Normalize each array based on the identified max values
    for ydx in range(len(XTrain)):
        for zdx in range(0,totalInputFeature,inputFeatureCount):
            XTrain[ydx,:,zdx] = XTrain[ydx,:,zdx]/maxRelativeX
            XTrain[ydx,:,zdx+1] = XTrain[ydx,:,zdx+1]/maxRelativeY
    
    for ydx in range(len(decoderTrain)):
        for zdx in range(0,totalDecoderFeature,decoderFetaureCount):
            decoderTrain[ydx,:,zdx] = decoderTrain[ydx,:,zdx]/maxRelativeX
            decoderTrain[ydx,:,zdx+1] = decoderTrain[ydx,:,zdx+1]/maxRelativeY

    for ydx in range(len(XVal)):
        for zdx in range(0,totalInputFeature,inputFeatureCount):
            XVal[ydx,:,zdx] = XVal[ydx,:,zdx]/maxRelativeX
            XVal[ydx,:,zdx+1] = XVal[ydx,:,zdx+1]/maxRelativeY

    for ydx in range(len(decoderVal)):
        for zdx in range(0,totalDecoderFeature,decoderFetaureCount):
            decoderVal[ydx,:,zdx] = decoderVal[ydx,:,zdx]/maxRelativeX
            decoderVal[ydx,:,zdx+1] = decoderVal[ydx,:,zdx+1]/maxRelativeY

    # Print relative X Y max and each normalized array min max
    print('Relative X max :' + str(maxRelativeX))
    print('Relative Y max :' + str(maxRelativeY))

    print('XTrain max :' + str(np.amax(XTrain)))
    print('decoderTrain max :' + str(np.amax(decoderTrain)))
    print('XVal max :' + str(np.amax(XVal)))
    print('decoderVal max :' + str(np.amax(decoderVal)))

    print('All input output data prepered!!!')


    # Stacked LSTM encoder decoder with connected output    

    # define training encoder
    encoder_inputs = Input(shape=(None, totalInputFeature))
    # First Encoder LSTM Layer
    encoder1 = LSTM(n_units, return_state=True, return_sequences=True)
    encoder_output, state_h1, state_c1 = encoder1(encoder_inputs)
    encoder_states1 = [state_h1, state_c1]
    # Second Encoder LSTM Layer
    encoder2 = LSTM(n_units, return_state=True)
    encoder_output, state_h2, state_c2 = encoder2(encoder_output)
    encoder_states2 = [state_h2, state_c2]

	# define training decoder
    decoder_inputs = Input(shape=(None, totalDecoderFeature))
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
        currentPredictInput = np.array(eachXVal).reshape(1,historyTemporal,totalInputFeature)
        currentDecoderInput = decoderVal[pdx]
        groundTruthPose = YVal[pdx]
        mapInfo = infoVal[pdx]
        currentSampleInitialX = float(mapInfo[0])
        currentSampleInitialY = float(mapInfo[1])
        currentSampleCityName = mapInfo[2]

        state = encoder_model.predict(currentPredictInput)

        # Prepare the start of target sequence with the last item of the input
        lastMovementForPred = eachXVal[-1]

        target_seq = np.array(lastMovementForPred).reshape(1,1,totalDecoderFeature)

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
            # Format the target vehicle features
            targetDecoder = [normalizedPredPoseX,normalizedPredPoseY]
            predAbsolutePoseX = posePred[0][0][0]+currentSampleInitialX
            predAbsolutePoseY = posePred[0][0][1]+currentSampleInitialY
            predPoseQuery = np.array([predAbsolutePoseX,predAbsolutePoseY])
            predMapfeatures = ExtractMapFeatures(predPoseQuery,currentSampleCityName)
            targetDecoder.extend(predMapfeatures)
            # Get the surrounding car features
            surroundingDecoder = currentDecoderInput[t][inputFeatureCount:]
            targetDecoder.extend(surroundingDecoder)
            target_seq = np.array(targetDecoder).reshape(1,1,totalDecoderFeature)
        
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
