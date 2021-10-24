import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.visualization.visualize_sequences import viz_sequence
import numpy as np
import cv2
import sys
from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt
from keras.layers import  Input,LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dropout, MaxPooling1D, Concatenate, division, subtract, Lambda, BatchNormalization
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras import backend as K
import math

# set root_dir to the correct path to your dataset folder
# rootSampleDir = '/home/saptarshi/Downloads/train/data/'
# rootValDir = '/home/saptarshi/Downloads/val/data/'
# rootTestDir = '/home/saptarshi/Downloads/test_obs/data/'
rootSampleDir = '/home/saptarshi/PythonCode/Junction/ArgoSample/'

# Global Min Max position for normalization
globalMinX = 9999
globalMaxX = 0

globalMinY = 9999
globalMaxY = 0

# Model parametrs
batchSize = 256
nepochs = 50
historyTemporal = 20
futureTemporal = 30
surroundingCarCount = 5
maximumSurroundingCarDist = 60
inputFeatureCount = 7 # 2 pose 5 map fatures
totalInputFeature = (surroundingCarCount+1)*inputFeatureCount
decoderFetaureCount = 7 # 2 pose 5 map fatures
totalDecoderFeature = (surroundingCarCount+1)*decoderFetaureCount
outputFeatureCount = 2

# String constanst
turnDirectionNone = 'NONE'
turnDirectionLeft = 'LEFT'
turnDirectionRight = 'RIGHT'

# Map object
avm = ArgoverseMap()



# Custome Loss function
def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True))

# Extract surrounding car positions
def GetSurroundingCarInfo(dataDf,eachUniqueTime,tempInput,absoluteX,absoluteY,initialX,initialY,cityName):
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
        otherAbsoluteX = absoluteSurroundingX[(int(eachReleventIndex[0]))]
        otherAbsoluteY = absoluteSurroundingY[int(eachReleventIndex[0])] 
        otherRelativeX = abs(initialX-otherAbsoluteX)
        otherRelativeY = abs(initialY-otherAbsoluteY)
        # Extract the map features for the current surrounding vehicle
        # poseQuery = np.array([otherAbsoluteX,otherAbsoluteY])
        poseQuery = np.array([2167.8152648278874,869.9854239545781])
        targetMapfeatures = ExtractMapFeatures(poseQuery,'PIT')
        # targetMapfeatures = ExtractMapFeatures(poseQuery,cityName)

        # Recalculate distance to eliminate distances higer than the cap
        otherDist = math.sqrt((((otherAbsoluteX-absoluteX)**2)+((otherAbsoluteY-absoluteY)**2)))

        if(otherDist<=maximumSurroundingCarDist):
            tempInput.extend([otherRelativeX,otherRelativeY])
            tempInput.extend(targetMapfeatures)
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
    laneDirection, confValue, laneObj = avm.get_lane_direction(poseQuery,cityName)
    laneDirectionVal = laneDirection[0]
    laneDirectionAng = laneDirection[1]
    # nearestLane = avm.get_nearest_centerline(poseQuery,cityName)
    # laneObj = nearestLane[0]
    turnDir = TurnStrToVal(laneObj.turn_direction)
    isIntersection = TrueFalseToVal(laneObj.is_intersection)
    hasTrafficControl = TrueFalseToVal(laneObj.has_traffic_control)
    mapFetaureList = [laneDirectionVal,laneDirectionAng,turnDir,isIntersection,hasTrafficControl]
    return mapFetaureList

# Load the Data create XTrain and YTrain for training
def LoadArgoData(rootSampleDir):
    #Global access for min max values
    global globalMinX,globalMaxX,globalMinY,globalMaxY

    # Load all the trajectory CSV files 
    afl = ArgoverseForecastingLoader(rootSampleDir)

    # Number of samples in the folder
    seqCount = len(afl)
    print('Number of samples in the folder : ' + str(seqCount))

    globalAgenInputList = []
    globalAgenDecoderInputList = []
    globalAgenOutputList = []
    globalInfoList = []

    for jdx in range(0,seqCount):

        print('Processing Sample: ' + str(jdx) + '/' + str(seqCount))
        # Get all the time stamps for the current sample
        dataDf = afl[jdx].seq_df
        uniqueTimeStamps = sorted(np.unique(dataDf['TIMESTAMP'].values))

        localAgentPoseList = []

        # Select the first pose to calculate relative movement
        firstTimeStamp = uniqueTimeStamps[0]
        selectedFirstRows = dataDf.loc[dataDf['TIMESTAMP'] == firstTimeStamp]
        selectedFirstAgentRow = selectedFirstRows.loc[dataDf['OBJECT_TYPE'] == 'AGENT']
        initialX = selectedFirstAgentRow['X'].values[0]
        initialY = selectedFirstAgentRow['Y'].values[0]
        cityName = str(dataDf['CITY_NAME'].values[0])

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
            targetMapfeatures = ExtractMapFeatures(poseQuery,cityName)

            # poseQuery = np.array([2167.8152648278874,869.9854239545781])
            # targetMapfeatures = ExtractMapFeatures(poseQuery,'PIT')

            tempInput.extend(targetMapfeatures)
            # Add the surrounding car positons
            tempInput = GetSurroundingCarInfo(dataDf,eachUniqueTime,tempInput,absoluteX,absoluteY,initialX,initialY,cityName)
            # Check Temp input length
            if(len(tempInput) != totalInputFeature):
                print('temp input length is ' + str(len(tempInput)) + ' instead of ' + str(totalInputFeature))
                sys.exit()
            # Add the target vehicle and surrounding vehicle info in the final list
            localAgentPoseList.append(tempInput)
        
        # Each frame should be of length 50
        if(len(localAgentPoseList) != 50):
            print('Agent length is ' + str(len(localAgentPoseList)) +' and not 50!!!')
            sys.exit()

        # Split the input output
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


        globalAgenInputList.append(localInput)
        globalAgenOutputList.append(localOutput)
        globalAgenDecoderInputList.append(decoderInput)
        globalInfoList.append(localInfoList)
    
    inputArray = np.array(globalAgenInputList)
    decoderInputArray = np.array(globalAgenDecoderInputList)
    outputArray = np.array(globalAgenOutputList)
    infoArray = np.array(globalInfoList)

    return inputArray,decoderInputArray,outputArray,infoArray


if __name__ == '__main__':

    # features = 2 # Only features poseX and poseY 
    # decoderFeatures = 2 # Only features poseX and poseY 
    n_units = 256

    XTrain,decoderTrain,YTrain,TrainInfoArray = LoadArgoData(rootSampleDir)
    XVal,decoderVal,YVal,ValInfoArray = LoadArgoData(rootSampleDir)


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
    # Encoder Batch Normalization Layer
    batchNormEnc = BatchNormalization()
    encoder_output = batchNormEnc(encoder_output)
    # Second Encoder LSTM Layer
    encoder2 = LSTM(n_units, return_state=True)
    encoder_output, state_h2, state_c2 = encoder2(encoder_output)
    encoder_states2 = [state_h2, state_c2]

	# define training decoder
    decoder_inputs = Input(shape=(None, totalDecoderFeature))
    # First Decoder LSTM Layer
    decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm1(decoder_inputs, initial_state=encoder_states1)
    # Decoder Batch Normalization Layer
    batchNormDec = BatchNormalization()
    decoder_outputs = batchNormDec(decoder_outputs)
    # Second Decoder LSTM Layer
    decoder_lstm2 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm2(decoder_outputs, initial_state=encoder_states2)

    encoder_states = [state_h1, state_c1, state_h2, state_c2]


    decoder_dense30a = Dense(1024, activation='relu')
    decoder_output3 = decoder_dense30a(decoder_outputs)
    decoder_dense30 = Dense(512, activation='relu')
    decoder_output3 = decoder_dense30(decoder_output3)
    decoder_dense31 = Dense(256, activation='relu')
    decoder_output3 = decoder_dense31(decoder_output3)
    decoder_dense32 = Dense(128, activation='relu')
    decoder_output3 = decoder_dense32(decoder_output3)
    decoder_dense33 = Dense(64, activation='relu')
    decoder_output3 = decoder_dense33(decoder_output3)
    decoder_dense34 = Dense(32, activation='relu')
    decoder_output3 = decoder_dense34(decoder_output3)
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
    decoder_outputs = batchNormDec(decoder_outputs)
    decoder_outputs, state_h2, state_c2 = decoder_lstm2(decoder_outputs, initial_state=decoder_states_inputs2)
    decoder_states = [state_h1, state_c1, state_h2, state_c2]

    #Inference  Decoder for position out
    decoder_output3 = decoder_dense30a(decoder_outputs)
    decoder_output3 = decoder_dense30(decoder_output3)
    decoder_output3 = decoder_dense31(decoder_output3)
    decoder_output3 = decoder_dense32(decoder_output3)
    decoder_output3 = decoder_dense33(decoder_output3)
    decoder_output3 = decoder_dense34(decoder_output3)
    positionOut = decoder_dense35(decoder_output3)

    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [positionOut] + decoder_states)

    # Custom decay rates
    # loss_history = LossHistory()
    # lrate = LearningRateScheduler(step_decay)
    # callbacks_list = [lrate]
    opt = RMSprop()

    model.compile(optimizer=opt, loss=[euclidean_distance_loss])
    model.summary()

    print('XTrain Shape : ' + str(XTrain.shape))
    print('YTrain Shape : ' + str(YTrain.shape))
    print('decoderTrain Shape : ' + str(decoderTrain.shape))

    model.fit([XTrain,decoderTrain], [YTrain], batch_size=batchSize, epochs=nepochs, validation_data=([XVal,decoderVal], [YVal]), verbose=1)



    # Test the trained model

    # Intialize the frame based distance error array with sample count as 0
    finalError = np.zeros(futureTemporal)
    count = 0

    # Predict sequence
    for pdx,eachXVal in  enumerate(XTrain):
        currentPredictInput = np.array(eachXVal).reshape(1,historyTemporal,totalInputFeature)
        groundTruthPose = YVal[pdx]
        currentDecoderInput = decoderVal[pdx]
        mapInfo = ValInfoArray[pdx]
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
        printList = np.around(np.array([finalError[0],finalError[4],finalError[9],finalError[14],finalError[19],finalError[24],finalError[29]])/count, 2)
        print(*printList, end='\r', flush=True)

    print('Final Error!!!')
    print(finalError/count)
















            
