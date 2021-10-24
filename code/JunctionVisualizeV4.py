import os
import PIL
from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt
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
from time import sleep
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K

# Specify the test trajectory csv file
testTrajFilePath = '/home/saptarshi/PythonCode/Junction/data/junc.csv'
# testTrajFilePath = '/home/saptarshi/PythonCode/Junction/data/Lankershim.csv'

# Specify the folder name with processed data to get the maxRelativeX and Y
folderName = '/home/saptarshi/PythonCode/Junction/data/Lankershim4RelativeDecoderJunc'

# Specify the model file paths
encoderModelFilename = '/home/saptarshi/PythonCode/Junction/models/SenServerModels/OldDataEncoder17V1.h5'
decoderModelFilename = '/home/saptarshi/PythonCode/Junction/models/SenServerModels/OldDataDecoder17V1.h5'

# Specify the map image file path
mapFileName = '/home/saptarshi/PythonCode/Junction/Maps/Lanekrshim.png'

# Create the visible window
cv2.namedWindow('test', cv2.WINDOW_NORMAL)

# Index of different features in the csv file
vechileIDIndex = 0
frameIDIndex = 1
totoalFrameIndex = 2
globalTimeIndex = 3
localXIndex = 4
localYIndex = 5
globalXIndex = 6
globalYIndex = 7
velocityIndex = 11
laneIDIndex = 13
originIndex = 14
destinationIndex = 15
intersectionIndex = 16
sectionIndex = 17 
directionIndex = 18
movementIndex = 19

# Model parametrs 
batchSize = 2048
nepochs = 250
historyTemporal = 30   #30
futureTemporal = 50   #50
surroudingCarCounts = 4
inputFeatureCount = 7  # 6 -> (poseX,poseY,velocity, LaneID, Movement, Direction, distance from junc)
globalInputFeatures = (surroudingCarCounts+1)*inputFeatureCount  
globalOutputFeatures = 6                          # 6 -> (poseX,poseY,velocity, Class0, Class1, Class2)
decoderFeatureCount = 7 # output + junc from dist (6+1=7)
globalDecoderFeatures = (surroudingCarCounts+1)*decoderFeatureCount 
leakyAlphaValue = 0.8   # 0.5
maximumAllowabelJuncDist = 250     #(250 Feet)
maximumSurroundingCarDist = 40     #(25 Feet)
predictionDistanceThreshold = 250  #(100 Feet )
ignoreFrameCount = 100

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

# Max relative X and Y for normalization
maxRealtiveX = -999
maxRealtiveY = -999

# Define the junction location distances
juncLocDict = {
  "1.0": 65,
  "2.0": 430,
  "3.0": 1068,
  "4.0": 1560
}

# DisplayImage limit
top = 4750
bottom = 9000
left = 300
right = 1849

# Unit constants
feetToMeter = 0.3048

# Convert Lat lon to pixel
EARTH_RADIUS = 6378137
EQUATOR_CIRCUMFERENCE = 2 * pi * EARTH_RADIUS
INITIAL_RESOLUTION = EQUATOR_CIRCUMFERENCE / 256.0
ORIGIN_SHIFT = EQUATOR_CIRCUMFERENCE / 2.0

# Declare the global in and out proj
# Create the projection from State plane to lat/lon
globalInProj = Proj(init='epsg:2229', preserve_units = True)
globalOutProj = Proj(init='epsg:4326')

def latlontopixels(lat, lon, zoom):
    mx = (lon * ORIGIN_SHIFT) / 180.0
    my = log(tan((90 + lat) * pi/360.0))/(pi/180.0)
    my = (my * ORIGIN_SHIFT) /180.0
    res = INITIAL_RESOLUTION / (2**zoom)
    px = (mx + ORIGIN_SHIFT) / res
    py = (my + ORIGIN_SHIFT) / res
    return px, py

# Get the global corner points to calculate the relative movements
globalCornerLat = 34.143
globalCornerLon = -118.363
globalCornerPixelX, globalCornerPixelY = latlontopixels(globalCornerLat, globalCornerLon, 21)

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

# Draw the trajectory using prev and current point on the passed image
def DrawGlobalTraj(prevX,prevY,currX,currY,GPSImage,color):
    # Calculate the pixel location for the prevPoint
    prevLon,prevLat = transform(globalInProj,globalOutProj,prevX,prevY)
    prevGPSX,prevGPSY = latlontopixels(prevLat, prevLon, 21) 
    prevDx = int(globalCornerPixelX - prevGPSX )*-1 - 80
    prevDy = int(globalCornerPixelY - prevGPSY)
    # Calculate the pixel location for the currPoint
    currLon,currLat = transform(globalInProj,globalOutProj,currX,currY)
    currGPSX,currGPSY = latlontopixels(currLat, currLon, 21) 
    currDx = int(globalCornerPixelX - currGPSX )*-1 - 80
    currDy = int(globalCornerPixelY - currGPSY)
    # raw the trajectory line
    GPSImage = cv2.line(GPSImage, (prevDx,prevDy), (currDx,currDy), color, 15) 
    return GPSImage

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



# # Create the two dictionaries one based on FrameID and other based on VehicleID
# def CreateVehicleAndFrameDict(loadFileName):

#     print('Creating Vehicle and Frame based dictionary')

#     loadFile = open(loadFileName, 'r')
#     loadReader = csv.reader(loadFile)
#     loadDataset = []
#     for loadRow in loadReader:
#         # if (loadRow[0] == '738' or loadRow[0] == '1755'): # remove two extreme car for better resolution
#         #     continue
#         loadDataset.append(loadRow[0:24])

#     loadDataset.pop(0)
#     sortedList = sorted(loadDataset, key=lambda x: (float(x[0]), float(x[1])))
#     datasetArray = np.array(sortedList, dtype=np.float)

#     #Create Dictionary for Mapper
#     mapper = dict()

#     # Create Dictionary with unique Frames
#     uniquFrameIds = list(np.unique(datasetArray[:,3]))
#     frameKeys = []
#     for idx in range(0, len(uniquFrameIds)):
#         frameKeys.append(str(uniquFrameIds[idx]))

#     dictionaryByFrames = {key : list() for key in frameKeys}

#     for jdx in range(0,len(datasetArray)):
#         key = str(datasetArray[jdx,3])
#         dictionaryByFrames[key].append(datasetArray[jdx])

#     # Create Dictionary with unique Vehicles
#     uniquVehicleIds = list(np.unique(datasetArray[:,0]))
#     vehicleKeys = []
#     for idx in range(0, len(uniquVehicleIds)):
#         vehicleKeys.append(str(uniquVehicleIds[idx]))

#     dictionaryByVehicles = {key : list() for key in vehicleKeys}

#     for jdx in range(0,len(datasetArray)):
#         key = str(datasetArray[jdx,0])
#         if len(dictionaryByVehicles[key])==0:
#             dictionaryByVehicles[key].append(datasetArray[jdx])
#             continue
#         lastFrame = dictionaryByVehicles[key][-1][1]
#         lastTime = dictionaryByVehicles[key][-1][3]
#         currentFrame = datasetArray[jdx][1]
#         currentTime = datasetArray[jdx][3]
#         if(abs(currentFrame-lastFrame)==1 and abs(currentTime-lastTime)==100):
#             dictionaryByVehicles[key].append(datasetArray[jdx])
#         else:
#             if key in mapper:
#                 updatedKey = mapper[key]
#                 lastFrame = dictionaryByVehicles[updatedKey][-1][1]
#                 lastTime = dictionaryByVehicles[updatedKey][-1][3]
#                 currentFrame = datasetArray[jdx][1]
#                 currentTime = datasetArray[jdx][3]
#                 if(abs(currentFrame-lastFrame)==1 and abs(currentTime-lastTime)==100):                    
#                     dictionaryByVehicles[updatedKey].append(datasetArray[jdx])
#                 else:
#                     print('Wrong Assumption regarding the  presensce of one vehicle ID exists only twice...')
#                     print('The problem occured for vehicle ID: ' + key + ' at frame: ' + str(currentFrame) + '...')
#                     sys.exit()
#             else:
#                 currentKeys = list(dictionaryByVehicles.keys())
#                 currentKeys.sort(key=float)
#                 newKey = str(float(currentKeys[-1]) + 1)
#                 mapper[key] = newKey
#                 dictionaryByVehicles[newKey] = list()
#                 dictionaryByVehicles[newKey].append(datasetArray[jdx])

#     loadFile.close()

#     return dictionaryByFrames,dictionaryByVehicles

# Plot all cars trajectory on the global GPS map
def GeneralVisualization(inputFileName, mapFileName):

    #Load the map 
    mapImage = cv2.imread(mapFileName)

    # Load the Vehicle and Frame based Dictionaries
    dictByFrames,dictByVehicles = CreateVehicleAndFrameDict(inputFileName)
    finalVehicleKeys = list(dictByVehicles.keys())
    finalVehicleKeys.sort(key=float)
    finalFrameKeys = list(dictByFrames.keys())
    finalFrameKeys.sort(key=float)

    intitalTime = 1118936700000.0    # Peachtree -> 1163030500 ----------- Lankershim -> 1118936700000.0 -> junc  ->  1118935680200.0

    # Create the projection from State plane to lat/lon
    inProj = Proj(init='epsg:2229', preserve_units = True)
    outProj = Proj(init='epsg:4326')

    # Get the corner points to calculate the relative movements
    cornerLat = 34.143
    cornerLon = -118.363
    cornerPixelX, cornerPixelY = latlontopixels(cornerLat, cornerLon, 21)

    targetSections = [1,2,3,4,5]
    targetIntersections = [1,2,3,4]

    straightCount = 0
    leftTurnCount = 0
    rightTurnCount = 0
    processedIds = []

    for currentFrame in finalFrameKeys:
        currentVehicleList = dictByFrames[str(currentFrame)]
        print('Processing Frame : ' + str(currentFrame))
        visImage = mapImage.copy()
        for eachCurrentVehicle in currentVehicleList:
            vehicleTimeStamp = eachCurrentVehicle[3]
            currentSection = eachCurrentVehicle[17]
            currentIntersection = eachCurrentVehicle[16]
            if ((vehicleTimeStamp == intitalTime) and ((currentSection in targetSections) or (currentIntersection in targetIntersections))):
                vehicleID = eachCurrentVehicle[0]
                globalX = eachCurrentVehicle[6]
                globalY = eachCurrentVehicle[7]
                localX = eachCurrentVehicle[4]
                localY = eachCurrentVehicle[5]
                direction = eachCurrentVehicle[18]
                movement = eachCurrentVehicle[19]
                destinationZone = eachCurrentVehicle[15]

                lon,lat = transform(inProj,outProj,globalX,globalY)
                pX,pY = latlontopixels(lat, lon, 21) 
                dx = int(cornerPixelX - pX )*-1 - 80
                dy = int(cornerPixelY - pY)

                color = (255,0,0)
                leftTurn = ((currentSection == 2) and (destinationZone == 211)) or ((currentSection == 3) and (destinationZone == 203)) or movement == 2
                rightTurn = ((currentSection == 2) and (destinationZone == 203)) or ((currentSection == 3) and (destinationZone == 211)) or movement == 3
                if (leftTurn):
                    color = (0,255,0)
                if (rightTurn):
                    color = (0,0,255)

                if vehicleID not in processedIds:
                    processedIds.append(vehicleID)
                    if(leftTurn):
                        leftTurnCount = leftTurnCount + 1
                    elif(rightTurn):
                        rightTurnCount = rightTurnCount + 1
                    else:
                        straightCount = straightCount + 1

                # print('Current Intersection ' + str(currentIntersection))
                # print('Current Section ' + str(currentSection))

                # if((currentIntersection == 0) and (currentSection==0)):
                # print('Check Found..............................')
                visImage = cv2.circle(visImage, (dx,dy), 25, color, -1)
                localLocStr = '(' + str(currentIntersection) + ',' + str(currentSection) + ')'
                visImage = cv2.putText(visImage, localLocStr, (dx+10, dy+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA) 

        intitalTime = intitalTime + 100
        displayImage = visImage[5062:8724,400:1749]
        displayImage = cv2.rotate(displayImage,cv2.ROTATE_90_CLOCKWISE)
        fontScale = 2
        thickness = 8
        font = cv2.FONT_HERSHEY_SIMPLEX 
        blueTextColor = (255, 0, 0) 
        greenTextColor = (0, 255, 0) 
        redTextColor = (0, 0, 255) 
        displayImage = cv2.putText(displayImage, 'Straight :', (2000,180), font, fontScale, blueTextColor, thickness, cv2.LINE_AA) 
        displayImage = cv2.putText(displayImage, 'Left Turn :', (2500,180), font, fontScale, greenTextColor, thickness, cv2.LINE_AA) 
        displayImage = cv2.putText(displayImage, 'Right Turn :', (3000,180), font, fontScale, redTextColor, thickness, cv2.LINE_AA) 

        straightCountStr = str(straightCount)
        leftTurnCountStr = str(leftTurnCount)
        rightTurnCountStr = str(rightTurnCount)

        displayImage = cv2.putText(displayImage, straightCountStr, (2325,180), font, fontScale, blueTextColor, thickness, cv2.LINE_AA) 
        displayImage = cv2.putText(displayImage, leftTurnCountStr, (2850,180), font, fontScale, greenTextColor, thickness, cv2.LINE_AA) 
        displayImage = cv2.putText(displayImage, rightTurnCountStr, (3400,180), font, fontScale, redTextColor, thickness, cv2.LINE_AA) 

        # if (float(currentFrame) > 400 and float(currentFrame) < 5000):
        #     cv2.imwrite('/home/saptarshi/PythonCode/AdvanceLSTM/JunctionVisual/' + str(currentFrame) + '.png', displayImage)

        cv2.imshow('test', displayImage)
        cv2.waitKey(1)

# Plot all cars trajectory on the global GPS map
def SectionIntersection(inputFileName, mapFileName):

    #Load the map 
    mapImage = cv2.imread(mapFileName)

    # Load the Vehicle and Frame based Dictionaries
    dictByFrames,dictByVehicles = CreateVehicleAndFrameDict(inputFileName)
    finalVehicleKeys = list(dictByVehicles.keys())
    finalVehicleKeys.sort(key=float)
    finalFrameKeys = list(dictByFrames.keys())
    finalFrameKeys.sort(key=float)

    intitalTime = 1118936700000.0    # Peachtree -> 1163030500 ----------- Lankershim -> 1118936700000.0 -> junc  ->  1118935680200.0

    # Create the projection from State plane to lat/lon
    inProj = Proj(init='epsg:2229', preserve_units = True)
    outProj = Proj(init='epsg:4326')

    # Get the corner points to calculate the relative movements
    cornerLat = 34.143
    cornerLon = -118.363
    cornerPixelX, cornerPixelY = latlontopixels(cornerLat, cornerLon, 21)

    targetSections = [2,3]
    targetIntersections = [2]

    for currentFrame in finalFrameKeys:
        currentVehicleList = dictByFrames[str(currentFrame)]
        print('Processing Frame : ' + str(currentFrame))
        visImage1 = mapImage.copy()
        visImage2 = mapImage.copy()
        for eachCurrentVehicle in currentVehicleList:
            vehicleTimeStamp = eachCurrentVehicle[3]
            currentSection = eachCurrentVehicle[17]
            currentIntersection = eachCurrentVehicle[16]
            if ((vehicleTimeStamp == intitalTime) and ((currentSection in targetSections) or (currentIntersection in targetIntersections))):
                vehicleID = eachCurrentVehicle[0]
                globalX = eachCurrentVehicle[6]
                globalY = eachCurrentVehicle[7]
                localX = eachCurrentVehicle[4]
                movement = eachCurrentVehicle[19]
                destinationZone = eachCurrentVehicle[15]

                lon,lat = transform(inProj,outProj,globalX,globalY)
                pX,pY = latlontopixels(lat, lon, 21) 
                dx = int(cornerPixelX - pX )*-1 - 80
                dy = int(cornerPixelY - pY)

                color1 = (255,0,0)
                color2 = (255,0,0)

                if(currentIntersection == 0):
                    color1 = (0,255,0)
                else:
                    color1 = (0,0,255)

                if(movement == 1):
                    color2 = (255,0,0)
                elif(movement == 2):
                    color2 = (0,255,0)
                elif(movement == 3):
                    color2 = (0,0,255)
                else:
                    color2 = (0,0,0)

                visImage1 = cv2.circle(visImage1, (dx,dy), 12, color1, -1)
                visImage2 = cv2.circle(visImage2, (dx,dy), 12, color2, -1)

        intitalTime = intitalTime + 100
        displayImage1 = visImage1[5062:8724,400:1749]
        displayImage1 = cv2.rotate(displayImage1,cv2.ROTATE_90_CLOCKWISE)
        
        displayImage2 = visImage2[5062:8724,400:1749]
        displayImage2 = cv2.rotate(displayImage2,cv2.ROTATE_90_CLOCKWISE)

        finalImage = cv2.vconcat([displayImage1,displayImage2])

        cv2.imshow('test', finalImage)
        cv2.waitKey(1)


# Play for each each vechicle...
def PlayByVehicle(inputFileName, mapFileName):

    #Load the map 
    mapImage = cv2.imread(mapFileName)

    # Load the Vehicle and Frame based Dictionaries
    dictByFrames,dictByVehicles, mapperDict = CreateVehicleAndFrameDict(inputFileName)
    finalVehicleKeys = list(dictByVehicles.keys())
    finalVehicleKeys.sort(key=float)
    finalFrameKeys = list(dictByFrames.keys())
    finalFrameKeys.sort(key=float)

    # intitalTime = 1118936700000.0    # Peachtree -> 1163030500 ----------- Lankershim -> 1118936700000.0 -> junc  ->  1118935680200.0

    # Create the projection from State plane to lat/lon
    inProj = Proj(init='epsg:2229', preserve_units = True)
    outProj = Proj(init='epsg:4326')

    # Get the corner points to calculate the relative movements
    cornerLat = 34.143
    cornerLon = -118.363
    cornerPixelX, cornerPixelY = latlontopixels(cornerLat, cornerLon, 21)

    targetSections = [2,3]
    targetIntersections = [2]

    straightCount = 0
    leftTurnCount = 0
    rightTurnCount = 0
    processedIds = []

    for currentVehicleKey in finalVehicleKeys:
        currentVehicleList = dictByVehicles[str(currentVehicleKey)]
        visImage = mapImage.copy()
        print('Processing vehicle : ' + str(currentVehicleKey))
        currentVehicleID = currentVehicleList[0][vechileIDIndex] # ID will be same for the whole vehicle list
        for idx,eachCurrentVehicle in enumerate(currentVehicleList):
            currentSection = eachCurrentVehicle[sectionIndex]
            currentIntersection = eachCurrentVehicle[intersectionIndex]
            if ((currentSection in targetSections) or (currentIntersection in targetIntersections)):
                globalX = eachCurrentVehicle[globalXIndex]
                globalY = eachCurrentVehicle[globalYIndex]
                localX = eachCurrentVehicle[localXIndex]
                localY = eachCurrentVehicle[localYIndex]
                direction = eachCurrentVehicle[directionIndex]
                movement = eachCurrentVehicle[movementIndex]
                destinationZone = eachCurrentVehicle[destinationIndex]

                lon,lat = transform(inProj,outProj,globalX,globalY)
                pX,pY = latlontopixels(lat, lon, 21) 
                dx = int(cornerPixelX - pX )*-1 - 80
                dy = int(cornerPixelY - pY)

                color = (255,0,0)
                leftTurn = ((currentSection == 2) and (destinationZone == 211)) or ((currentSection == 3) and (destinationZone == 203)) or movement == 2
                rightTurn = ((currentSection == 2) and (destinationZone == 203)) or ((currentSection == 3) and (destinationZone == 211)) or movement == 3
                if (leftTurn):
                    color = (0,255,0)
                if (rightTurn):
                    color = (0,0,255)

                if currentVehicleID not in processedIds:
                    processedIds.append(currentVehicleID)
                    if(leftTurn):
                        leftTurnCount = leftTurnCount + 1
                    elif(rightTurn):
                        rightTurnCount = rightTurnCount + 1
                    else:
                        straightCount = straightCount + 1

                # visImage = cv2.circle(visImage, (dx,dy), 25, color, -1)
                fontScale = 2
                thickness = 8
                font = cv2.FONT_HERSHEY_SIMPLEX 
                # Draw the trajecotry using the last instace
                if(idx>0):
                    prevGlobalX = currentVehicleList[idx-1][globalXIndex]
                    prevGlobalY = currentVehicleList[idx-1][globalYIndex]
                    lon,lat = transform(inProj,outProj,prevGlobalX,prevGlobalY)
                    pX,pY = latlontopixels(lat, lon, 21) 
                    prevDx = int(cornerPixelX - pX )*-1 - 80
                    prevDy = int(cornerPixelY - pY)

                    visImage = cv2.line(visImage, (prevDx,prevDy), (dx,dy), color, thickness)

                # localLocStr = '(' + str(currentIntersection) + ',' + str(currentSection) + ')'
                # visImage = cv2.putText(visImage, localLocStr, (dx+10, dy+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA) 

                # intitalTime = intitalTime + 100
                displayImage = visImage[5062:8724,400:1749]
                displayImage = cv2.rotate(displayImage,cv2.ROTATE_90_CLOCKWISE)

                blueTextColor = (255, 0, 0) 
                greenTextColor = (0, 255, 0) 
                redTextColor = (0, 0, 255) 
                displayImage = cv2.putText(displayImage, 'Straight :', (2000,180), font, fontScale, blueTextColor, thickness, cv2.LINE_AA) 
                displayImage = cv2.putText(displayImage, 'Left Turn :', (2500,180), font, fontScale, greenTextColor, thickness, cv2.LINE_AA) 
                displayImage = cv2.putText(displayImage, 'Right Turn :', (3000,180), font, fontScale, redTextColor, thickness, cv2.LINE_AA) 

                straightCountStr = str(straightCount)
                leftTurnCountStr = str(leftTurnCount)
                rightTurnCountStr = str(rightTurnCount)

                displayImage = cv2.putText(displayImage, straightCountStr, (2325,180), font, fontScale, blueTextColor, thickness, cv2.LINE_AA) 
                displayImage = cv2.putText(displayImage, leftTurnCountStr, (2850,180), font, fontScale, greenTextColor, thickness, cv2.LINE_AA) 
                displayImage = cv2.putText(displayImage, rightTurnCountStr, (3400,180), font, fontScale, redTextColor, thickness, cv2.LINE_AA) 

                # if (float(currentFrame) > 400 and float(currentFrame) < 5000):
                #     cv2.imwrite('/home/saptarshi/PythonCode/AdvanceLSTM/JunctionVisual/' + str(currentFrame) + '.png', displayImage)

                cv2.imshow('test', displayImage)
                cv2.waitKey(1)


# Class to hold all the relevet vehicle ID specific predicition intermediate information
class PredictionInfos():
    def __init__(self, input = [], decoderInput = [], state = [], output=[], groundTruth = [], initialPose = [], sectionIntersection = [], globalInitialPose = []):
        self.input = input
        self.decoderInput = decoderInput
        self.state = state
        self.output = output
        self.groundTruth = groundTruth
        self.initialPose = initialPose
        self.sectionIntersection = sectionIntersection
        self.globalInitialPose = globalInitialPose


# VehicleID in string to predict that specific vehicle
def PredictByVehicle(eachRelevenatVehicle):

    # Load the map 
    mapImage = cv2.imread(mapFileName)

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

    print('Waiting for GPU devices!!!')
    sleep(0.5)

    # Normalize output velocity to before concatenate
    minVelConst = K.constant(value=minVel, dtype='float32')
    minMaxVelDiffConst = K.constant(value=(maxVel-minVel), dtype='float32')

    encoder_model = load_model(encoderModelFilename)
    print('Encoder loaded!!!')
    decoder_model = load_model(decoderModelFilename, custom_objects={'minVelConst': minVelConst,'minMaxVelDiffConst' : minMaxVelDiffConst})
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
    for idx in range(historyTemporal,currentReleventVehicleLength-futureTemporal,int(futureTemporal/2)):

        # Prepare the trakcer Dict
        trackerDict = dict()
        trackerDict[targetUpdatedID] = []

        # print('Trakcer initilized!!!!')

        # for input
        for jdx in range(idx-historyTemporal,idx+futureTemporal):

            # Create a copy of map image for the current sample
            predImage = mapImage.copy()

            currentVechicleID = currentReleventVehicleList[jdx][vechileIDIndex]
            currentLocalX = currentReleventVehicleList[jdx][localXIndex]
            currentLocalY = currentReleventVehicleList[jdx][localYIndex]
            currentGPSX = currentReleventVehicleList[jdx][globalXIndex]
            currentGPSY = currentReleventVehicleList[jdx][globalYIndex]
            currentVelocity = currentReleventVehicleList[jdx][velocityIndex]
            currentLaneID = currentReleventVehicleList[jdx][laneIDIndex]
            currentDirection = currentReleventVehicleList[jdx][directionIndex]
            currentMovement = currentReleventVehicleList[jdx][movementIndex]
            currentTime = currentReleventVehicleList[jdx][globalTimeIndex]
            currentFrame = currentReleventVehicleList[jdx][frameIDIndex]
            currentSection = currentReleventVehicleList[jdx][sectionIndex]
            currentIntersection = currentReleventVehicleList[jdx][intersectionIndex]

            # Prepeare the target vehicle current input and append and at the end of the tracker dict list
            dictInput = [currentLocalX,currentLocalY,currentVelocity,currentLaneID,currentDirection,currentMovement,currentTime,currentFrame,currentSection,currentIntersection,currentGPSX,currentGPSY]
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
                currentGPSX = eachOtherVehicle[globalXIndex]
                currentGPSY = eachOtherVehicle[globalYIndex]
                currentVelocity = eachOtherVehicle[velocityIndex]
                currentLaneID = eachOtherVehicle[laneIDIndex]
                currentDirection = eachOtherVehicle[directionIndex]
                currentMovement = eachOtherVehicle[movementIndex]
                currentTime = eachOtherVehicle[globalTimeIndex]
                currentFrame = eachOtherVehicle[frameIDIndex]
                currentSection = eachOtherVehicle[sectionIndex]
                currentIntersection = eachOtherVehicle[intersectionIndex]

                dictInput = [currentLocalX,currentLocalY,currentVelocity,currentLaneID,currentDirection,currentMovement,currentTime,currentFrame,currentSection,currentIntersection,currentGPSX,currentGPSY]

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

        

        # print('Eligible vehicles found!!!!')



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
            predictionInfoObj = PredictionInfos([],[],[],[],[],[],[],[])
            predictionDict[eachEligibleKey] = predictionInfoObj
            # Get all the input infos from the traker dict for that specific vehicle
            totalInfo = trackerDict[eachEligibleKey].copy()
            inputInfo = totalInfo[0:historyTemporal]

            # Get the first element (local and GPS) for relative movement calculation and add in the prediction object
            intitalX = inputInfo[0][0]  # 0 for first item and 0 for poseX index is 0 in trakcer dict
            intitalY = inputInfo[0][1]  # 0 for first item and 1 for poseY index is 0 in trakcer dict
            predictionDict[eachEligibleKey].initialPose = [intitalX,intitalY]
            initialGPSX = inputInfo[0][10]  # 0 for first item and 10 for gpsX index is 10 in trakcer dict
            initialGPSY = inputInfo[0][11]  # 0 for first item and 10 for gpsY index is 11 in trakcer dict
            predictionDict[eachEligibleKey].globalInitialPose = [initialGPSX,initialGPSY]

            predicitionInputList = []
            for udx, eachInputInfo in enumerate(inputInfo):

                targetLocalX = eachInputInfo[0]  # 0 is poseX index in trakcer dict list
                targetLocalY = eachInputInfo[1]  # 1 is poseY index in trakcer dict list
                targetSection = eachInputInfo[8]  # 8 is section index in trakcer dict list
                targetIntersection = eachInputInfo[9]  # 9 is intersection index in trakcer dict list

                # Add check for time
                targetTime = eachInputInfo[6]  # 6 is time index in trakcer dict list

                tempPredictionInput = eachInputInfo.copy()[:-6] # Ignore the last six items (Time, FrameID, section, intersection, GPSX and GPSY ) for the input
                # convert the absolute position to normalized relative position
                tempPredictionInput[0] = abs(tempPredictionInput[0]-intitalX)/maxRealtiveX
                tempPredictionInput[1] = abs(tempPredictionInput[1]-intitalY)/maxRealtiveY

                # Calculate Nearest junction distance and extend to the input temporary row list
                juncDist = CalculateNearestJuncLoc(targetSection, targetIntersection, targetLocalX, targetLocalY)
                tempPredictionInput.insert(len(tempPredictionInput),juncDist)

                # Draw the current trajectory on the map only for the target vechile ID
                if(udx > 0 and eachEligibleKey == targetUpdatedID):
                    # Get the last global GPS pose 
                    prevGPSX = inputInfo[udx-1][10] # 10 is GPSX pose index in trakcer dict list
                    prevGPSY = inputInfo[udx-1][11] # 11 is GPSY pose index in trakcer dict list
                    # Get the current global GPS pose 
                    currGPSX = inputInfo[udx][10] # 10 is GPSX pose index in trakcer dict list
                    currGPSY = inputInfo[udx][11] # 11 is GPSY pose index in trakcer dict list
                    # Pass the global GPS locations to draw the input tajectory on the map
                    inputColorBlue = (255,0,0)
                    predImage = DrawGlobalTraj(prevGPSX,prevGPSY,currGPSX,currGPSY,predImage,inputColorBlue)
                    dispImage = predImage[top:bottom,left:right]
                    dispImage = cv2.rotate(dispImage,cv2.ROTATE_90_CLOCKWISE)
                    cv2.imshow('test',dispImage)
                    cv2.waitKey(1)


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

        # Retrive the initial local and GPS locations to plot on global map
        targetInitialLocalPose = predictionDict[targetUpdatedID].initialPose
        targetInitialLocalPoseX = targetInitialLocalPose[0]
        targetInitialLocalPoseY = targetInitialLocalPose[1]
        targetInitialGPSPose = predictionDict[targetUpdatedID].globalInitialPose
        targetInitialGPSPoseX = targetInitialGPSPose[0]
        targetInitialGPSPoseY = targetInitialGPSPose[1]

        # Intitialize the local error list to stroe the current error
        localErrorList = []
        for errorIdx,eachPose in enumerate(predictedIntermediatePose):
            # Calculate the euclidian distance error between true and predicted trajectories
            predX = predictedIntermediatePose[errorIdx][0] # 0 is poseX index in output list of prediction dict 
            predY = predictedIntermediatePose[errorIdx][1] # 1 is poseY index in output list of prediction dict
            trueX = totalGroundTruthOutputList[errorIdx][4] # 4 is poseX index in Ground Truth list of prediction dict 
            trueY = totalGroundTruthOutputList[errorIdx][5] # 5 is poseY index in Ground Truth list of prediction dict
            euclidianError = math.sqrt(((predX-trueX)**2) + ((predY-trueY)**2)) * feetToMeter
            localErrorList.append(euclidianError)
            # Draw the true and predicted trajectories
            if(errorIdx>0):
                # Draw the true traj
                # Convert the current true pose to global GPS pose
                currGlobalTurePoseX = targetInitialGPSPoseX + (trueX + targetInitialLocalPoseX)
                currGlobalTurePoseY = targetInitialGPSPoseY + (trueY + targetInitialLocalPoseY)
                # Convert the prev true pose to global GPS pose
                prevTrueX = totalGroundTruthOutputList[errorIdx-1][4] # 4 is poseX index in Ground Truth list of prediction dict 
                prevTrueY = totalGroundTruthOutputList[errorIdx-1][5] # 5 is poseY index in Ground Truth list of prediction dict
                
                prevGlobalTurePoseX = targetInitialGPSPoseX + (prevTrueX + targetInitialLocalPoseX)
                prevGlobalTurePoseY = targetInitialGPSPoseY + (prevTrueY + targetInitialLocalPoseY)
                # Draw the current true trajectoy using prev and curr global true pose
                ##########################################
                # Draw the pred traj
                # Convert the current predicted pose to global GPS pose
                currGlobalPredPoseX = targetInitialGPSPoseX + (predX + targetInitialLocalPoseX)
                currGlobalPredPoseY = targetInitialGPSPoseY + (predY + targetInitialLocalPoseY)
                # Convert the prev predicted pose to global GPS pose
                prevPredX = predictedIntermediatePose[errorIdx-1][0] # 0 is poseX index in output list of prediction dict 
                prevPredY = predictedIntermediatePose[errorIdx-1][1] # 1 is poseY index in output list of prediction dict 
                prevGlobalPredPoseX = targetInitialGPSPoseX + (prevPredX + targetInitialLocalPoseX)
                prevGlobalPredPoseY = targetInitialGPSPoseY + (prevPredY + targetInitialLocalPoseY)
                # Draw the current predicted trajectoy using prev and curr global pedicted pose
                # Pass the global True GPS locations to draw the output tajectory on the map
                trueColorGreen = (0,255,0)
                predImage = DrawGlobalTraj(prevGlobalTurePoseX,prevGlobalTurePoseY,currGlobalTurePoseX,currGlobalTurePoseY,predImage,trueColorGreen)
                # Pass the global predicted GPS locations to draw the output tajectory on the map
                predColorRed = (0,0,255)
                predImage = DrawGlobalTraj(prevGlobalPredPoseX,prevGlobalPredPoseY,currGlobalPredPoseX,currGlobalPredPoseY,predImage,predColorRed)
                # Display the image
                dispImage = predImage[top:bottom,left:right]
                dispImage = cv2.rotate(dispImage,cv2.ROTATE_90_CLOCKWISE)
                cv2.imshow('test',dispImage)
                cv2.waitKey(1)


        # Check the error list should be of lenght futureTemporal
        errorListLen = len(localErrorList)
        if(errorListLen != futureTemporal):
            print('Error list is of not expected length!!!')
            print('Expected error list length : ' + str(futureTemporal))
            print('Received error list length : ' + str(errorListLen))
            sys.exit()

        # # # Append the current local list to the main manager list
        # # errorManagerList.append(localErrorList)
        # # errorCountList.append(0)

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



    countList.append(0)
    totalSamplesProcessed = len(countList)
    print('Finished Processing Sample : ' + str(totalSamplesProcessed))


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

# Update maxRelativeX and maxRelativeY for later use
def UpdateMaxRelativeXY():

    global maxRealtiveX, maxRealtiveY

    # Read the files and populate the arrays
    # Prepare the final lists of train and validation data
    # Train final lists
    XTrain = ReadFromFile('finalXTrain', historyTemporal)
    print('Finished XTrain Array!!!')
    decoderTrainInput = ReadFromFile('finalTrainDecoderInput', futureTemporal)
    print('Finished decoderTrainInput Array!!!')

    # Validation final lists
    XVal = ReadFromFile('finalXVal', historyTemporal)
    print('Finished XVal Array!!!')
    decoderValInput = ReadFromFile('finalValDecoderInput', futureTemporal)
    print('Finished decoderValInput Array!!!')

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


if __name__ == '__main__':

    # Re-Load the Vehicle and Frame based Dictionaries to populate the min max gloab values and global dicts
    # global dictByFrames, dictByVehicles, validationVehicles, mapperDict
    dictByFrames,dictByVehicles, mapperDict = CreateVehicleAndFrameDict(testTrajFilePath)
    finalVehicleKeys = list(dictByVehicles.keys())
    finalVehicleKeys.sort(key=float)
    finalFrameKeys = list(dictByFrames.keys())
    finalFrameKeys.sort(key=float)

    # Update maxRelativeX and maxRelatievY for later use
    UpdateMaxRelativeXY()


    mapFile = '/home/saptarshi/PythonCode/Junction/Maps/Lanekrshim.png'

    # GeneralVisualization(testTrajFilePath, mapFile)

    #SectionIntersection(testTrajFilePath, mapFile)

    # PlayByVehicle(testTrajFilePath, mapFile)

    PredictByVehicle('63.0')

    print('All the cars are plotted in the scene.')

    sys.exit()







