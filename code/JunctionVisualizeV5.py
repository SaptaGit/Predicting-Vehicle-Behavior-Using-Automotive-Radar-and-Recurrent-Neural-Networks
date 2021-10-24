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
from multiprocessing import Process, Manager
import multiprocessing as mp




# Specify the test trajectory csv file
# testTrajFilePath = '/home/saptarshi/PythonCode/Junction/data/junc.csv'
testTrajFilePath = '/home/saptarshi/PythonCode/Junction/data/Lankershim.csv'

# Specify the image save path
imageSavePath = '/home/saptarshi/PythonCode/Junction/CaseStudy/LankershimMap1.png'

# Set the different Occupancy Grid map and scene dimensions

# Create the visible window
# cv2.namedWindow('test', cv2.WINDOW_NORMAL)


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


manager = Manager()
straightCountManager = manager.list()
leftTurnCountManager = manager.list()
rightTurnCountManager = manager.list()
sideToSideCountManager = manager.list()
straightToStraightCountMnager = manager.list()

globalDictByVeh = dict()
globalDictByFra = dict()


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
def GeneralVisualizationByVehicle(inputFileName, mapFileName):

    #Load the map 
    mapImage = cv2.imread(mapFileName)

    # Load the Vehicle and Frame based Dictionaries
    dictByFrames,dictByVehicles = CreateVehicleAndFrameDict(inputFileName)
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

    targetSections = [1,2,3]
    targetIntersections = [1,2]

    straightCount = 0
    leftTurnCount = 0
    rightTurnCount = 0
    processedIds = []

    for currentKey in finalVehicleKeys:
        currentVehicleList = dictByVehicles[str(currentKey)]
        print('Processing vehicle : ' + str(currentKey))
        vehicleLength = len(currentVehicleList)
        # decide vehicle manuever 
        initialOriginZone = currentVehicleList[0][originIndex]
        intitalDestinationZone = currentVehicleList[0][destinationIndex]

        # Get the unique movements to decide the final maneuver
        # 1/0 - through (TH), 2/0.5 - left-turn (LT), 3/1 - right-turn.....
        movementList = []
        for jdx in range(50,vehicleLength):
            movementList.append(currentVehicleList[jdx][movementIndex])

        uniqueIds = np.unique(np.array(movementList))

        if((2.0 in uniqueIds) and (3.0 in uniqueIds)):
            print('Case not solved!!!')
            sys.exit()
        
        if(2.0 in uniqueIds):
            leftTurnCount = leftTurnCount + 1
        elif(3.0 in uniqueIds):
            rightTurnCount = rightTurnCount + 1
        else:
            straightCount = straightCount + 1


        color = (255,0,0)
        if(initialOriginZone == 101 and intitalDestinationZone == 203):
            color = (0,255,0)
        if(initialOriginZone == 101 and intitalDestinationZone == 211):
            color = (0,0,255)

        lineThickness = 10
        for idx in range(1,vehicleLength):
            # vehicleTimeStamp = eachCurrentVehicle[3]
            currentSection = currentVehicleList[idx][sectionIndex]
            currentIntersection = currentVehicleList[idx][intersectionIndex]
            if ((currentSection in targetSections) or (currentIntersection in targetIntersections)):
                vehicleID = currentVehicleList[idx][vechileIDIndex]
                currentGlobalX = currentVehicleList[idx][globalXIndex]
                currentGlobalY = currentVehicleList[idx][globalYIndex]
                prevGlobalX = currentVehicleList[idx-1][globalXIndex]
                prevGlobalY = currentVehicleList[idx-1][globalYIndex]
                localX = currentVehicleList[idx][localXIndex]
                localY = currentVehicleList[idx][localYIndex]
                direction = currentVehicleList[idx][directionIndex]
                movement = currentVehicleList[idx][movementIndex]
                destinationZone = currentVehicleList[idx][destinationIndex]

                currLon,currLat = transform(inProj,outProj,currentGlobalX,currentGlobalY)
                currPx,currPy = latlontopixels(currLat, currLon, 21) 
                currDx = int(cornerPixelX - currPx )*-1 - 80
                currDy = int(cornerPixelY - currPy)

                prevLon,prevLat = transform(inProj,outProj,prevGlobalX,prevGlobalY)
                prevPx,prevPy = latlontopixels(prevLat, prevLon, 21) 
                prevDx = int(cornerPixelX - prevPx )*-1 - 80
                prevDy = int(cornerPixelY - prevPy)

                # Draw th trajectory
                mapImage = cv2.line(mapImage, (prevDx, prevDy), (currDx, currDy), color, lineThickness)



    displayImage = mapImage[5062:8724,400:1749]
    displayImage = cv2.rotate(displayImage,cv2.ROTATE_90_CLOCKWISE)
    fontScale = 2
    thickness = 8
    font = cv2.FONT_HERSHEY_SIMPLEX 
    blueTextColor = (255, 0, 0) 
    greenTextColor = (0, 255, 0) 
    redTextColor = (0, 0, 255) 
    displayImage = cv2.putText(displayImage, 'Blue:Straight', (2000,180), font, fontScale, blueTextColor, thickness, cv2.LINE_AA) 
    displayImage = cv2.putText(displayImage, 'Green:Right Turn', (2500,180), font, fontScale, greenTextColor, thickness, cv2.LINE_AA) 
    displayImage = cv2.putText(displayImage, 'Red:Left Turn1aqa', (3000,180), font, fontScale, redTextColor, thickness, cv2.LINE_AA) 


    cv2.imwrite(imageSavePath, displayImage)

    cv2.imshow('test', displayImage)
    cv2.waitKey(1)


def ManeuverProcess(item):

    currentVehicleList = globalDictByVeh[str(item)]
    print('Processing vehicle : ' + str(item))
    vehicleLength = len(currentVehicleList)

    # Add the check for the side origins and side destination
    sideOrigin = currentVehicleList[0][originIndex]
    sideDestination = currentVehicleList[0][destinationIndex]
    if ((sideOrigin == 111 and sideDestination == 203) or (sideOrigin == 103 and sideDestination == 211) or (sideOrigin == 105 and sideDestination == 210) or (sideOrigin == 110 and sideDestination == 205) or (sideOrigin == 107 and sideDestination == 209) or (sideOrigin == 109 and sideDestination == 207)):
        sideToSideCountManager.append(0)
        return

    # and straight to straight vehicles
    if ((sideOrigin == 101 and sideDestination == 208) or (sideOrigin == 108 and sideDestination == 201)):
        straightToStraightCountMnager.append(0)
        return


    # Get the unique movements to decide the final maneuver
    # 1/0 - through (TH), 2/0.5 - left-turn (LT), 3/1 - right-turn.....
    movementList = []
    for jdx in range(50,vehicleLength):
        movementList.append(currentVehicleList[jdx][movementIndex])

    uniqueIds = np.unique(np.array(movementList))

    if((2.0 in uniqueIds) and (3.0 in uniqueIds)):
        print('Case not solved!!!')
        sys.exit()
    
    if(2.0 in uniqueIds):
        leftTurnCountManager.append(0)
    elif(3.0 in uniqueIds):
        rightTurnCountManager.append(0)
    else:
        straightCountManager.append(0)


# Plot all cars trajectory on the global GPS map
def MultiManeuverCount(inputFileName, mapFileName):

    global globalDictByVeh,globalDictByFra

    # Load the Vehicle and Frame based Dictionaries
    globalDictByFra,globalDictByVeh = CreateVehicleAndFrameDict(inputFileName)
    finalVehicleKeys = list(globalDictByVeh.keys())
    finalVehicleKeys.sort(key=float)
    finalFrameKeys = list(globalDictByFra.keys())
    finalFrameKeys.sort(key=float)

    processList = []
    for currentKey in finalVehicleKeys:
        processList.append(currentKey)

    n = 50
    splittedList = [processList[i * n:(i + 1) * n] for i in range((len(processList) + n - 1) // n )] 

    for eachSplitedList in splittedList:
        # Create the process list inside the outer loop for each n vehicles
        processes = []
        for eachVehiclePorcItem in eachSplitedList:
            p = mp.Process(target=ManeuverProcess, args=(eachVehiclePorcItem,))
            processes.append(p)
            p.start()

        # Wait for all the current n process to finish. 
        for process in processes:
            process.join()

    # Count the list
    leftTurnFinal = len(leftTurnCountManager)
    rightTurnFinal = len(rightTurnCountManager)
    straightFinal = len(straightCountManager)
    sideToSideFinal = len(sideToSideCountManager)
    straightToStraightFinal = len(straightToStraightCountMnager)

    print('Left Turn : ' + str(leftTurnFinal))
    print('Right Turn : ' + str(rightTurnFinal))
    print('Straight  : ' + str(straightFinal))
    print('Side To Side  : ' + str(sideToSideFinal))
    print('Straight To Straight : ' + str(straightToStraightFinal))


# Plot all cars trajectory on the global GPS map
def DrawByVehicle(inputFileName, mapFileName):

    #Load the map 
    mapImage = cv2.imread(mapFileName)
    lineThickness = 4

    # Load the Vehicle and Frame based Dictionaries
    dictByFrames,dictByVehicles = CreateVehicleAndFrameDict(inputFileName)
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

    targetSections = [1,2,3]
    targetIntersections = [1,2]

    straightCount = 0
    leftTurnCount = 0
    rightTurnCount = 0
    processedIds = []

    for currentKey in finalVehicleKeys:
        currentVehicleList = dictByVehicles[str(currentKey)]
        print('Processing vehicle : ' + str(currentKey))
        vehicleLength = len(currentVehicleList)
        # decide vehicle manuever 
        initialOriginZone = currentVehicleList[0][originIndex]
        intitalDestinationZone = currentVehicleList[0][destinationIndex]

        # If origin zone is 101 then only draw
        if(initialOriginZone != 101):
            continue


        color = (0,255,0)
        if(initialOriginZone == 101 and intitalDestinationZone == 203):
            color = (0,255,255)
        if(initialOriginZone == 101 and intitalDestinationZone == 211):
            color = (0,0,255)

        for idx in range(1,vehicleLength):
            # vehicleTimeStamp = eachCurrentVehicle[3]
            currentSection = currentVehicleList[idx][sectionIndex]
            currentIntersection = currentVehicleList[idx][intersectionIndex]
            if ((currentSection in targetSections) or (currentIntersection in targetIntersections)):
                vehicleID = currentVehicleList[idx][vechileIDIndex]
                currentGlobalX = currentVehicleList[idx][globalXIndex]
                currentGlobalY = currentVehicleList[idx][globalYIndex]
                prevGlobalX = currentVehicleList[idx-1][globalXIndex]
                prevGlobalY = currentVehicleList[idx-1][globalYIndex]
                localX = currentVehicleList[idx][localXIndex]
                localY = currentVehicleList[idx][localYIndex]
                direction = currentVehicleList[idx][directionIndex]
                movement = currentVehicleList[idx][movementIndex]
                destinationZone = currentVehicleList[idx][destinationIndex]

                currLon,currLat = transform(inProj,outProj,currentGlobalX,currentGlobalY)
                currPx,currPy = latlontopixels(currLat, currLon, 21) 
                currDx = int(cornerPixelX - currPx )*-1 - 80
                currDy = int(cornerPixelY - currPy)

                prevLon,prevLat = transform(inProj,outProj,prevGlobalX,prevGlobalY)
                prevPx,prevPy = latlontopixels(prevLat, prevLon, 21) 
                prevDx = int(cornerPixelX - prevPx )*-1 - 80
                prevDy = int(cornerPixelY - prevPy)

                # Draw th trajectory
                mapImage = cv2.line(mapImage, (prevDx, prevDy), (currDx, currDy), color, lineThickness)


    displayImage = mapImage[5062:8724,400:1749]
    displayImage = cv2.rotate(displayImage,cv2.ROTATE_90_CLOCKWISE)
    fontScale = 3.5
    thickness = 10
    font = cv2.FONT_HERSHEY_SIMPLEX 
    blueTextColor = (0, 255, 255) 
    greenTextColor = (0, 255, 0) 
    redTextColor = (0, 0, 255) 
    displayImage = cv2.putText(displayImage, 'Green : Straight', (2300,200), font, fontScale, greenTextColor, thickness, cv2.LINE_AA) 
    displayImage = cv2.putText(displayImage, 'Yellow : Right Turn', (2300,320), font, fontScale, blueTextColor, thickness, cv2.LINE_AA) 
    displayImage = cv2.putText(displayImage, 'Red: Left Turn', (2300,440), font, fontScale, redTextColor, thickness, cv2.LINE_AA) 


    cv2.imwrite(imageSavePath, displayImage)

    # # # cv2.imshow('test', displayImage)
    # # # cv2.waitKey(1)





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

if __name__ == '__main__':

    mapFile = '/home/saptarshi/PythonCode/AdvanceLSTM/Maps/Lanekrshim.png'

    # GeneralVisualizationByVehicle(testTrajFilePath, mapFile)

    #SectionIntersection(testTrajFilePath, mapFile)

    DrawByVehicle(testTrajFilePath, mapFile)

    # MultiManeuverCount(testTrajFilePath, mapFile)

    print('All the cars are plotted in the scene.')

    sys.exit()





# # # # # # Left Turn : 548
# # # # # # Right Turn : 488
# # # # # # Straight  : 429
# # # # # # Side To Side  : 87
# # # # # # Straight To Straight : 861


