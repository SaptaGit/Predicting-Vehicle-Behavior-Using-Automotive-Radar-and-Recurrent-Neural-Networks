# Added surrounding car info
# Changing the prev3Angle(calculated tan angle) to annotation rotation to draw the vehicle bbox during testing.
# Current loss is around 3.5-4.5 (in case things get worse change it back)
# Diffent Maneuver and velocity fix and batchNorm and removed angle and multiple prediction for distribution and removed redundent lanes confusion matrix and augmentation
# No Angle, no maneuver, and the whole traj same class
# only loss ratio no batchnorm no dropout, check for jerky trajectory no jerk
# change batchsize to 64 to check jerky
# 64 jerky check with 32

import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
# import matplotlib.pyplot as plt
import json
# import pandas as pd
import math
import scipy.interpolate as interp
# import matplotlib.pyplot as plt
import PFHelper
import math
from keras.models import Model, load_model
from keras.models import model_from_json
#from keras.utils import Sequence
from keras.layers import  Input,LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dropout, MaxPooling1D, Concatenate, division, subtract, Lambda, BatchNormalization, LeakyReLU, Reshape, ELU, Dropout
# import matplotlib.pyplot as plt
from keras import optimizers
from keras.optimizers import RMSprop, Adam, Nadam
from keras import backend as K
from keras.callbacks import LearningRateScheduler
from keras import callbacks
from keras.losses import logcosh
import tensorflow as tf
import sys
from collections import Counter
import time
# import pptk

# # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
# # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

sequence_folder = '/home/saptarshi/Dropbox (Heriot-Watt University Team)/RES_EPS_PathCad/datasets/pathcad_van_dataset/junction_1'
# sequence_folder = '/home/saptarshi/PythonCode/Junction/data/alvie'
# sequence_folder = '/home/saptarshi/PythonCode/Junction/SightHillSample'

# imageSaveFolder = '/home/saptarshi/PythonCode/Junction/avgAngle/'
errorHistFilePath = '/home/saptarshi/PythonCode/Junction/infos/errorHistManeuver.txt'

# Error Hist Path
mapImagePath = '/home/saptarshi/PythonCode/Junction/Maps/AlvieMap.png'

# Train or test the model
trainStr = 'Train'
testStr = 'Test'
# DO NOT re-train... just test with frame drop
# Keep the model names same and just test......
trainOrTest = testStr

# Perfrom data display or not
# 1 -> Display 0 -> no display
display = 0

radarImageDimension = 1152
radarImageCentre = radarImageDimension/2

if(display == 1 or trainOrTest == testStr):
    globalRadarDisplayImage = np.zeros((radarImageDimension,radarImageDimension,3))
    globalRadarSmoothTraj = np.zeros((radarImageDimension,radarImageDimension,3))
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    # cv2.namedWindow('smooth',cv2.WINDOW_NORMAL)

    # Read the globaL RADAR Image
    radarPath = sequence_folder + '/junction_1_0/Navtech_Cartesian/000001.png'
    globalRadarDisplayImage = cv2.imread(radarPath)

# Model params
turn = 0
straight = 0
processedObjectIds = []
str_format = '{:06d}'
straightStr = 'Straight'
turnStr = 'Turn'
imageCount = 0
straightAngles = [0,90,180,270,360]
cellResolution = 0.173611
metersPerPixel = 0.29858
historyTemporal = 10   #30
futureTemporal = 20  #50
inputFeature = 9
inputFeatureWithoutID = inputFeature-3    # (2 -> withoutID and without angle and without movementInfo in input)
inputFeatureZeroPad = [0,0,0,0,0,0,0,0,0]
decoderInputFeature = 7
leakyAlphaValue = 0.8
dropOutFrac = 0.05
batchSize = 16    #16  #16    # 16  32 #64
nepochs = 100   #120
surroudingCarCounts = 3   # originally 3
totalInputFeature = (surroudingCarCounts+1)*inputFeatureWithoutID
totalDecoderFeature = (surroudingCarCounts+1)*decoderInputFeature
maxSurroudingDist = 150
repeatCount = 1
frameDropRate = 10 #  10   # Drop every nth frame and use motion to estimate position
# Custome weight ratios
classWeight = 0.355
velocityWeight = 0.2
poseWeight = 0.445

###########################################
########### Weight Ratios #################
###########################################
# 0.365,0.2,0.435 -> 0.61/1.22/3.81....(^-4)/0.95/3.6...(^-4)/0.97/3.2....
# 0.36,0.205,0.435 -> 0.62/0.97/4.0...(^-5)/0.89/3.1...(^-5)/0.92/2.9...
# 0.35,0.21,0.44 -> (^-4)/0.93/3.1...0.3,1.33,5.1,...0.0085,0.98,3.7...
# 0.34,0.22,0.44 -> 0.004,1.05,4.19...0.07,1.33,89.4...0.8,1.5,4.6....
# 0.36,0.205,0.435 -> train to test expected 2nd row......3.48, 3.58, 3.79,87.23,4.5,3.42
# 0.41,0.2,0.39 -> (old setting due to bad result....) 0.57,1.47,4.03...... (^-4),8,3.2.... error (5.6,4.5)
# 0.401,0.2,399 (step 15.0, leaky 0.8) -> (10^-5,0.98,3.99-> error )....
# train and test with the new val list...1.3,1.3,6.6.....not converging....
# new val list, 0.41,0.2,0.39 -> running....   (if not converged check the val vehicle distribution: old list converging....)
# Modified list working..
# 0.35,0.2,0.45 -> runnig... weired class result fitting class better
# 0.355,0.2,0.445 -> (^-4),0.9,4.0... error ...5.5, 4.6 dont touch anything.. remove vehicle from val do the histogram....
# then check the confusion matrix... wrong...
# change step and leaky.....
##########################################


# Main road
laneLines =   [[537,287,519,1021], [537,287,690,286], [537,287,522,63]]
# laneLines = [[140,418,456,427], [130,458,583,470], [456,427,1106,450], [583,470,1097,484], [448,427,456,89], [465,427,470,89], [600,470,605,1015], [564,470,592,1015], [567,533,112,513], [570,553,100,533], [601,537,1077,553], [601,561,1069,573]]


# Car width and height in pixel
carWidth = 15
carHeight = 30

# Info field index
poseXIndex = 0
poseYIndex = 1
velXIndex = 2
velYIndex = 3
laneDistIndex = 4
laneNumberIndex = 5
movementIndex = 6
angleIndex = 7
idIndex = 8

# Global min max poses for normalization
# Global Min max for poses
globalMaxXPose = 0
globalMaxYPose = 0
globalMinXPose = 9999
globalMinYPose = 9999
# Global Min max for Velocity
globalMaxXVelocity = 0
globalMaxYVelocity = 0
globalMinXVelocity = 9999
globalMinYVelocity = 9999
# Global Min max for Lane Dist
globalMinLaneDist = 9999
globalMaxLaneDist = 0
# Global Max Lane number
globalMaxLanenumber = len(laneLines)

# Car Info list
carInfoList = []  # Without frame dropped
carInfoListDropped = [] # With frame dropped

ignoredCars = 0

# Class strings
straightStr = 'Straight'
leftTurnStr = 'Left Turn'
rightTurnStr = 'Right Turn'

# Validation car ID list as to make the training and validation trajs same type
# # # # # validationVehicleList = [52,58,68,101,105,155]     # 159 new
# # # # # validationVehicleList = [101]    # [9,10,101,107]
# # # # # validationVehicleList = [2,9,10,17,20,22,31,50,56,87,98,101,103,107,119]
# # # # # validationVehicleList = [126,139,110,119,64,70,83,98,8,20,31,33,47,54,4,9,10,22,27,28,29,31,38,39,47,55,56,67,71,74,87,88,103,104,89,122,132,137,142,150,153]
# # # # # validationVehicleList = [126,110,119,64,83,98,8,31,33,47,54,4,9,10,22,27,28,29,31,38,39,47,55,56,67,71,74,87,103,104,89,122,132,137,142,150]


# validationVehicleList = [9,10,28,29,71,74,83,89,98,103,104,110,119,122,126,132,142]   # old list
# Vehicle pairs -> 8,9,10--27,28--38,39--47,49--52,53--54,55,56--105,106,107--153,154--171,172--177,178
# validationVehicleList = [5,6,9,10,22,28,38,39,47,49,66,71,105,106,107,171,172,178]   # new list for train ******************
# validationVehicleList = [6,10,28] # for video...
# # # # validationVehicleList = [6,10,22,28,38,39,47,66,107,31,22,58,66]   # new list for test..... removed 5.0,171,172,49,71,106,9,105 added 31,22,58,66  
# validationVehicleList = [31,3,22,58,66]
# Adding for some turn specific cases
# validationVehicleList = [4,5,6,9, 178]  # [4,5,17,27,28,29,52,53,56,74,89,105,106,107,178]
# add a few similar before testing
# validationVehicleList = [107]
validationVehicleList = [5]    # Best wrong trun trajectory..
# show input trajectory insted of predicted trajectory to show the speed, lower trajectory length indicates lower speed/velocity


# ignoreList = [30,49,64,78,90,103,174,175]

# One map to see all trajs
mapImage = cv2.imread(mapImagePath)

# Custome Loss function
def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


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


# Smooth out a jerky trajectory
def interpolate_polyline(polyline, num_points):
    duplicates = []
    for i in range(1, len(polyline)):
        if np.allclose(polyline[i], polyline[i-1]):
            duplicates.append(i)
    if duplicates:
        polyline = np.delete(polyline, duplicates, axis=0)
    tck, u = interp.splprep(polyline.T, s=0)
    u = np.linspace(0.0, 1.0, num_points)
    return np.column_stack(interp.splev(u, tck))

# Smooth the trajectories using particle filter
def SmoothTrajcetoryGeneration(poseList):

    smothedTraj = []
    smoothDroppedTraj = []
    particleCount = 500
    intialIndex = 3  # 2
    intitalCovariance = 2 #2
    pfObj = PFHelper.ParticleFilter(particleCount,[],'Classical',poseList[intialIndex][0],poseList[intialIndex][1],intitalCovariance)
    pfObjFrameDropped = PFHelper.ParticleFilter(particleCount,[],'Classical',poseList[intialIndex][0],poseList[intialIndex][1],intitalCovariance)

    trajLength = len(poseList)
    for mdx in range(0,trajLength):
        # Extract the current poses
        currentPoseX = poseList[mdx][0]
        currentPoseY = poseList[mdx][1]
        annotateAngle = poseList[mdx][2]
        # Add the first two poses
        if mdx<intialIndex:
            smothedTraj.append([int(currentPoseX),int(currentPoseY),annotateAngle])
            smoothDroppedTraj.append([int(currentPoseX),int(currentPoseY),annotateAngle])
            continue
        # Extract the prev poses for average velcity calculation
        prev1PoseX = poseList[mdx-1][0]
        prev1PoseY = poseList[mdx-1][1]
        prev2PoseX = poseList[mdx-2][0]
        prev2PoseY = poseList[mdx-2][1]
        prev3PoseX = poseList[mdx-3][0]
        prev3PoseY = poseList[mdx-3][1]
        # prev4PoseX = centrePoses[mdx-4][0]
        # prev4PoseY = centrePoses[mdx-4][1]

        # Calculate the prev velocities for filter
        prev1Vx = (prev1PoseX-prev2PoseX)
        prev1Vy = (prev1PoseY-prev2PoseY)
        prev2Vx = (prev2PoseX-prev3PoseX)
        prev2Vy = (prev2PoseY-prev3PoseY)
        # prev3Vx = (prev3PoseX-prev4PoseX)
        # prev3Vy = (prev3PoseY-prev4PoseY)

        # Calculate the average velocity
        avgVx = prev1Vx*(1/2) + prev2Vx*(1/2)
        avgVy = prev1Vy*(1/2) + prev2Vy*(1/2)

        # Check if it is every frameDropRate (10th) frame
        # If the frameDropRate is zero means no drop at all
        # If yes pass None as observation and particle filter will do the rest
        # Else update the dropped filter normally 
        if(frameDropRate != 0 and ((mdx%frameDropRate == 0) or ((mdx+1)%frameDropRate == 0) or ((mdx+2)%frameDropRate == 0))):
            pfObjFrameDropped.update(None, avgVx, avgVy)
        else:
            pfObjFrameDropped.update([currentPoseX,currentPoseY], avgVx, avgVy)
        # Now update the filter
        # pfObjFrameDropped.update([droppedPoseX,droppedPoseY], avgVx, avgVy)
        droppedFilteredX = int(pfObjFrameDropped.particleMean[0])
        droppedFilteredY = int(pfObjFrameDropped.particleMean[1])
        smoothDroppedTraj.append([int(droppedFilteredX),int(droppedFilteredY),annotateAngle])

        # Maintain the orginal filter for output
        pfObj.update([currentPoseX,currentPoseY], avgVx, avgVy)
        filteredX = int(pfObj.particleMean[0])
        filteredY = int(pfObj.particleMean[1])
        smothedTraj.append([int(filteredX),int(filteredY),annotateAngle])

    # Return the smoothed trajectory
    return smothedTraj,smoothDroppedTraj

# Estimate the nearest lane index and calculate the distance
def CalculateNearestLaneAndDist(currentPose):
    # Calculate distance from current point from each lane lines
    nearestLaneIndex = -1
    lowestDist = 9999
    for laneIndex,eacLine in enumerate(laneLines):
        p1 = np.asarray((eacLine[0],eacLine[1]))
        p2 = np.asarray((eacLine[2],eacLine[3]))
        p3 = np.asarray((currentPose[0],currentPose[1]))

        d = point_to_line_dist(p3, [p1, p2])

        if(d < lowestDist):
            lowestDist = d
            nearestLaneIndex = laneIndex
    return nearestLaneIndex,lowestDist

# Estimate the movement info based on the provided angle
def CalculateManeuverInfo(prev3Angle,originLaneIndex):
    # angleColor = (0,255,0)
    movementInfo = 'Turn'  # Turn = 1
    movementInfoFloat = -1.0 # Straight = 0, left = 0.5, right 1.0
    # Check the current angle with each angle with margin
    for eachStraightAngle in straightAngles:
        if(prev3Angle > eachStraightAngle-15 and prev3Angle < eachStraightAngle+15):
            movementInfo = 'Straight' # Straight = 0
            movementInfoFloat = 0
            break

    # Check if the movementInfo is turn. If yes check left or right turn
    if(movementInfo == 'Turn'):
        if(originLaneIndex == 4 or originLaneIndex == 5 or originLaneIndex == 6 or originLaneIndex == 7):
            if(prev3Angle>0 and prev3Angle<90) or (prev3Angle>180 and prev3Angle<270):
                movementInfo = 'Left'
                movementInfoFloat = 0.5
            elif(prev3Angle>90 and prev3Angle<180) or (prev3Angle>270 and prev3Angle<359):
                movementInfo = 'Right'
                movementInfoFloat = 1.0
            else:
                print('Unknown angle range!!!')
                sys.exit()
        else:
            if(prev3Angle>0 and prev3Angle<90) or (prev3Angle>180 and prev3Angle<270):
                movementInfo = 'Right'
                movementInfoFloat = 1.0
            elif(prev3Angle>90 and prev3Angle<180) or (prev3Angle>270 and prev3Angle<359):
                movementInfo = 'Left'
                movementInfoFloat = 0.5
            else:
                print('Unknown angle range!!!')
                sys.exit()

    return movementInfo,movementInfoFloat


# Lane based maneuver estimation
def LaneBasedManeuver(firstLaneIndex, lastLaneIndex):
    if(firstLaneIndex == 0 and lastLaneIndex == 1):
        movementInfo = 'Right'
        movementInfoFloat = 1.0
    elif(firstLaneIndex == 0 and lastLaneIndex == 2):
        movementInfo = 'Straight' # Straight = 0
        movementInfoFloat = 0
    else:
        print('Unknown angle lane!!!')
        sys.exit()

    return movementInfo,movementInfoFloat




# Calculate tangent for each point on the trajectory
def CalculateTangent(traj):

    # Extract the trajectory length and the X-Y poses
    xData = traj[:,0]
    yData = traj[:,1]
    trajLen = len(traj)

    # polynomial curve fit the test data
    fittedParameters = np.polyfit(xData, yData, 25)

    # polynomial derivative from numpy
    deriv = np.polyder(fittedParameters)

    # create data for the fitted equation plot
    xModel = np.linspace(min(xData), max(xData))
    yModel = np.polyval(fittedParameters, xModel)

    # Create empty list to hold the tangent info of each point (minX, maxX, ylow, yhigh)
    tangentInfo = []

    # Check the fitted trajectory
    fittedTraj = []
    for trajIndex in range(0,trajLen):
        currentTrajXPoint = int(traj[trajIndex,0])
        y_value_at_point = int(np.polyval(fittedParameters, currentTrajXPoint))
        fittedTraj.append([currentTrajXPoint,y_value_at_point])

    for trajIndex in range(0,trajLen):

        currentTrajXPoint = traj[trajIndex,0]
        currentTrajYPoint = traj[trajIndex,1]

        minX = currentTrajXPoint - 40
        maxX = currentTrajXPoint + 40

        y_value_at_point = np.polyval(fittedParameters, currentTrajXPoint)
        slope_at_point = np.polyval(deriv, currentTrajXPoint)

        ylow = (minX - currentTrajXPoint) * slope_at_point + y_value_at_point
        yhigh = (maxX - currentTrajXPoint) * slope_at_point + y_value_at_point

        # now the tangent as a line plot
        tangentInfo.append([minX,maxX,ylow,yhigh])

    return tangentInfo,fittedTraj



def point_to_line_dist(point, line):
    """Calculate the distance between a point and a line segment.

    To calculate the closest distance to a line segment, we first need to check
    if the point projects onto the line segment.  If it does, then we calculate
    the orthogonal distance from the point to the line.
    If the point does not project to the line segment, we calculate the
    distance to both endpoints and take the shortest distance.

    :param point: Numpy array of form [x,y], describing the point.
    :type point: numpy.core.multiarray.ndarray
    :param line: list of endpoint arrays of form [P1, P2]
    :type line: list of numpy.core.multiarray.ndarray
    :return: The minimum distance to a point.
    :rtype: float
    """
    # unit vector
    unit_line = line[1] - line[0]
    norm_unit_line = unit_line / np.linalg.norm(unit_line)

    # compute the perpendicular distance to the theoretical infinite line
    segment_dist = (
        np.linalg.norm(np.cross(line[1] - line[0], line[0] - point)) /
        np.linalg.norm(unit_line)
    )

    diff = (
        (norm_unit_line[0] * (point[0] - line[0][0])) +
        (norm_unit_line[1] * (point[1] - line[0][1]))
    )

    x_seg = (norm_unit_line[0] * diff) + line[0][0]
    y_seg = (norm_unit_line[1] * diff) + line[0][1]

    endpoint_dist = min(
        np.linalg.norm(line[0] - point),
        np.linalg.norm(line[1] - point)
    )

    # decide if the intersection point falls on the line segment
    lp1_x = line[0][0]  # line point 1 x
    lp1_y = line[0][1]  # line point 1 y
    lp2_x = line[1][0]  # line point 2 x
    lp2_y = line[1][1]  # line point 2 y
    is_betw_x = lp1_x <= x_seg <= lp2_x or lp2_x <= x_seg <= lp1_x
    is_betw_y = lp1_y <= y_seg <= lp2_y or lp2_y <= y_seg <= lp1_y
    if is_betw_x and is_betw_y:
        return segment_dist
    else:
        # if not, then return the minimum distance to the segment endpoints
        return endpoint_dist

# Calculate the confusion matrix
def CalcConfusionMatrix(predList):
    straightArray = np.array([0,0,0])
    leftTurnArray = np.array([0,0,0])
    rightTurnArray = np.array([0,0,0])

    for eachPredValues in predList:
        gtValue = eachPredValues[0]
        predValue = eachPredValues[1]
        if gtValue == straightStr:
            if predValue == straightStr:
                straightArray[0] = straightArray[0] + 1
            elif predValue == rightTurnStr:  
                straightArray[1] = straightArray[1] + 1
            elif predValue == leftTurnStr:
                straightArray[2] = straightArray[2] + 1
            else:
                print('Unknow movement predicted in the consufion matrix calculation')
                sys.exit()
        elif gtValue == leftTurnStr:
            if predValue == straightStr:
                leftTurnArray[0] = leftTurnArray[0] + 1
            elif predValue == rightTurnStr:
                leftTurnArray[1] = leftTurnArray[1] + 1
            elif predValue == leftTurnStr:
                leftTurnArray[2] = leftTurnArray[2] + 1
            else:
                print('Unknow movement predicted in the consufion matrix calculation')
                sys.exit()
        elif gtValue == rightTurnStr:
            if predValue == straightStr:
                rightTurnArray[0] = rightTurnArray[0] + 1
            elif predValue == rightTurnStr:
                rightTurnArray[1] = rightTurnArray[1] + 1
            elif predValue == leftTurnStr:
                rightTurnArray[2] = rightTurnArray[2] + 1
            else:
                print('Unknow movement predicted in the consufion matrix calculation')
                sys.exit()
        else:
            print('Unknow Groud truth movement in the consufion matrix calculation')
            sys.exit()

    # print('          Staright Left-Trun Right-Turn')
    # print('straight: '  + str(straightArray))
    # print('Left Turn: ' + str(leftTurnArray))
    # print('Right Turn: ' + str(rightTurnArray))
    finalConfusionMatrix = np.column_stack((straightArray,leftTurnArray,rightTurnArray))
    return finalConfusionMatrix

# Covert the movement float to movementStr for confusion calculation
def MovementToStr(movementFloat):
    returnStr = ''
    if(movementFloat == 0):
        returnStr = straightStr
    elif(movementFloat == 0.5):
        returnStr = leftTurnStr
    elif(movementFloat == 1.0):
        returnStr = rightTurnStr
    else:
        print('Unknown movement float in MovementToStr func!!!')
        print('Received movement float value is : ' + str(movementFloat))
        sys.exit()

    return returnStr


# Draw a rotated bounding box using cx,cy and angle with fixed boudning box
def RotateBoundingBox(im, pose, angle, color):
    theta = np.deg2rad(-angle)
    # theta = np.deg2rad(angle)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])

    cx = pose[0]
    cy = pose[1]
    T = np.array([[cx], [cy]])

    bbox = [0,0,carWidth,carHeight]
    bbox[0] = cx-(carWidth/2)
    bbox[1] = cy-(carHeight/2)

    points = np.array([[bbox[0], bbox[1]],
                        [bbox[0] + bbox[2], bbox[1]],
                        [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                        [bbox[0], bbox[1] + bbox[3]]]).T


    points = points - T
    points = np.matmul(R, points) + T
    points = points.astype(int)

    cv2.line(im, tuple(points[:,0]), tuple(points[:,1]), color, 3)
    cv2.line(im, tuple(points[:,1]), tuple(points[:,2]), color, 3)
    cv2.line(im, tuple(points[:,2]), tuple(points[:,3]), color, 3)
    cv2.line(im, tuple(points[:,3]), tuple(points[:,0]), color, 3)

    return im


class Sequence:

    def __init__(self, sequence_path):
        self.sequence_path = sequence_path

    def draw_boundingbox_rot(self, im, bbox, angle, color):
        theta = np.deg2rad(-angle)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        points = np.array([[bbox[0], bbox[1]],
                           [bbox[0] + bbox[2], bbox[1]],
                           [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                           [bbox[0], bbox[1] + bbox[3]]]).T

        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        T = np.array([[cx], [cy]])

        points = points - T
        points = np.matmul(R, points) + T
        points = points.astype(int)

        cv2.line(im, tuple(points[:, 0]), tuple(points[:, 1]), color, 3)
        cv2.line(im, tuple(points[:, 1]), tuple(points[:, 2]), color, 3)
        cv2.line(im, tuple(points[:, 2]), tuple(points[:, 3]), color, 3)
        cv2.line(im, tuple(points[:, 3]), tuple(points[:, 0]), color, 3)

        return im

    def load_annotations(self, annotation_path):
        if (os.path.exists(annotation_path)):
            f = open(annotation_path)
            self.annotations = json.load(f)
        else:
            self.annotations = None

    def load_timestamp(self, timestamp_path):
        genfromtxt = np.genfromtxt(
            timestamp_path, dtype=(str, int, str, float))
        timestamps = {'frame': [], 'time': []}
        for line in genfromtxt:
            timestamps['frame'].append(line[1])
            timestamps['time'].append(line[3])
        return timestamps

    # get the ids from radar ids
    def get_frame_ids(self, timestamps_radar,):

        self.total_nb_frames_radar = len(timestamps_radar['frame'])
        radar_ids = np.array(np.arange(1,self.total_nb_frames_radar+1))
        time = timestamps_radar['time']
        return radar_ids

    def load_sequence(self,sequence_path):
        # get all time stamps
        timestamps_radar_path = os.path.join(sequence_path, 'Navtech_Cartesian.txt')
        timestamps_radar = self.load_timestamp(timestamps_radar_path)

        radar_ids = self.get_frame_ids(timestamps_radar)
        self.total_nb_frames_radar = len(timestamps_radar['frame'])

        self.radar_ids = radar_ids

    def play(self,sequence_path,idOffset):

        global globalRadarDisplayImage,turn,straight

        for i in range(1, self.total_nb_frames_radar - 1):
            # get correct frames
            radar_id = int(self.radar_ids[i])
            radar_cartesian_path = os.path.join(sequence_path, 'Navtech_Cartesian', str_format.format(radar_id) + '.png')

            radar_cartesian = cv2.imread(radar_cartesian_path)
            if(i == 1 and idOffset == 0):
                globalRadarDisplayImage = cv2.imread(radar_cartesian_path) #radar_cartesian

            if (self.annotations != None):
                for object in self.annotations:
                    if (object['bboxes'][i]):
                        if (object['deleted'][i] == 0):
                            if (object['visible'][i] == 'visible'):
                                bbox = object['bboxes'][i]['position']
                                angle = object['bboxes'][i]['rotation']
                                color = object['color']
                                radar_cartesian = self.draw_boundingbox_rot(radar_cartesian, bbox, angle, (0,255,0))
                                cx = int(bbox[0] + bbox[2]/2)
                                cy = int(bbox[1] + bbox[3]/2)
                                radar_cartesian = cv2.putText(radar_cartesian, str(object['id']+idOffset), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                                #radar_cartesian = cv2.putText(radar_cartesian, str(cx), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
                                #check if previous frame exists
                                if(len(object['bboxes'][i-1])>0):
                                    prevBox = object['bboxes'][i-1]['position']
                                    prevCx = int(prevBox[0] + prevBox[2]/2)
                                    prevCy = int(prevBox[1] + prevBox[3]/2)
                                    # check if turn or straight trajectory. Travers the list from the back and pick the last position.
                                    # If the position is > 580 its a turn
                                    lastLocation = 0
                                    turnBool = False
                                    for j in reversed(object['bboxes']):
                                        if isinstance(j, dict):
                                            lastLocation = j['position'][0]
                                            break

                                    if (lastLocation>580):
                                        turnBool = True
                                    #globalRadarDisplayImage = cv2.circle(globalRadarDisplayImage, (cx,cy), 2, (0,0,255))
                                    if(turnBool):
                                        globalRadarDisplayImage = cv2.line(globalRadarDisplayImage, (cx,cy), (prevCx,prevCy), (0,255,0), 2)
                                        if (object['id']+idOffset) not in processedObjectIds:
                                            processedObjectIds.append((object['id']+idOffset))
                                            turn = turn + 1
                                    else:
                                        globalRadarDisplayImage = cv2.line(globalRadarDisplayImage, (cx,cy), (prevCx,prevCy), (0,255,0), 2)
                                        if (object['id']+idOffset) not in processedObjectIds:
                                            processedObjectIds.append((object['id']+idOffset))
                                            straight = straight + 1

            turnStr = 'Turn : ' + str (turn)
            straightStr = 'Straight : ' + str (straight)
            radar_cartesian = cv2.putText(radar_cartesian, turnStr, (800,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            radar_cartesian = cv2.putText(radar_cartesian, straightStr, (800,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

            # Check the regions
            globalRadarDisplayImage = cv2.circle(globalRadarDisplayImage, (300,450), 40, (0,0,255), 2)
            globalRadarDisplayImage = cv2.circle(globalRadarDisplayImage, (480,300), 40, (0,0,255), 2)
            globalRadarDisplayImage = cv2.circle(globalRadarDisplayImage, (760,435), 40, (0,0,255), 2)
            globalRadarDisplayImage = cv2.circle(globalRadarDisplayImage, (595,675), 40, (0,0,255), 2)
            globalRadarDisplayImage = cv2.circle(globalRadarDisplayImage, (605,1070), 40, (0,0,255), 2)

            finalImage = np.hstack((radar_cartesian,globalRadarDisplayImage))

            cv2.imshow('image', finalImage)
            cv2.waitKey(10)

        return finalImage

    def playByVehicle(self,sequence_path,idOffset):

        global globalRadarDisplayImage,globalRadarSmoothTraj, imageCount, resizedMap, ignoredCars   #, mapImage

        # One map to see all trajs
        # mapImage = cv2.imread(mapImagePath)

        if (self.annotations != None):
            for anotationIndex, object in enumerate(self.annotations):
                if(display == 1):
                    # reload the image for trajectory visualization
                    intitalImagePath = os.path.join(sequence_path, 'Navtech_Cartesian', str_format.format(1) + '.png')
                    globalRadarDisplayImage = cv2.imread(intitalImagePath)
                    globalRadarSmoothTraj = cv2.imread(intitalImagePath)

                    # # Draw Lane lines
                    # This one is for new map for every vehicle
                    mapImage = cv2.imread(mapImagePath)
                    for laneIndex,eacLine in enumerate(laneLines):
                        mapImage = cv2.line(mapImage, (eacLine[0], eacLine[1]), (eacLine[2], eacLine[3]), (255,0,0), 2)

                    # Write the map info
                    mapImage = cv2.putText(mapImage, 'Red : Straight', (750,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                    mapImage = cv2.putText(mapImage, 'Green : Left Turn', (750,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                    mapImage = cv2.putText(mapImage, 'Blue : Right Turn', (750,140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                    mapImage = cv2.putText(mapImage, 'Yellow : Nearset Lane', (750,170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)

                centrePoses = []
                allBoundingBoxes = object['bboxes']
                allVisibleParam = object['visible']
                allDeleteParam = object['deleted']
                vehicleID = object['id'] + idOffset
                currentVehicleAngleList = []

                # Grab the surrounding bounding boxes
                surroundingCentrePosesDict = dict()
                allSurroundingBoundingBoxes = []
                allSurroundingVisibleParam = []
                allSurroundingDeleteParam = []
                allSurroundingIds = []
                for surroundingIndex, surroundingObject in enumerate(self.annotations):
                    # Ignore the target car as surrouding
                    if(surroundingIndex == anotationIndex):
                        continue
                    # Extract the surrouding car BB boxes and other info
                    surroundingBoundingBoxes = surroundingObject['bboxes']
                    surroundingVisibleParam = surroundingObject['visible']
                    surroundingDeleteParam = surroundingObject['deleted']
                    surroundingID = surroundingObject['id'] + idOffset
                    # Append the surrouding car info into the all surroudning car info list
                    allSurroundingBoundingBoxes.append(surroundingBoundingBoxes)
                    allSurroundingVisibleParam.append(surroundingVisibleParam)
                    allSurroundingDeleteParam.append(surroundingDeleteParam)
                    allSurroundingIds.append(surroundingID)
                    # Create a key in surroundingCentrePosesDict with the surrounding ID
                    surroundingCentrePosesDict[str(surroundingID)] = []

                for idx,eachBbbox in enumerate(allBoundingBoxes):
                    if (eachBbbox):
                        if ((allDeleteParam[idx] == 0) and (allVisibleParam[idx] == 'visible')):
                            # Read the current Frame
                            if(display == 1):
                                radar_id = int(self.radar_ids[idx])
                                radar_cartesian_path = os.path.join(sequence_path, 'Navtech_Cartesian', str_format.format(radar_id) + '.png')
                                radar_cartesian = cv2.imread(radar_cartesian_path)

                            # Gather info related to the target car
                            bbox = eachBbbox['position']
                            angle = eachBbbox['rotation']
                            currentVehicleAngleList.append(angle)
                            cx = int(bbox[0] + bbox[2]/2)
                            cy = int(bbox[1] + bbox[3]/2)
                            centrePoses.append([cx,cy,angle])

                            # Gather info related to surrounding cars and populate the dict car
                            for eachSurroundingBoxList,eachSurroundingVisibleList,eachSurroundingDeleteList,eachSurroundingId in zip(allSurroundingBoundingBoxes,allSurroundingVisibleParam,allSurroundingDeleteParam,allSurroundingIds):
                                sameFrameBBox = eachSurroundingBoxList[idx]
                                # sameFrameVisible = eachSurroundingVisibleList[idx]
                                sameFrameDelete = eachSurroundingDeleteList[idx]
                                sameFrameID = eachSurroundingId
                                if (sameFrameBBox):
                                    # if ((sameFrameDelete == 0) and (sameFrameVisible == 'visible')):
                                    if (sameFrameDelete == 0):
                                        # Extract the pose info
                                        surroundingBbox = sameFrameBBox['position']
                                        surroundingAngle = sameFrameBBox['rotation']
                                        # surroundingCurrentVehicleAngleList.append(surroundingAngle) have to prep another list of list for angle
                                        surroundingCx = int(surroundingBbox[0] + surroundingBbox[2]/2)
                                        surroundingCy = int(surroundingBbox[1] + surroundingBbox[3]/2)
                                        surroundingCentrePosesDict[str(sameFrameID)].append([surroundingCx,surroundingCy,surroundingAngle])
                                        # radar_cartesian = self.draw_boundingbox_rot(radar_cartesian, surroundingBbox, surroundingAngle, (0,0,255))
                                    else:
                                        # If the car doesn't exist in the current frame append -1 for future use
                                        surroundingCentrePosesDict[str(sameFrameID)].append([-1,-1,-1])
                                else:
                                    # If the car doesn't exist in the current frame append -1 for future use
                                    surroundingCentrePosesDict[str(sameFrameID)].append([-1,-1,-1])

                            if(display == 1):
                                # Draw the target vehicle trajectory
                                if(len(centrePoses) > 2):
                                    for kdx in range(1,len(centrePoses)):
                                        prevCx = centrePoses[kdx-1][0]
                                        prevCy = centrePoses[kdx-1][1]
                                        Cx = centrePoses[kdx][0]
                                        Cy = centrePoses[kdx][1]
                                        radar_cartesian = cv2.line(radar_cartesian, (prevCx, prevCy), (Cx, Cy), (0,0,255), 3)

                                # Draw the surrounding trajectories
                                for surroundingKey, surroundingValue in surroundingCentrePosesDict.items():
                                    # No need to check the length of surroudning vehicle list is greater than 2 or not as the length is same as centrePoses
                                    for ldx in range(1,len(surroundingValue)):
                                        prevCx = surroundingValue[ldx-1][0]
                                        prevCy = surroundingValue[ldx-1][1]
                                        Cx = surroundingValue[ldx][0]
                                        Cy = surroundingValue[ldx][1]
                                        # Draw the line if the values are not equal to -1 as which means there are no car
                                        if(prevCx != -1 and prevCy != -1 and Cx != -1 and Cy != -1):
                                            radar_cartesian = cv2.line(radar_cartesian, (prevCx, prevCy), (Cx, Cy), (0,0,255), 3)


                                # Based on the angle change the color
                                angleColor = (255,0,0)
                                # Check the current angle with each angle with margin
                                for eachStraightAngle in straightAngles:
                                    if(angle > eachStraightAngle-4 and angle < eachStraightAngle+4):
                                        angleColor = (0,255,0)
                                        break

                                radar_cartesian = self.draw_boundingbox_rot(radar_cartesian, bbox, angle, angleColor)

                                # Display each vehicle trajectory along with the ID
                                radar_cartesian = cv2.putText(radar_cartesian, str(vehicleID), (int(cx),int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

                                # Display the angle
                                # radar_cartesian = cv2.putText(radar_cartesian, str(angle), (int(cx+20),int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)

                                finalImage = np.hstack((radar_cartesian,mapImage))

                                cv2.imshow('image', finalImage)
                                # cv2.imshow('smooth', finalImageTraj)
                                cv2.waitKey(10)

                                imageCount = imageCount+1
                                # imagePath = imageSaveFolder + str(imageCount) + '.png'
                                # cv2.imwrite(imagePath, finalImage)


                # All the target and surrounding cars dict lists are populated
                # After the population the length of the centerPoses (which is the target car) and length of each list in the surrounding dict
                # should be same. So adding the check

                TargetVehiclePoseLength = len(centrePoses)
                for surroundingKey, surroundingValue in surroundingCentrePosesDict.items():
                    surroundingPoseLength = len(surroundingValue)
                    if(surroundingPoseLength != TargetVehiclePoseLength):
                        print('Target vehicle and surroudning vehicle pose length mismatch!!!')
                        print('Target Vehicle ' + str(vehicleID) + ', pose length ' + str(TargetVehiclePoseLength))
                        print('Surrounding Vehicle ' + surroundingKey + ', pose length ' + str(surroundingPoseLength))
                        sys.exit()

                # Smooth the jerky trajectories using particle Filter
                if (TargetVehiclePoseLength < 3):
                    continue

                targetSmoothTraj,smoothDroppedTraj = SmoothTrajcetoryGeneration(centrePoses)

                # Remove the dict entires wehre the whole traj is (-1,-1)
                # Identify the accecpted Keys
                acceptedKeyList = []
                for acceptKey, eachPoseCheckList in surroundingCentrePosesDict.items():
                    for eachCheckPose in eachPoseCheckList:
                        if(eachCheckPose[0] != -1 and eachCheckPose[1] != -1):
                            acceptedKeyList.append(acceptKey)
                            break

                # Identify the rejected Keys
                rejectedKeys = []
                for eachSurroudningDictKey in surroundingCentrePosesDict.keys():
                    if(eachSurroudningDictKey not in acceptedKeyList):
                        rejectedKeys.append(eachSurroudningDictKey)

                # Remove all the dict entries for which the key is not in the acceptedKey list
                for eachRejectKey in rejectedKeys:
                    del surroundingCentrePosesDict[eachRejectKey]

                # # # This is for smoothing the surrounding trajectory. Ignored at the moement
                # # for surroundingKey, eachJerkySurrounding in surroundingCentrePosesDict.items():
                # #     print('test')
                #
                #

                # Prepeare a dictionary for surrouding vehicleID and orginalLaneIndex for later use
                orignalLaneIndexID = dict()
                for surroundingKey, eachSurroudingPoseList in surroundingCentrePosesDict.items():
                    # Loop through the surrounding pose list till it's a value not equal to -1
                    for eachSurroungPoseItem in eachSurroudingPoseList:
                        if(eachSurroungPoseItem[0] != -1 and eachSurroungPoseItem[1] != -1):
                            surroundingNearestLaneIndex,surroudingLowestDist = CalculateNearestLaneAndDist(eachSurroungPoseItem)
                            orignalLaneIndexID[surroundingKey] = surroundingNearestLaneIndex
                            break

                # Afte the originalLaneIndexID dict prep both the original dict and the laneIndex dict should have the same keys
                orignalLaneIndexIDKeys = orignalLaneIndexID.keys()
                surroundingCentrePosesDictkeys = surroundingCentrePosesDict.keys()

                if(Counter(orignalLaneIndexIDKeys) != Counter(surroundingCentrePosesDictkeys)):
                    print('Original surrounding dict and nearestLaneIndex dict are not the same!!!')
                    print('orignalLaneIndexIDKeys are : ')
                    print(orignalLaneIndexIDKeys)
                    print('surroundingCentrePosesDictkeys are : ')
                    print(surroundingCentrePosesDictkeys)
                    sys.exit()

                # If the length of center poses is zero no need to write the image and write the vehicle ID
                if(len(centrePoses) == 0):
                    print('Zero traj lenght vehicle ID ' + str(vehicleID))
                    continue

                if(display == 1):
                    # Write the vehicle ID the traj image
                    mapImage = cv2.putText(mapImage, 'Red : Straight', (750,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                    mapImage = cv2.putText(mapImage, 'Green : Left Turn', (750,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                    mapImage = cv2.putText(mapImage, 'Blue : Right Turn', (750,140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                    mapImage = cv2.putText(mapImage, 'Yellow : Nearset Lane', (750,170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)

                    globalRadarDisplayImage = cv2.putText(globalRadarDisplayImage, 'Red : Straight', (750,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                    globalRadarDisplayImage = cv2.putText(globalRadarDisplayImage, 'Green : Turn', (750,130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)



                # Ignor the cars originating at 1,2 check only with the first location
                # Estimate the nearest lane index and calculate the distance
                nearestLaneIndex,lowestDist = CalculateNearestLaneAndDist(targetSmoothTraj[0])

                if(nearestLaneIndex == 1 or nearestLaneIndex == 2):
                    print('Another car ignored!!!')
                    ignoredCars = ignoredCars + 1
                    continue


                # Draw the color coded trajectory
                lineThickness = 3
                originLaneIndex = -1
                localInfoList = []
                localInfoListDropped = []
                # Bool to check if the trajectory has a turn
                trajTurnBool = False

                # Take the first and last location with the nearest lane to estimate the menuver
                firstPose = targetSmoothTraj[3]
                lastPose = targetSmoothTraj[-1]

                firstNeasestLane, firstLaneDist = CalculateNearestLaneAndDist(firstPose)
                lastNeasestLane, lastLaneDist = CalculateNearestLaneAndDist(lastPose)

                if(firstNeasestLane == lastNeasestLane):
                    print('Super short traj!!!')
                    ignoredCars = ignoredCars + 1
                    continue

                movementInfo,movementInfoFloat = LaneBasedManeuver(firstNeasestLane,lastNeasestLane)


                # # # collect manauver info for each car to check majority
                # # maneuverList = []
                for trajIndex in range(3,TargetVehiclePoseLength):
                    ###################################################
                    # Target vehicle extraction without any frame drop
                    ##################################################
                    # Extract the current and prev poses
                    currentPose = targetSmoothTraj[trajIndex]
                    prev2Pose = targetSmoothTraj[trajIndex-2]
                    prev3Pose = targetSmoothTraj[trajIndex-3]
                    annotateAngle = prev2Pose[2] # As prev2Pose is added as pose
                    # Calculate the tangent for angle
                    prev3Angle = math.degrees(math.atan2((prev3Pose[1]-currentPose[1]),(prev3Pose[0]-currentPose[0])))
                    if prev3Angle < 0 : prev3Angle = prev3Angle + 360

                    # Estimate the nearest lane index and calculate the distance
                    nearestLaneIndex,lowestDist = CalculateNearestLaneAndDist(currentPose)

                    # Update the origin lane index for left/right estimtion
                    if(trajIndex == 3):
                        originLaneIndex = nearestLaneIndex

                    # Estimate the Turn or straight movement based on the angle
                    # # # movementInfo,movementInfoFloat = CalculateManeuverInfo(prev3Angle,originLaneIndex)

                    # # Indetify the main maneuver
                    # # maneuverList.append(movementInfoFloat)

                    # Based on the estimate the Turn or straight movement set the flag as this will be added multiple times
                    if((trajTurnBool == False) and (movementInfo == 'Left' or movementInfo == 'Right')):
                        trajTurnBool = True


                    # Estimate the soomthed instantenious velocity
                    smoothVx = prev3Pose[0] - prev2Pose[0]
                    smoothVy = prev3Pose[1] - prev2Pose[1]

                    # If the velocity is zero change it to 2 to avoid confusion from the padding
                    if(smoothVx == 0):
                        smoothVx = 2
                    if(smoothVy == 0):
                        smoothVy = 2

                    ###################################################
                    # Target vehicle extraction with frame drop
                    ##################################################
                    # Extract the current and prev poses
                    currentPoseDroppped = smoothDroppedTraj[trajIndex]
                    prev2PoseDroppped = smoothDroppedTraj[trajIndex-2]
                    prev3PoseDroppped = smoothDroppedTraj[trajIndex-3]
                    annotateAngleDropped = prev2PoseDroppped[2] # As prev2Pose is added as pose
                    # Calculate the tangent for angle
                    prev3AngleDropped = math.degrees(math.atan2((prev3PoseDroppped[1]-currentPoseDroppped[1]),(prev3PoseDroppped[0]-currentPoseDroppped[0])))
                    if prev3AngleDropped < 0 : prev3AngleDropped = prev3AngleDropped + 360

                    # Estimate the nearest lane index and calculate the distance
                    nearestLaneIndexDropped,lowestDistDropped = CalculateNearestLaneAndDist(currentPoseDroppped)

                    # Update the origin lane index for left/right estimtion
                    if(trajIndex == 3):
                        originLaneIndexDropped = nearestLaneIndexDropped

                    # Estimate the Turn or straight movement based on the angle
                    movementInfoDropped,movementInfoFloatDropped = CalculateManeuverInfo(prev3AngleDropped,originLaneIndexDropped)

                    # # Indetify the main maneuver
                    # # maneuverList.append(movementInfoFloat)

                    # # # Based on the estimate the Turn or straight movement set the flag as this will be added multiple times
                    # # if((trajTurnBool == False) and (movementInfo == 'Left' or movementInfo == 'Right')):
                    # #     trajTurnBool = True


                    # Estimate the soomthed instantenious velocity
                    smoothVxDropped = prev3PoseDroppped[0] - prev2PoseDroppped[0]
                    smoothVyDropped = prev3PoseDroppped[1] - prev2PoseDroppped[1]

                    # If the velocity is zero change it to 2 to avoid confusion from the padding
                    if(smoothVxDropped == 0):
                        smoothVxDropped = 2
                    if(smoothVyDropped == 0):
                        smoothVyDropped = 2



                    # Add the vehicle current information in the list both in the normal and dropped list
                    # currentvehicleInfoList = [prev2Pose[0], prev2Pose[1], smoothVx, smoothVy, lowestDist, nearestLaneIndex, movementInfoFloat, prev3Angle,float(vehicleID)]
                    currentvehicleInfoList = [prev2Pose[0], prev2Pose[1], smoothVx, smoothVy, lowestDist, nearestLaneIndex, movementInfoFloat, annotateAngle,float(vehicleID)]
                    currentvehicleInfoListDropped = [prev2PoseDroppped[0], prev2PoseDroppped[1], smoothVxDropped, smoothVyDropped, lowestDistDropped, nearestLaneIndexDropped, movementInfoFloatDropped, annotateAngleDropped,float(vehicleID)]
                    # currentvehicleInfoList = currentInfo

                    # Add the information regarding the surrounding cars
                    # Extract the corresponding positions from the surrounding car list
                    currentSurroudingPoseList = []
                    for surroundingKey, eachJerkySurrounding in surroundingCentrePosesDict.items():
                        currentSurroundingPose = eachJerkySurrounding[trajIndex]
                        prev2SurroundingPose = eachJerkySurrounding[trajIndex-2]
                        prev3SurroundingPose = eachJerkySurrounding[trajIndex-3]
                        surroudingAnnotateAngle = prev2SurroundingPose[2]
                        # Check only one of the X/Y pose for -1
                        if(currentSurroundingPose[0] != -1 and prev2SurroundingPose[0] != -1 and prev3SurroundingPose[0] != -1):
                            currentSurroundingPoseInfo = [currentSurroundingPose,prev2SurroundingPose,prev3SurroundingPose,surroundingKey]
                            currentSurroudingPoseList.append(currentSurroundingPoseInfo)

                    # Check the number of surrounding vehicles and compare with the mentioned surroundingCarCount
                    # Based on the missing number append zeros.
                    currentSurroundingCarCount = len(currentSurroudingPoseList)
                    paddingCount = surroudingCarCounts - currentSurroundingCarCount
                    if(paddingCount < 0):
                        paddingCount = 0

                    # Calculate distance from each surrounding car and pick the nearest surroundingCarCount number of cars
                    distanceAndIndexlist = []
                    for distIndex,eachSurroundingPose in enumerate(currentSurroudingPoseList):
                        currentSurroundingX = eachSurroundingPose[0][0] # First entry is the currentPose and First entry in the poseX in currentSurroudingPoseList
                        currentSurroundingY = eachSurroundingPose[0][1] # First entry is the currentPose and Second entry in the poseY in currentSurroudingPoseList
                        targetPoseX = currentPose[0]
                        targetPoseY = currentPose[1]
                        # Calculate the Euclidian distance to pick the nearest cars
                        ##########surroudingDist = math.sqrt(((currentSurroundingX-currentSurroundingY)**2) + ((currentSurroundingY-targetPoseY)**2))
                        ##### once all done change this and retrain ###########
                        surroudingDist = math.sqrt(((currentSurroundingX-targetPoseX)**2) + ((currentSurroundingY-targetPoseY)**2))#####
                        ###########################################################################
                        distanceAndIndexlist.append([distIndex,surroudingDist])

                    # Sort the list based on distance and gather the lowest indexes
                    distanceAndIndexlist = sorted(distanceAndIndexlist,key=lambda x: x[1])
                    distanceAndIndexArray = np.array(distanceAndIndexlist)
                    # If there are zero surrouding car ignore the whole surroudning process and just do the padding (***** do the max dist  check ****)
                    if(currentSurroundingCarCount != 0):
                        if(currentSurroundingCarCount >= surroudingCarCounts):
                            releventSurroundingIndexes = distanceAndIndexArray[0:surroudingCarCounts,0:1]
                            releventSurroundingDists = distanceAndIndexArray[0:surroudingCarCounts,1:2]
                        else:
                            releventSurroundingIndexes = distanceAndIndexArray[:,0:1]
                            releventSurroundingDists = distanceAndIndexArray[:,1:2]

                        # Extract the input information for the relevant surrounding vehicles and extend to current input list
                        for eachRelevenetIndex,eachRelevenetDist in zip(releventSurroundingIndexes,releventSurroundingDists):
                            releventSurroudingCarInfo = currentSurroudingPoseList[int(eachRelevenetIndex[0])]
                            surroundingCurrentPose = releventSurroudingCarInfo[0] # As current info is the first item in currentSurroundingPoseInfo list
                            surroundingPrev2Pose = releventSurroudingCarInfo[1] # As Prev2 info is the second item in currentSurroundingPoseInfo list
                            surroundingPrev3Pose = releventSurroudingCarInfo[2] # As Prev3 info is the third item in currentSurroundingPoseInfo list
                            surroundingCarID = releventSurroudingCarInfo[3] # As surroudning vehicle Id is the forth item in currentSurroundingPoseInfo list
                            surroudingAnnotateAngle = surroundingPrev2Pose[2] # As we are appending the prev2Pose

                            # Extract the other infos regarding this surronding car
                            # Calculate the tangent for angle
                            surroundingAngle = math.degrees(math.atan2((surroundingPrev3Pose[1]-surroundingCurrentPose[1]),(surroundingPrev3Pose[0]-surroundingCurrentPose[0])))
                            if surroundingAngle < 0 : surroundingAngle = surroundingAngle + 360
                            # Estimate the nearest lane index and calculate the distance
                            surroudningNearestLaneIndex,surroundingLowestDist = CalculateNearestLaneAndDist(surroundingCurrentPose)

                            # Estimate the Turn or straight movement based on the angle
                            # Get the surroudning car's originalLane index
                            surroudingOriginalLaneIndex = orignalLaneIndexID[surroundingCarID]
                            surroudingMovementInfo,surroudingMovementInfoFloat = CalculateManeuverInfo(surroundingAngle,surroudingOriginalLaneIndex)

                            # Estimate the soomthed instantenious velocity for the surrounding car
                            smoothSurroundingVx = surroundingPrev3Pose[0] - surroundingPrev2Pose[0]
                            smoothSurroundingVy = surroundingPrev3Pose[1] - surroundingPrev2Pose[1]

                            # If the velocity is zero change it to 2 to avoid confusion from the padding
                            if(smoothSurroundingVx == 0):
                                smoothSurroundingVx = 2
                            if(smoothSurroundingVy == 0):
                                smoothSurroundingVy = 2

                            # Add the surrounding vehicle current information in the list
                            # surroundingPrev2Pose is added as for the target vehicle prev2Pose pose is added
                            # currentSurroundingInfo = [surroundingPrev2Pose[0], surroundingPrev2Pose[1], smoothSurroundingVx, smoothSurroundingVy, surroundingLowestDist, surroudningNearestLaneIndex, surroudingMovementInfoFloat, surroundingAngle,float(surroundingCarID)]
                            currentSurroundingInfo = [surroundingPrev2Pose[0], surroundingPrev2Pose[1], smoothSurroundingVx, smoothSurroundingVy, surroundingLowestDist, surroudningNearestLaneIndex, surroudingMovementInfoFloat, surroudingAnnotateAngle,float(surroundingCarID)]

                            # Check the threshold distance and extend if it is with in the threshold
                            currentDist = eachRelevenetDist[0]
                            if(currentDist < maxSurroudingDist):
                                currentvehicleInfoList.extend(currentSurroundingInfo)
                                currentvehicleInfoListDropped.extend(currentSurroundingInfo)
                            else:
                                currentvehicleInfoList.extend(inputFeatureZeroPad)
                                currentvehicleInfoListDropped.extend(inputFeatureZeroPad)

                    # Add the remaining zero padding based on the padding count
                    for paddingVal in range(0,paddingCount):
                        currentvehicleInfoList.extend(inputFeatureZeroPad)
                        currentvehicleInfoListDropped.extend(inputFeatureZeroPad)

                    # Without Dropped
                    # After this padding the vehicle feature length should be surroundingCarCount*inputFeatureCount
                    vehicleInfoCount = len(currentvehicleInfoList)
                    if(vehicleInfoCount != (surroudingCarCounts+1)*inputFeature):
                        print('Input fetaure count is not matching for without dropped!!!')
                        print('Expected feature count : ' + str((surroudingCarCounts+1)*inputFeature))
                        print('Received feature count : ' + str(vehicleInfoCount))
                        sys.exit()

                    # With Dropped
                    # After this padding the vehicle feature length should be surroundingCarCount*inputFeatureCount
                    vehicleInfoCountDropped = len(currentvehicleInfoListDropped)
                    if(vehicleInfoCountDropped != (surroudingCarCounts+1)*inputFeature):
                        print('Input fetaure count is not matching fr with dropped!!!')
                        print('Expected feature count : ' + str((surroudingCarCounts+1)*inputFeature))
                        print('Received feature count : ' + str(vehicleInfoCountDropped))
                        sys.exit()

                    # Add the currentvehicleInfoList in the localInfoList (final info list for the current vehicle)
                    # Both for the with and without frame dropped
                    localInfoList.append(currentvehicleInfoList)
                    localInfoListDropped.append(currentvehicleInfoListDropped)


                    if(display == 1):
                        # Draw Lane lines
                        for laneIndex,eacLine in enumerate(laneLines):
                            mapImage = cv2.putText(mapImage, str(laneIndex), (int((eacLine[0]+eacLine[2])/2),int((eacLine[1]+eacLine[3])/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                            if(laneIndex == nearestLaneIndex):
                                mapImage = cv2.line(mapImage, (eacLine[0], eacLine[1]), (eacLine[2], eacLine[3]), (0,255,255), 2)
                            else:
                                mapImage = cv2.line(mapImage, (eacLine[0], eacLine[1]), (eacLine[2], eacLine[3]), (255,0,0), 2)

                        # Decide the color based on the movement
                        if(movementInfo == 'Straight'):
                            angleColor = (0,0,255)
                        elif(movementInfo == 'Left'):
                            angleColor = (0,255,0)
                        elif(movementInfo == 'Right'):
                            angleColor = (255,0,0)
                        else:
                            print('Unknown Movement info!!!')
                            sys.exit()

                        # Draw the target vehicle trajectory
                        mapImage = cv2.line(mapImage, (prev3Pose[0], prev3Pose[1]), (prev2Pose[0], prev2Pose[1]), angleColor, lineThickness)
                        globalRadarDisplayImage = cv2.line(globalRadarDisplayImage, (prev3Pose[0], prev3Pose[1]), (prev2Pose[0], prev2Pose[1]), angleColor, lineThickness)

                        # Draw the relevent surrounding car info
                        for eachSurroundingDrawPose in currentSurroudingPoseList:
                            surroundingPrev2Pose = eachSurroundingDrawPose[1] # Second entry is the prev2 in currentSurroudingPoseList
                            surroundingPrev3Pose = eachSurroundingDrawPose[2] # Third entry is the prev3 in currentSurroudingPoseList
                            # Draw the corresponding surrounding car traj line
                            mapImage = cv2.line(mapImage, (surroundingPrev3Pose[0], surroundingPrev3Pose[1]), (surroundingPrev2Pose[0], surroundingPrev2Pose[1]), angleColor, lineThickness)
                            globalRadarDisplayImage = cv2.line(globalRadarDisplayImage, (surroundingPrev3Pose[0], surroundingPrev3Pose[1]), (surroundingPrev2Pose[0], surroundingPrev2Pose[1]), angleColor, lineThickness)

                        finalImage = np.hstack((globalRadarDisplayImage,mapImage))

                        cv2.imshow('image', finalImage)
                        cv2.waitKey(10)

                        imageCount = imageCount+1
                        # imagePath = imageSaveFolder + str(imageCount) + '.png'
                        # cv2.imwrite(imagePath, finalImage)


                # # Indetify the main maneuver
                # # # # # Check the main maneuver 0.5 -> left turn and 1.0 -> right turn
                # # # # leftTurnCount = maneuverList.count(0.5)
                # # # # rightTurnCount = maneuverList.count(1.0)
                # # # # mainManeuver = 'STRAIGHT!!!'
                # # # # # Check if there is occurance of turn
                # # # # if(leftTurnCount > 5 or rightTurnCount > 5):
                # # # #     if(leftTurnCount > rightTurnCount):
                # # # #         mainManeuver = 'LEFT TURN!!!'
                # # # #     else:
                # # # #         mainManeuver = 'RIGHT TURN!!!'

                # # # # print('****************************')
                # # # # print('LAST MANEUVER IS ' + mainManeuver)
                # # # # print('****************************')
                # # # # time.sleep(2.0)


                # Add the current vehicle list in the final list
                # If the "currentvehicleInfoList" is not empty then add the local list in the main list
                if(len(localInfoList) > 0):
                    # Add the local info once for all type trajectories in the Without frame drop main list
                    carInfoList.append(localInfoList)
                    # Add the local info once for all type trajectories in the With frame drop main list
                    carInfoListDropped.append(localInfoListDropped)

                    # Dont worry about making multiple copies as this is not for training
                    # If the trajectory has turn add multiple times to balance the class
                    if(trajTurnBool == True and trainOrTest == trainStr):
                        carInfoList.append(localInfoList)
                        carInfoList.append(localInfoList)
                        carInfoList.append(localInfoList)

# Prepare the input, decoderInput and output data
def CheckValueNormalized(val):
    retBool = True
    if(val > 1.0 or val < 0.0):
        retBool = False
    return retBool


# Prepare the input, decoderInput and output data
def DataPrep():

    # Access the global min max values
    # Global Min max for X/Y poses
    global globalMaxXPose, globalMaxYPose, globalMinXPose, globalMinYPose
    # Global Min max for Velocity
    global globalMaxXVelocity, globalMaxYVelocity, globalMinXVelocity, globalMinYVelocity
    # Global Min max for Lane Dist
    global globalMinLaneDist, globalMaxLaneDist

    # Initialize train the input output list
    xTrainList = []
    decoderTrainList = []
    YClassTrainList = []
    YVelTrainList = []
    YPoseTrainList = []

    # Initialize validation the input output list
    xValList = []
    decoderValList = []
    YClassValList = []
    YVelValList = []
    YPoseValList = []

    # For min max at the moment just use the not dropped one
    # For each item in the main list get the min max values for later normlization
    for eachCar in carInfoList:

        eachCarArray = np.array(eachCar)

        for eachSurroundingCount in range(0,surroudingCarCounts+1):

            # Calculate the min max for the current vehicle
            # Min max for poses
            maxXPose = max(eachCarArray[:,(eachSurroundingCount*inputFeature)+poseXIndex])
            maxYPose = max(eachCarArray[:,(eachSurroundingCount*inputFeature)+poseYIndex])
            minXPose = min(eachCarArray[:,(eachSurroundingCount*inputFeature)+poseXIndex])
            minYPose = min(eachCarArray[:,(eachSurroundingCount*inputFeature)+poseYIndex])

            # Min max for Velocity
            maxXVelocity = max(eachCarArray[:,(eachSurroundingCount*inputFeature)+velXIndex])
            maxYVelocity = max(eachCarArray[:,(eachSurroundingCount*inputFeature)+velYIndex])
            minXVelocity = min(eachCarArray[:,(eachSurroundingCount*inputFeature)+velXIndex])
            minYVelocity = min(eachCarArray[:,(eachSurroundingCount*inputFeature)+velYIndex])

            minLaneDist = min(eachCarArray[:,(eachSurroundingCount*inputFeature)+laneDistIndex])
            maxLaneDist = max(eachCarArray[:,(eachSurroundingCount*inputFeature)+laneDistIndex])

            # Update the global pose  min max
            if(maxXPose > globalMaxXPose):
                globalMaxXPose = maxXPose
            if(maxYPose > globalMaxYPose):
                globalMaxYPose = maxYPose
            if(minXPose < globalMinXPose):
                globalMinXPose = minXPose
            if(minYPose < globalMinYPose):
                globalMinYPose = minYPose

            # Update the global velocity min max
            if(maxXVelocity > globalMaxXVelocity):
                globalMaxXVelocity = maxXVelocity
            if(maxYVelocity > globalMaxYVelocity):
                globalMaxYVelocity = maxYVelocity
            if(minXVelocity < globalMinXVelocity):
                globalMinXVelocity = minXVelocity
            if(minYVelocity < globalMinYVelocity):
                globalMinYVelocity = minYVelocity

            # Update the global lane distance
            if(maxLaneDist > globalMaxLaneDist):
                globalMaxLaneDist = maxLaneDist
            if(minLaneDist < globalMinLaneDist):
                globalMinLaneDist = minLaneDist

    for eachTrainCar in carInfoList:
        currentVehicleLength = len(eachTrainCar)
        # Pick the current target vehicle ID to sepereate the training and validation vehicles.
        # In every eachTrainCar targetVehicle ID is same hence the first itme index 0 and first set is target vehicle and hence the idIndex withoput any offset
        targetVehicleID = eachTrainCar[0][idIndex]
        # Check all the movement types (Left/Right Straight) for just the target vehicle, and if there is atlst one frame with left or right make the whole trajectory same class/manueuver
        maneuverList = list(np.array(eachTrainCar)[:,movementIndex])
        # Check the main maneuver 0.5 -> left turn and 1.0 -> right turn
        leftTurnCount = maneuverList.count(0.5)
        rightTurnCount = maneuverList.count(1.0)
        mainManeuver = 0
        # Check if there is occurance of turn
        if(leftTurnCount > 5 or rightTurnCount > 5):
            if(leftTurnCount > rightTurnCount):
                mainManeuver = 0.5
            else:
                mainManeuver = 1.0

        for idx in range(historyTemporal,currentVehicleLength-futureTemporal):
            # Prepeare the input
            localInput = []
            for jdx in range(idx-historyTemporal,idx):
                currentPoseX = eachTrainCar[jdx][poseXIndex]
                currentPoseY = eachTrainCar[jdx][poseYIndex]
                currentVx = eachTrainCar[jdx][velXIndex]
                currentVy = eachTrainCar[jdx][velYIndex]
                currentLaneDist = eachTrainCar[jdx][laneDistIndex]
                currentLaneNumber = eachTrainCar[jdx][laneNumberIndex]
                # currentMovement = eachTrainCar[jdx][movementIndex]
                # if(currentMovement != 0):
                currentMovement = mainManeuver
                currentAngle = eachTrainCar[jdx][angleIndex]
                # Normalize each field
                normCurrentPoseX = (currentPoseX-globalMinXPose)/(globalMaxXPose-globalMinXPose)
                normCurrentPoseY = (currentPoseY-globalMinYPose)/(globalMaxYPose-globalMinYPose)
                normCurrentVx = (currentVx-globalMinXVelocity)/(globalMaxXVelocity-globalMinXVelocity)
                normCurrentVy = (currentVy-globalMinYVelocity)/(globalMaxYVelocity-globalMinYVelocity)
                normCurrentLaneDist = (currentLaneDist-globalMinLaneDist)/(globalMaxLaneDist-globalMinLaneDist)
                normCurrentLaneNumber = currentLaneNumber/globalMaxLanenumber
                normCurrentMovement = currentMovement
                normAngle = currentAngle/360

                currentFrameInput = [normCurrentPoseX,normCurrentPoseY,normCurrentVx,normCurrentVy,normCurrentLaneDist,normCurrentLaneNumber]

                # Check if all the inputs are normalized
                for eachNormItem in currentFrameInput:
                    normBool = CheckValueNormalized(eachNormItem)
                    if not normBool:
                        print('Normalization Failed Target!!!')
                        sys.exit()

                # Add the surrounding Car info
                # Seperate the surrouding info from the target vehicle
                surroudingInfo = eachTrainCar[jdx][inputFeature:]

                for indexRange in range(0,len(surroudingInfo),inputFeature):

                    surroudingPoseX = surroudingInfo[indexRange+poseXIndex]
                    surroudingPoseY = surroudingInfo[indexRange+poseYIndex]
                    surroudingVx = surroudingInfo[indexRange+velXIndex]
                    surroudingVy = surroudingInfo[indexRange+velYIndex]
                    surroudingLaneDist = surroudingInfo[indexRange+laneDistIndex]
                    surroudingLaneNumber = surroudingInfo[indexRange+laneNumberIndex]
                    surroudingMovement = surroudingInfo[indexRange+movementIndex]
                    surroudingAngle = surroudingInfo[indexRange+angleIndex]

                    # Normalize each field
                    # For pose if the values are zero means padding
                    if(surroudingPoseX == 0 and surroudingPoseY == 0):
                        normSurroudingPoseX = 0
                        normSurroudingPoseY = 0
                        normSurroudingVx = 0
                        normSurroudingVy = 0
                        normSurroudingLaneDist = 0
                        normSurroudingLaneNumber = 0
                        normSurroudingMovement = 0
                        normSurrouningAngle = 0
                    else:
                        normSurroudingPoseX = (surroudingPoseX-globalMinXPose)/(globalMaxXPose-globalMinXPose)
                        normSurroudingPoseY = (surroudingPoseY-globalMinYPose)/(globalMaxYPose-globalMinYPose)
                        normSurroudingVx = (surroudingVx-globalMinXVelocity)/(globalMaxXVelocity-globalMinXVelocity)
                        normSurroudingVy = (surroudingVy-globalMinYVelocity)/(globalMaxYVelocity-globalMinYVelocity)
                        normSurroudingLaneDist = (surroudingLaneDist-globalMinLaneDist)/(globalMaxLaneDist-globalMinLaneDist)
                        normSurroudingLaneNumber = surroudingLaneNumber/globalMaxLanenumber
                        normSurroudingMovement = surroudingMovement
                        normSurrouningAngle = surroudingAngle/360

                    currentSurroundingInfo = [normSurroudingPoseX,normSurroudingPoseY,normSurroudingVx,normSurroudingVy,normSurroudingLaneDist,normSurroudingLaneNumber]

                    # Check if all the inputs are normalized
                    for eachNormItem in currentSurroundingInfo:
                        normBool = CheckValueNormalized(eachNormItem)
                        if not normBool:
                            print('Normalization Failed Surroudning!!!')
                            sys.exit()

                    currentFrameInput.extend(currentSurroundingInfo)

                # Append the input info to local list
                # Check the currentInputFrame should be a length of the (surroundingCarCount+1)*inputFeature
                currentInputLen = len(currentFrameInput)
                expectedLen = (surroudingCarCounts+1)*inputFeatureWithoutID
                if(currentInputLen != expectedLen):
                    print('During Data prep mismatch between expected and actual feature lenght!!!')
                    print('Expected Lenght : ' + str(expectedLen))
                    print('Actual length : ' + str(currentInputLen))
                    sys.exit()

                localInput.append(currentFrameInput)

            # Get the last input for the decoder first input (use the variable as they have the last instance)
            # Prep the targer decoder
            # Convert the movement info to class form
            movementClassInfo = MovementToClassForm(normCurrentMovement)
            firstDecoderInput = [movementClassInfo[0],movementClassInfo[1],movementClassInfo[2],normCurrentVx,normCurrentVy,normCurrentPoseX,normCurrentPoseY]
            # prep the surrouding decoder (use jdx as this will hold the last index)
            firstDecoderSurrounding = eachTrainCar[jdx][inputFeature:]
            for indexRange in range(0,len(firstDecoderSurrounding),inputFeature):  # Add the inputFeature as the index jump as the carInfo is prepered for input format
                surroudingPoseX = firstDecoderSurrounding[indexRange+poseXIndex]
                surroudingPoseY = firstDecoderSurrounding[indexRange+poseYIndex]
                surroudingVx = firstDecoderSurrounding[indexRange+velXIndex]
                surroudingVy = firstDecoderSurrounding[indexRange+velYIndex]
                surroudingMovement = firstDecoderSurrounding[indexRange+movementIndex]

                # Normalize each field
                # For pose if the values are zero means padding
                if(surroudingPoseX == 0 and surroudingPoseY == 0):
                    normSurroudingPoseX = 0
                    normSurroudingPoseY = 0
                    normSurroudingVx = 0
                    normSurroudingVy = 0
                    surroudingMovementClassInfo = [0,0,0]
                else:
                    normSurroudingPoseX = (surroudingPoseX-globalMinXPose)/(globalMaxXPose-globalMinXPose)
                    normSurroudingPoseY = (surroudingPoseY-globalMinYPose)/(globalMaxYPose-globalMinYPose)
                    normSurroudingVx = (surroudingVx-globalMinXVelocity)/(globalMaxXVelocity-globalMinXVelocity)
                    normSurroudingVy = (surroudingVy-globalMinYVelocity)/(globalMaxYVelocity-globalMinYVelocity)
                    surroudingMovementClassInfo = MovementToClassForm(surroudingMovement)

                intermediateFirstDecoderInfo = [surroudingMovementClassInfo[0],surroudingMovementClassInfo[1],surroudingMovementClassInfo[2],normSurroudingVx,normSurroudingVy,normSurroudingPoseX,normSurroudingPoseY]
                firstDecoderInput.extend(intermediateFirstDecoderInfo)

            # Check the firstDecoderInput should be a length of the (surroundingCarCount+1)*decoderInputFeature
            firstDecoderLen = len(firstDecoderInput)
            expectedLen = (surroudingCarCounts+1)*decoderInputFeature
            if(firstDecoderLen != expectedLen):
                print('During first decoder data prep mismatch between expected and actual feature length!!!')
                print('Expected Lenght : ' + str(expectedLen))
                print('Actual length : ' + str(firstDecoderLen))
                sys.exit()

            # Prepeare the output
            localDecoderInput = []
            localClassOut = []
            localVelOut = []
            localPoseOut = []
            for kdx in range(idx,idx+futureTemporal):
                currentPoseX = eachTrainCar[kdx][poseXIndex]
                currentPoseY = eachTrainCar[kdx][poseYIndex]
                currentVx = eachTrainCar[kdx][velXIndex]
                currentVy = eachTrainCar[kdx][velYIndex]
                # currentMovement = eachTrainCar[kdx][movementIndex]
                # if(currentMovement != 0):
                currentMovement = mainManeuver
                # Convert the movement info to class form
                movementClassInfo = MovementToClassForm(currentMovement)
                # Append the input info to local list
                localClassOut.append([movementClassInfo[0],movementClassInfo[1],movementClassInfo[2]])
                localVelOut.append([currentVx,currentVy])
                localPoseOut.append([currentPoseX,currentPoseY])

                # normalize the fields before appending to the Decoder
                # currentDecoderInfo = []
                normCurrentMovement = movementClassInfo
                normCurrentPoseX = (currentPoseX-globalMinXPose)/(globalMaxXPose-globalMinXPose)
                normCurrentPoseY = (currentPoseY-globalMinYPose)/(globalMaxYPose-globalMinYPose)
                normCurrentVx = (currentVx-globalMinXVelocity)/(globalMaxXVelocity-globalMinXVelocity)
                normCurrentVy = (currentVy-globalMinYVelocity)/(globalMaxYVelocity-globalMinYVelocity)

                # Add the target vehicle decoder info
                currentDecoderInfo = [movementClassInfo[0],movementClassInfo[1],movementClassInfo[2],normCurrentVx,normCurrentVy,normCurrentPoseX,normCurrentPoseY]

                # Add the surrounding Car info in the decoder
                # Seperate the surrouding info from the target vehicle
                surroudingDecoderInfo = eachTrainCar[kdx][inputFeature:]

                for indexRange in range(0,len(surroudingInfo),inputFeature):  # Add the inputFeature as the index jump as the carInfo is prepered for input format

                    surroudingPoseX = surroudingDecoderInfo[indexRange+poseXIndex]
                    surroudingPoseY = surroudingDecoderInfo[indexRange+poseYIndex]
                    surroudingVx = surroudingDecoderInfo[indexRange+velXIndex]
                    surroudingVy = surroudingDecoderInfo[indexRange+velYIndex]
                    surroudingMovement = surroudingDecoderInfo[indexRange+movementIndex]

                    # Normalize each field
                    # For pose if the values are zero means padding
                    if(surroudingPoseX == 0 and surroudingPoseY == 0):
                        normSurroudingPoseX = 0
                        normSurroudingPoseY = 0
                        normSurroudingVx = 0
                        normSurroudingVy = 0
                        surroudingMovementClassInfo = [0,0,0]
                    else:
                        normSurroudingPoseX = (surroudingPoseX-globalMinXPose)/(globalMaxXPose-globalMinXPose)
                        normSurroudingPoseY = (surroudingPoseY-globalMinYPose)/(globalMaxYPose-globalMinYPose)
                        normSurroudingVx = (surroudingVx-globalMinXVelocity)/(globalMaxXVelocity-globalMinXVelocity)
                        normSurroudingVy = (surroudingVy-globalMinYVelocity)/(globalMaxYVelocity-globalMinYVelocity)
                        surroudingMovementClassInfo = MovementToClassForm(surroudingMovement)

                    intermediateSurroundingInfo = [surroudingMovementClassInfo[0],surroudingMovementClassInfo[1],surroudingMovementClassInfo[2],normSurroudingVx,normSurroudingVy,normSurroudingPoseX,normSurroudingPoseY]
                    currentDecoderInfo.extend(intermediateSurroundingInfo)

                localDecoderInput.append(currentDecoderInfo)

            # Prepeare the decoder input (add the first item and remove the last item)
            localDecoderInput.insert(0,firstDecoderInput)
            localDecoderInput.pop()

            # # # # If the vehicle is in the ignore list don't add
            # # # # No Need to ignore weired trajectory vehicles
            # # # if(targetVehicleID in ignoreList):
            # # #     continue

            # Append each datapoint to the main list based in the training or validation data
            if(targetVehicleID in validationVehicleList):
                xValList.append(localInput)
                decoderValList.append(localDecoderInput)
                YClassValList.append(localClassOut)
                YVelValList.append(localVelOut)
                YPoseValList.append(localPoseOut)
            else:
                xTrainList.append(localInput)
                decoderTrainList.append(localDecoderInput)
                YClassTrainList.append(localClassOut)
                YVelTrainList.append(localVelOut)
                YPoseTrainList.append(localPoseOut)

    # Convert all train lists to array and return.
    xTrainArray = np.array(xTrainList)
    decoderTrainArray = np.array(decoderTrainList)
    YClassTrainArray = np.array(YClassTrainList)
    YVelTrainArray = np.array(YVelTrainList)
    YPoseTrainArray = np.array(YPoseTrainList)

    # Convert all val lists to array and return.
    xValArray = np.array(xValList)
    decoderValArray = np.array(decoderValList)
    YClassValArray = np.array(YClassValList)
    YVelValArray = np.array(YVelValList)
    YPoseValArray = np.array(YPoseValList)

    return xTrainArray,decoderTrainArray,YClassTrainArray,YVelTrainArray,YPoseTrainArray,xValArray,decoderValArray,YClassValArray,YVelValArray,YPoseValArray

# Define the Custome learing rate decays
def step_decay(epoch):
    initial_lrate = 0.002   # 0.001 for Adma and RMSProp , 0.002 for Nadam
    drop = 0.5
    epochs_drop = 10.0    # 15
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
    def __init__(self, monitor='val_loss', value=4.0, verbose=0):
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


# Create the model
# def TrainModel(XTrain,decoderTrainInput,YClassTrain,YVelTrain,YPoseTrain):
def CreateModel():

    n_units = 256

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
    minXVelConst = K.constant(value=globalMinXVelocity, dtype='float32')
    minXMaxVelDiffConst = K.constant(value=(globalMaxXVelocity-globalMinXVelocity), dtype='float32')

    minYVelConst = K.constant(value=globalMinYVelocity, dtype='float32')
    minYMaxVelDiffConst = K.constant(value=(globalMaxYVelocity-globalMinYVelocity), dtype='float32')

    velocityXNormalized = Lambda(lambda x: (x-minXVelConst)/minXMaxVelDiffConst)
    velocityYNormalized = Lambda(lambda x: (x-minYVelConst)/minYMaxVelDiffConst)

    # Prepeare the slice layers nad separate the Vx and Vy
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

    # dropOut33 = Dropout(dropOutFrac)
    # batchNorm33 = BatchNormalization()
    decoder_dense30 = Dense(512)
    decoder_Leaky30 = ELU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense30(decoder_output3)
    # decoder_output3 = batchNorm33(decoder_output3)
    decoder_output3 = decoder_Leaky30(decoder_output3)
    # decoder_output3 = dropOut33(decoder_output3, training=True)

    # dropOut34 = Dropout(dropOutFrac)
    # batchNorm34 = BatchNormalization()
    decoder_dense31 = Dense(256)
    decoder_Leaky31 = ELU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense31(decoder_output3)
    # decoder_output3 = batchNorm34(decoder_output3)
    decoder_output3 = decoder_Leaky31(decoder_output3)
    # decoder_output3 = dropOut34(decoder_output3, training=True)

    # dropOut35 = Dropout(dropOutFrac)
    # batchNorm35 = BatchNormalization()
    decoder_dense32 = Dense(128)
    decoder_Leaky32 = ELU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense32(decoder_output3)
    # decoder_output3 = batchNorm35(decoder_output3)
    decoder_output3 = decoder_Leaky32(decoder_output3)
    # decoder_output3 = dropOut35(decoder_output3, training=True)

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
    # decoder_output3 = batchNorm33(decoder_output3)
    decoder_output3 = decoder_Leaky30(decoder_output3)
    # decoder_output3 = dropOut33(decoder_output3, training=True)

    decoder_output3 = decoder_dense31(decoder_output3)
    # decoder_output3 = batchNorm34(decoder_output3)
    decoder_output3 = decoder_Leaky31(decoder_output3)
    # decoder_output3 = dropOut34(decoder_output3, training=True)

    decoder_output3 = decoder_dense32(decoder_output3)
    # decoder_output3 = batchNorm35(decoder_output3)
    decoder_output3 = decoder_Leaky32(decoder_output3)
    # decoder_output3 = dropOut35(decoder_output3, training=True)

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
    model.compile(optimizer=opt, loss=['categorical_crossentropy', logcosh, euclidean_distance_loss], loss_weights=[classWeight, velocityWeight, poseWeight])
    # model.summary()

    # Return the encoder and decoder model
    return model,encoder_model,decoder_model


# Fit the model
def FitModel(model,encoder_model,decoder_model,XTrain,decoderTrainInput,YClassTrain,YVelTrain,YPoseTrain,xVal,decoderValInput,YClassVal,YVelVal,YPoseVal):

    # Custom decay rates
    # loss_history = LossHistory()
    lrate = LearningRateScheduler(step_decay)
    esObj = EarlyStoppingByLossVal()
    # callbacks_list = [esObj,lrate]   #[loss_history, lrate]
    callbacks_list = [lrate]   #[loss_history, lrate]


    print('XTrain Shape : ' + str(XTrain.shape))
    print('DecoderInput Shape : ' + str(decoderTrainInput.shape))
    print('YClassTrain Shape : ' + str(YClassTrain.shape))
    print('YVelTrain Shape : ' + str(YVelTrain.shape))
    print('YPoseTrain Shape : ' + str(YPoseTrain.shape))

    print('XVal Shape : ' + str(xVal.shape))
    print('decoderValInput Shape : ' + str(decoderValInput.shape))
    print('YClassVal Shape : ' + str(YClassVal.shape))
    print('YVelVal Shape : ' + str(YVelVal.shape))
    print('YPoseVal Shape : ' + str(YPoseVal.shape))

    model.fit([XTrain,decoderTrainInput],[YClassTrain,YVelTrain,YPoseTrain], batch_size=batchSize, epochs=nepochs, verbose=1, validation_data=([xVal,decoderValInput],[YClassVal,YVelVal,YPoseVal]), callbacks=callbacks_list, shuffle=True)

    return model,encoder_model,decoder_model

# Test the model
def TestModel(encoder_model,decoder_model):

    lineThickness = 3
    finalErrorArray = np.zeros(futureTemporal)
    finalMotionErrorArray = np.zeros(futureTemporal)
    errorCount = 0
    classStrList = []
    localSaveCount = 0
    errorHistList = []

    for eachTrainCar,eachTrainCarDropped in zip(carInfoList,carInfoListDropped):
        ###########################################################################
        ## For vehicle Length and main maneuver identification use the un-dropped ##
        ###########################################################################
        currentVehicleLength = len(eachTrainCar)
        # Pick the current target vehicle ID to sepereate the training and validation vehicles.
        # In every eachTrainCar targetVehicle ID is same hence the first itme index 0 and first set is target vehicle and hence the idIndex withoput any offset
        targetVehicleID = eachTrainCar[0][idIndex]
        # Test only on test vehicles
        if not (targetVehicleID in validationVehicleList):
            continue

        print('Current Vehicle ID ' + str(targetVehicleID))

        # Check all the movement types (Left/Right Straight) for just the target vehicle, and if there is atlst one frame with left or right make the whole trajectory same class/manueuver
        maneuverList = list(np.array(eachTrainCar)[:,movementIndex])
        # Check the main maneuver 0.5 -> left turn and 1.0 -> right turn
        leftTurnCount = maneuverList.count(0.5)
        rightTurnCount = maneuverList.count(1.0)
        mainManeuver = 0
        # Check if there is occurance of turn
        if(leftTurnCount > 5 or rightTurnCount > 5):
            if(leftTurnCount > rightTurnCount):
                mainManeuver = 0.5
            else:
                mainManeuver = 1.0
        ###############################################################################

        for idx in range(historyTemporal,currentVehicleLength-futureTemporal,10):
            # Load the radar and map image and draw the lane lines for visualization
            resultRadarImage = cv2.imread(radarPath)
            # resultRadarImage = cv2.cvtColor(resultRadarImage, cv2.COLOR_GRAY2BGR)
            resultMapImage = cv2.imread(mapImagePath)

            # # # Add the notations blue input, green predicted straight, red predicted left, yellow predicted right

            # # # for laneIndex,eacLine in enumerate(laneLines):
            # # #     resultMapImage = cv2.line(resultMapImage, (eacLine[0], eacLine[1]), (eacLine[2], eacLine[3]), (255,0,0), 2)

            # # # Add the color code for input, groundtruth and predicted trajectory
            # # resultMapImage = cv2.putText(resultMapImage, 'Red : Input Traj', (750,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            # # resultMapImage = cv2.putText(resultMapImage, 'Green : Predicted', (750,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            # # resultMapImage = cv2.putText(resultMapImage, 'Blue : Ground Truth', (750,140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

            # Re read the radar and map image to draw car boudning box
            resultRadarImage = cv2.imread(radarPath)
            resultMapImage = cv2.imread(mapImagePath)

            # Prepeare the input
            localInput = []
            for jdx in range(idx-historyTemporal,idx):
                # Comment the re read to show the input trajectory
                # # # Re read the radar and map image to draw car boudning box
                # # resultRadarImage = cv2.imread(radarPath)
                # # resultMapImage = cv2.imread(mapImagePath)

                # Add the color code for input, groundtruth and predicted trajectory
                # Add the notations blue input, green predicted straight, red predicted left, yellow predicted right
                resultMapImage = cv2.putText(resultMapImage, 'Blue : Predicted', (750,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                resultMapImage = cv2.putText(resultMapImage, 'Green : Ground Truth', (750,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                resultMapImage = cv2.putText(resultMapImage, 'Red : Surrounding Car', (750,140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                # resultMapImage = cv2.putText(resultMapImage, 'yellow : Far Surroundings', (750,170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
                resultMapImage = cv2.putText(resultMapImage, 'Violet : Target Car', (750,170), cv2.FONT_HERSHEY_SIMPLEX, 1, (238,130,238), 2, cv2.LINE_AA)

                resultRadarImage = cv2.putText(resultRadarImage, 'Blue : Predicted', (750,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                resultRadarImage = cv2.putText(resultRadarImage, 'Green : Ground Truth', (750,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                resultRadarImage = cv2.putText(resultRadarImage, 'Red : Surrounding Car', (750,140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                # resultRadarImage = cv2.putText(resultRadarImage, 'yellow : Far Surroundings', (750,170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
                resultRadarImage = cv2.putText(resultRadarImage, 'Violet : Target Car', (750,170), cv2.FONT_HERSHEY_SIMPLEX, 1, (238,130,238), 2, cv2.LINE_AA)

                # Add the target vehicle input
                # For the target vehicle input use the dropped list items
                # As both the lists are of same length use the same index
                currentPoseX = eachTrainCarDropped[jdx][poseXIndex]
                currentPoseY = eachTrainCarDropped[jdx][poseYIndex]
                currentVx = eachTrainCarDropped[jdx][velXIndex]
                currentVy = eachTrainCarDropped[jdx][velYIndex]
                currentLaneDist = eachTrainCarDropped[jdx][laneDistIndex]
                currentLaneNumber = eachTrainCarDropped[jdx][laneNumberIndex]
                # currentMovement = eachTrainCar[jdx][movementIndex]
                # if(currentMovement != 0):
                currentMovement = mainManeuver
                currentAngle = eachTrainCarDropped[jdx][angleIndex]
                # Normalize each field
                normCurrentPoseX = (currentPoseX-globalMinXPose)/(globalMaxXPose-globalMinXPose)
                normCurrentPoseY = (currentPoseY-globalMinYPose)/(globalMaxYPose-globalMinYPose)
                normCurrentVx = (currentVx-globalMinXVelocity)/(globalMaxXVelocity-globalMinXVelocity)
                normCurrentVy = (currentVy-globalMinYVelocity)/(globalMaxYVelocity-globalMinYVelocity)
                normCurrentLaneDist = (currentLaneDist-globalMinLaneDist)/(globalMaxLaneDist-globalMinLaneDist)
                normCurrentLaneNumber = currentLaneNumber/globalMaxLanenumber
                normCurrentMovement = currentMovement
                normCurrentAngle = currentAngle/360

                tempInput = [normCurrentPoseX,normCurrentPoseY,normCurrentVx,normCurrentVy,normCurrentLaneDist,normCurrentLaneNumber]

                # Add the surrounding Car info
                # For surrounding car info nothing to do with the drop
                # Seperate the surrouding info from the target vehicle
                surroudingInfo = eachTrainCar[jdx][inputFeature:]

                # Get the prev surrouding info for traj drawing
                if(jdx>0):
                    prevSurroudingInfo = eachTrainCar[jdx-1][inputFeature:]

                for indexRange in range(0,len(surroudingInfo),inputFeature):
                    surroudingPoseX = surroudingInfo[indexRange+poseXIndex]
                    surroudingPoseY = surroudingInfo[indexRange+poseYIndex]
                    surroudingVx = surroudingInfo[indexRange+velXIndex]
                    surroudingVy = surroudingInfo[indexRange+velYIndex]
                    surroudingLaneDist = surroudingInfo[indexRange+laneDistIndex]
                    surroudingLaneNumber = surroudingInfo[indexRange+laneNumberIndex]
                    surroudingMovement = surroudingInfo[indexRange+movementIndex]
                    surroudingAngle = surroudingInfo[indexRange+angleIndex]
                    surroudingID = surroudingInfo[indexRange+idIndex]
                    # Normalize each field
                    if(surroudingPoseX == 0 and surroudingPoseY == 0):
                        normSurroudingPoseX = 0
                        normSurroudingPoseY = 0
                        normSurroudingVx = 0
                        normSurroudingVy = 0
                        normSurroudingLaneDist = 0
                        normSurroudingLaneNumber = 0
                        normSurroudingMovement = 0
                        normSurrouningAngle = 0
                    else:
                        normSurroudingPoseX = (surroudingPoseX-globalMinXPose)/(globalMaxXPose-globalMinXPose)
                        normSurroudingPoseY = (surroudingPoseY-globalMinYPose)/(globalMaxYPose-globalMinYPose)
                        normSurroudingVx = (surroudingVx-globalMinXVelocity)/(globalMaxXVelocity-globalMinXVelocity)
                        normSurroudingVy = (surroudingVy-globalMinYVelocity)/(globalMaxYVelocity-globalMinYVelocity)
                        normSurroudingLaneDist = (surroudingLaneDist-globalMinLaneDist)/(globalMaxLaneDist-globalMinLaneDist)
                        normSurroudingLaneNumber = surroudingLaneNumber/globalMaxLanenumber
                        normSurroudingMovement = surroudingMovement
                        normSurrouningAngle = surroudingAngle/360

                    currentSurroundingInfo = [normSurroudingPoseX,normSurroudingPoseY,normSurroudingVx,normSurroudingVy,normSurroudingLaneDist,normSurroudingLaneNumber]
                    tempInput.extend(currentSurroundingInfo)

                    # Draw the traj for the surrouding car
                    if(jdx>0):
                        prevSurroudingPoseX = prevSurroudingInfo[indexRange+poseXIndex]
                        prevSurroudingPoseY = prevSurroudingInfo[indexRange+poseYIndex]
                        prevSurroudingID = prevSurroudingInfo[indexRange+idIndex]
                        surroudingColor = (0,0,255)
                        surroudingtrajColor = (0,75,150)
                        if(prevSurroudingPoseX != 0 and prevSurroudingPoseY != 0 and surroudingPoseX != 0 and surroudingPoseY != 0):
                            # Draw the bounding box only at at the last frame
                            if(jdx+1 == idx): 
                                # Draw the rotated bounding box as car (no need to check the last id as new car every time)
                                resultRadarImage = RotateBoundingBox(resultRadarImage,[surroudingPoseX,surroudingPoseY], surroudingAngle, surroudingColor)
                                resultMapImage = RotateBoundingBox(resultMapImage,[surroudingPoseX,surroudingPoseY], surroudingAngle, surroudingColor)
                            if(surroudingID == prevSurroudingID):
                                resultRadarImage = cv2.line(resultRadarImage, (prevSurroudingPoseX, prevSurroudingPoseY), (surroudingPoseX, surroudingPoseY), surroudingtrajColor, lineThickness)
                                resultMapImage = cv2.line(resultMapImage, (prevSurroudingPoseX, prevSurroudingPoseY), (surroudingPoseX, surroudingPoseY), surroudingtrajColor, lineThickness)


                # Append the input info to local list
                localInput.append(tempInput)

                # Draw the input traj using poses and movement
                # For input traj drawing use the dropped one
                if(jdx > 0):
                    prevPoseX = eachTrainCarDropped[jdx-1][poseXIndex]
                    prevPoseY = eachTrainCarDropped[jdx-1][poseYIndex]
                    angleColor = (238,130,238)
                    # Check the movement info and decide the traj color
                    if(currentMovement == 0):
                        angleColor = (238,130,238)
                    resultRadarImage = cv2.line(resultRadarImage, (prevPoseX, prevPoseY), (currentPoseX, currentPoseY), (0,75,150), lineThickness)
                    resultMapImage = cv2.line(resultMapImage, (prevPoseX, prevPoseY), (currentPoseX, currentPoseY), (0,75,150), lineThickness)
                    # Draw the target car bounding box
                    # Draw the bounding box only at at the last frame
                    if(jdx+1 == idx):
                        resultRadarImage = RotateBoundingBox(resultRadarImage,[currentPoseX,currentPoseY], currentAngle, angleColor)
                        resultMapImage = RotateBoundingBox(resultMapImage,[currentPoseX,currentPoseY], currentAngle, angleColor)

                # Concat the images for visualitzation
                finalImage = np.hstack((resultRadarImage,resultMapImage))
                cv2.imshow('image', finalImage)
                cv2.waitKey(200)


            # Get the last input for the decoder first input (use the variable as they have the last instance)
            normMovementInfo = MovementToClassForm(normCurrentMovement)
            firstDecoderInput = [normMovementInfo[0],normMovementInfo[1],normMovementInfo[2],normCurrentVx,normCurrentVy,normCurrentPoseX,normCurrentPoseY]
            # Add the surrouding info for the first decoder input (use jdx as it it still the last index)
            # For the first decoder input also use the dropped one as htis is the last input basically
            firstDecoderSurrounding = eachTrainCarDropped[jdx][inputFeature:]
            for indexRange in range(0,len(firstDecoderSurrounding),inputFeature):  # Add the inputFeature as the index jump as the carInfo is prepered for input format
                surroudingPoseX = firstDecoderSurrounding[indexRange+poseXIndex]
                surroudingPoseY = firstDecoderSurrounding[indexRange+poseYIndex]
                surroudingVx = firstDecoderSurrounding[indexRange+velXIndex]
                surroudingVy = firstDecoderSurrounding[indexRange+velYIndex]
                surroudingMovement = firstDecoderSurrounding[indexRange+movementIndex]

                # Normalize each field
                # For pose if the values are zero means padding
                if(surroudingPoseX == 0 and surroudingPoseY == 0):
                    normSurroudingPoseX = 0
                    normSurroudingPoseY = 0
                    normSurroudingVx = 0
                    normSurroudingVy = 0
                    surroudingMovementClassInfo = [0,0,0]
                else:
                    normSurroudingPoseX = (surroudingPoseX-globalMinXPose)/(globalMaxXPose-globalMinXPose)
                    normSurroudingPoseY = (surroudingPoseY-globalMinYPose)/(globalMaxYPose-globalMinYPose)
                    normSurroudingVx = (surroudingVx-globalMinXVelocity)/(globalMaxXVelocity-globalMinXVelocity)
                    normSurroudingVy = (surroudingVy-globalMinYVelocity)/(globalMaxYVelocity-globalMinYVelocity)
                    surroudingMovementClassInfo = MovementToClassForm(surroudingMovement)

                intermediateFirstDecoderInfo = [surroudingMovementClassInfo[0],surroudingMovementClassInfo[1],surroudingMovementClassInfo[2],normSurroudingVx,normSurroudingVy,normSurroudingPoseX,normSurroudingPoseY]
                firstDecoderInput.extend(intermediateFirstDecoderInfo)


            # Gather the ground truth pose array and maneuver class
            # For ground truth output use the un dropped one as this way we will have the original output to compare with the dropped input
            groundTruthPose = []
            groundTruthClass = []
            surroudingDecoder = []
            surroudingDecoderDrawList = []
            for kdx in range(idx,idx+futureTemporal):
                currentPoseX = eachTrainCar[kdx][poseXIndex]
                currentPoseY = eachTrainCar[kdx][poseYIndex]
                currentVx = eachTrainCar[kdx][velXIndex]
                currentVy = eachTrainCar[kdx][velYIndex]
                # currentMovement = eachTrainCar[kdx][movementIndex]
                # if(currentMovement != 0):
                currentMovement = mainManeuver

                # Normalize each field
                normPoseX = (currentPoseX-globalMinXPose)/(globalMaxXPose-globalMinXPose)
                normPoseY = (currentPoseY-globalMinYPose)/(globalMaxYPose-globalMinYPose)
                normVx = (currentVx-globalMinXVelocity)/(globalMaxXVelocity-globalMinXVelocity)
                normVy = (currentVy-globalMinYVelocity)/(globalMaxYVelocity-globalMinYVelocity)
                targetMovementClassInfo = MovementToClassForm(currentMovement)

                # Append the ground truth info to local list
                groundTruthPose.append([currentPoseX,currentPoseY])
                groundTruthClass.append(targetMovementClassInfo)

                tempDecoderSurrounding = eachTrainCar[kdx][inputFeature:]
                # tempDecoderInput = [targetMovementClassInfo[0],targetMovementClassInfo[1],targetMovementClassInfo[2],normVx,normVy,normPoseX,normPoseY]
                tempDecoderInput = []
                tempDecoderDraw = []
                for indexRange in range(0,len(firstDecoderSurrounding),inputFeature):  # Add the inputFeature as the index jump as the carInfo is prepered for input format
                    surroudingPoseX = tempDecoderSurrounding[indexRange+poseXIndex]
                    surroudingPoseY = tempDecoderSurrounding[indexRange+poseYIndex]
                    surroudingVx = tempDecoderSurrounding[indexRange+velXIndex]
                    surroudingVy = tempDecoderSurrounding[indexRange+velYIndex]
                    surroudingMovement = tempDecoderSurrounding[indexRange+movementIndex]
                    surroudingID = tempDecoderSurrounding[indexRange+idIndex]

                    # Normalize each field
                    # For pose if the values are zero means padding
                    if(surroudingPoseX == 0 and surroudingPoseY == 0):
                        normSurroudingPoseX = 0
                        normSurroudingPoseY = 0
                        normSurroudingVx = 0
                        normSurroudingVy = 0
                        surroudingMovementClassInfo = [0,0,0]
                    else:
                        normSurroudingPoseX = (surroudingPoseX-globalMinXPose)/(globalMaxXPose-globalMinXPose)
                        normSurroudingPoseY = (surroudingPoseY-globalMinYPose)/(globalMaxYPose-globalMinYPose)
                        normSurroudingVx = (surroudingVx-globalMinXVelocity)/(globalMaxXVelocity-globalMinXVelocity)
                        normSurroudingVy = (surroudingVy-globalMinYVelocity)/(globalMaxYVelocity-globalMinYVelocity)
                        surroudingMovementClassInfo = MovementToClassForm(surroudingMovement)

                    intermediateDecoderInfo = [surroudingMovementClassInfo[0],surroudingMovementClassInfo[1],surroudingMovementClassInfo[2],normSurroudingVx,normSurroudingVy,normSurroudingPoseX,normSurroudingPoseY]
                    tempDecoderInput.extend(intermediateDecoderInfo)
                    intermediateDecoderDraw = [surroudingPoseX,surroudingPoseY,surroudingID]
                    tempDecoderDraw.extend(intermediateDecoderDraw)

                # Now add the current decoder input to the main surrouding decoder input
                surroudingDecoder.append(tempDecoderInput)
                surroudingDecoderDrawList.append(tempDecoderDraw)


            for repeat in range(0,repeatCount):
                # Format the input array for prediction
                localInputArray = np.array(localInput).reshape(1,historyTemporal,totalInputFeature)
                target_seq = np.array(firstDecoderInput).reshape(1,1,totalDecoderFeature)

                # Start the prediction
                # Predict the encoder state
                state = encoder_model.predict(localInputArray)

                # Declare list to gather the predicted pose and the calculated error
                outputPredPose = []
                currentError = []
                maneuverForDraw = []
                currentMotionError = []

                # Declaring outside for scope for later use in histgram
                trueClassFloat = -999
                PredictedClassFloat = -999

                # Intilize the pose with origin and un n0rmalized
                initialPoseX = (localInputArray[0][-1][poseXIndex]*(globalMaxXPose-globalMinXPose)) + globalMinXPose
                initialPoseY = (localInputArray[0][-1][poseYIndex]*(globalMaxYPose-globalMinYPose)) + globalMinYPose
                intitalVelocityX = (localInputArray[0][-1][velXIndex]*(globalMaxXVelocity-globalMinXVelocity)) + globalMinXVelocity
                intitalVelocityY = (localInputArray[0][-1][velYIndex]*(globalMaxYVelocity-globalMinYVelocity)) + globalMinYVelocity

                # Perfrom the sequential prediction
                for t in range(futureTemporal):
                    # predict next Features
                    classPred, velcoityPred, posePred, h1, c1, h2, c2 = decoder_model.predict([target_seq] + state)

                    # store prediction
                    outputPredPose.append([posePred[0][0][0],posePred[0][0][1]])

                    # Normalize the predicted velocity for next instance prediction
                    normalizedPredVelocityX = (velcoityPred[0][0][0]-globalMinXVelocity)/(globalMaxXVelocity-globalMinXVelocity)
                    normalizedPredVelocityY = (velcoityPred[0][0][1]-globalMinYVelocity)/(globalMaxYVelocity-globalMinYVelocity)

                    # Normalize the predicted local poses for next instance prediction
                    normalizedPredPoseX = (posePred[0][0][0]-globalMinXPose)/(globalMaxXPose-globalMinXPose)
                    normalizedPredPoseY = (posePred[0][0][1]-globalMinYPose)/(globalMaxYPose-globalMinYPose)

                    # Get the poses from motion
                    motionPredPoseX = initialPoseX - intitalVelocityX
                    motionPredPoseY = initialPoseY - intitalVelocityY
                    normalizedMotionpredX = (motionPredPoseX-globalMinXPose)/(globalMaxXPose-globalMinXPose)
                    normalizedMotionPredY = (motionPredPoseY-globalMinYPose)/(globalMaxYPose-globalMinYPose)
                    intitalVelocityX = velcoityPred[0][0][0]
                    intitalVelocityY = velcoityPred[0][0][1]

                    # Update the initialPoses
                    initialPoseX = motionPredPoseX
                    initialPoseY = motionPredPoseY

                    # update state
                    state = [h1, c1, h2, c2]

                    # update target sequence
                    # Update the target sequence till second last frame. At the last frame no need to update the seq as it will not be used
                    if(t<(futureTemporal-1)):
                        targetDecoder = [classPred[0][0][0],classPred[0][0][1],classPred[0][0][2],normalizedPredVelocityX,normalizedPredVelocityY,normalizedPredPoseX,normalizedPredPoseY]
                        # Add the surrouding GT info into the target seq
                        currentSurroudingDecoder = surroudingDecoder[t]
                        targetDecoder.extend(currentSurroudingDecoder)
                        target_seq = np.array(targetDecoder).reshape(1,1,totalDecoderFeature)

                    # Calculate the confusion Matrix for classification
                    trueClassFloat = (np.array(groundTruthClass[t]).argmax())/2   # divide by 2 to convert 0,1,2  to 0,0.5,1.0
                    trueClassStr = MovementToStr(trueClassFloat)
                    PredictedClassFloat = (classPred[0][0].argmax())/2  # divide by 2 to convert 0,1,2  to 0,0.5,1.0
                    predictedClassStr = MovementToStr(PredictedClassFloat)

                    # Append the true and predicted class info str
                    classStrList.append([trueClassStr,predictedClassStr])
                    maneuverForDraw.append(predictedClassStr)

                    # Calculate the Euclidian Error
                    truePoseX = groundTruthPose[t][0]
                    truePoseY = groundTruthPose[t][1]

                    predPoseX = outputPredPose[t][0]
                    predPoseY = outputPredPose[t][1]

                    euclidianError = math.sqrt(((truePoseX-predPoseX)**2) + ((truePoseY-predPoseY)**2))
                    euclidianErrorMeter = euclidianError*cellResolution

                    currentError.append(euclidianErrorMeter)

                    euclidianMotionError = math.sqrt(((truePoseX-motionPredPoseX)**2) + ((truePoseY-motionPredPoseY)**2))
                    euclidianMotionErrorMeter = euclidianMotionError*cellResolution

                    currentMotionError.append(euclidianMotionErrorMeter)

                # Check the currentError list lenght. Should be equal to futureTemporal
                if(len(currentError) != futureTemporal):
                    print('Future Temporal lenght is not match the expected')
                    print('Expected Error list lentth : ' + str(futureTemporal))
                    print('Actual Error list lentth : ' + str(len(currentError)))
                    sys.exit()

                # Use the last frame true and predicted class to identify right or wrong prediction
                # Use the variable trueClassFloat and PredictedClassFloat directly as the will have the last value
                predictionCategory = -999
                if(trueClassFloat == PredictedClassFloat):
                    predictionCategory = 1  # Correct maneuver prediction
                else:
                    predictionCategory = 0  # Wrong maneuver prediction

                # Same for the poisition information 
                lastError = euclidianErrorMeter

                # Append both the info into errorHistList
                errorHistList.append([lastError,predictionCategory])

                currentErrorArray = np.array(currentError)
                finalErrorArray = finalErrorArray + currentErrorArray
                currentMotionErrorArray = np.array(currentMotionError)
                finalMotionErrorArray = finalMotionErrorArray + currentMotionErrorArray
                errorCount = errorCount+1

                # Draw the predicted and ground truth trajetory on the map and radar image
                # Check the length of ground truth and predicted pose length same
                predPoseLength = len(outputPredPose)
                groundTruthLength = len(groundTruthPose)
                if(predPoseLength != groundTruthLength):
                    print('predpose and groundTruthPose are not equal!!!')
                    print('predPose Length = ' + str(predPoseLength))
                    print('groundTruthPose Length = ' + str(groundTruthLength))
                    sys.exit()

                predColor = (255,0,0) # blue
                trueColor = (0,255,0) # green
                # Get the intital surrounding IDs
                initialIdList = []
                initialSurroundingCars = surroudingDecoderDrawList[0]
                for intitalIdIndex in range(0,len(initialSurroundingCars),3):
                    initialId = initialSurroundingCars[intitalIdIndex+2]
                    initialIdList.append(initialId)

                # Maintain list of new IDs for which the cars have been drawn
                newIdListDrawn = []

                for drawIndex in range(1,predPoseLength):
                    # Get the current and last predicted pose
                    currentPredPoseX = outputPredPose[drawIndex][0]
                    currentPredPoseY = outputPredPose[drawIndex][1]
                    prevPredPoseX = outputPredPose[drawIndex-1][0]
                    prevPredPoseY = outputPredPose[drawIndex-1][1]
                    # Get the current and last ground Truth pose
                    currentTruePoseX = groundTruthPose[drawIndex][0]
                    currentTruePoseY = groundTruthPose[drawIndex][1]
                    prevTruePoseX = groundTruthPose[drawIndex-1][0]
                    prevTruePoseY = groundTruthPose[drawIndex-1][1]


                    # Insted of color traj for the manuevr put it as text
                    currentDrawMan = maneuverForDraw[drawIndex]
                    maneuverStr = 'Manueuver : '
                    if(currentDrawMan ==  straightStr):
                        maneuverStr = maneuverStr + 'Straight'
                    elif(currentDrawMan == leftTurnStr):
                        maneuverStr = maneuverStr + 'Left'
                    elif(currentDrawMan == rightTurnStr):
                        maneuverStr = maneuverStr + 'Right'  
                    else:
                        print('Unknown maneuver string predicted!!!')
                        print('Maneuver string is ' + currentDrawMan)
                        # print('leftTurnStr string is ' + leftTurnStr)
                        sys.exit()

                    # Add the maneuver text
                    # Commenting out the maneuver at the moment
                    resultMapImage = cv2.putText(resultMapImage, maneuverStr, (750,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
                    resultRadarImage = cv2.putText(resultRadarImage, maneuverStr, (750,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)

                    # Draw the true and predicted on the radar image
                    # Only draw the true trajectory at the moment
                    resultRadarImage = cv2.line(resultRadarImage, (prevPredPoseX, prevPredPoseY), (currentPredPoseX, currentPredPoseY), predColor, lineThickness)
                    resultRadarImage = cv2.line(resultRadarImage, (prevTruePoseX, prevTruePoseY), (currentTruePoseX, currentTruePoseY), trueColor, lineThickness)

                    # Draw the true and predicted on the map image
                    # Only draw the true trajectory at the moment
                    resultMapImage = cv2.line(resultMapImage, (prevPredPoseX, prevPredPoseY), (currentPredPoseX, currentPredPoseY), predColor, lineThickness)
                    resultMapImage = cv2.line(resultMapImage, (prevTruePoseX, prevTruePoseY), (currentTruePoseX, currentTruePoseY), trueColor, lineThickness)

                    # # # # Draw all the surrouding car traj
                    # # # currentDrawSurrouding = surroudingDecoderDrawList[drawIndex]
                    # # # prevDrawSurrouding = surroudingDecoderDrawList[drawIndex-1]
                    # # # for decodeDrawIndex in range(0,len(currentDrawSurrouding),3):  # Add the inputFeature as the index jump as the carInfo is prepered for input format, 3 jump beacuse there are 3 items in the draw list poseX, poseY and ID
                    # # #     currentSurroudingPoseX = currentDrawSurrouding[decodeDrawIndex]  # first item is the poseX for the surroudingDecoderDrawList
                    # # #     currentSurroudingPoseY = currentDrawSurrouding[decodeDrawIndex+1]  # second item is the poseY for the surroudingDecoderDrawList
                    # # #     currentID = currentDrawSurrouding[decodeDrawIndex+2]  # third item is the ID for the surroudingDecoderDrawList

                    # # #     prevSurroudingPoseX = prevDrawSurrouding[decodeDrawIndex]  # first item is the poseX for the surroudingDecoderDrawList
                    # # #     prevSurroudingPoseY = prevDrawSurrouding[decodeDrawIndex+1]  # second item is the poseX for the surroudingDecoderDrawList
                    # # #     prevID = prevDrawSurrouding[decodeDrawIndex+2]  # third item is the poseX for the surroudingDecoderDrawList

                    # # #     # Draw the surrouding traj on the radar and map image
                    # # #     if(currentSurroudingPoseX != 0 and currentSurroudingPoseY != 0 and prevSurroudingPoseX != 0 and prevSurroudingPoseY != 0):
                    # # #         if(currentID == prevID):
                    # # #             if(currentID in initialIdList):
                    # # #                 resultRadarImage = cv2.line(resultRadarImage, (prevSurroudingPoseX, prevSurroudingPoseY), (currentSurroudingPoseX, currentSurroudingPoseY), (0,0,255), lineThickness)
                    # # #                 resultMapImage = cv2.line(resultMapImage, (prevSurroudingPoseX, prevSurroudingPoseY), (currentSurroudingPoseX, currentSurroudingPoseY), (0,0,255), lineThickness)
                    # # #             else:
                    # # #                 if not (currentID in newIdListDrawn):
                    # # #                     # draw the car
                    # # #                     resultRadarImage = cv2.circle(resultRadarImage, (prevSurroudingPoseX, prevSurroudingPoseY), 10, (255,255,0), lineThickness)
                    # # #                     resultMapImage = cv2.circle(resultMapImage, (prevSurroudingPoseX, prevSurroudingPoseY),10, (255,255,0), lineThickness)
                    # # #                     newIdListDrawn.append(currentID)

                    # # #                 resultRadarImage = cv2.line(resultRadarImage, (prevSurroudingPoseX, prevSurroudingPoseY), (currentSurroudingPoseX, currentSurroudingPoseY), (255,255,0), lineThickness)
                    # # #                 resultMapImage = cv2.line(resultMapImage, (prevSurroudingPoseX, prevSurroudingPoseY), (currentSurroudingPoseX, currentSurroudingPoseY), (255,255,0), lineThickness)

                    # Concat the images for visualitzation
                    finalImage = np.hstack((resultRadarImage,resultMapImage))
                    cv2.imshow('image', finalImage)
                    cv2.waitKey(200)
                    # print('idx value is ' + str(idx))
                    # # Save the image for limited seq
                    # if(idx>35):
                    #     localSaveCount = localSaveCount + 1
                    #     imageSaveFilePath = '/home/saptarshi/PythonCode/Junction/CaseStudy/case4/' + str(localSaveCount) + '.png'
                    #     cv2.imwrite(imageSaveFilePath,finalImage)

    # Print the final error
    print('Final Error is : ')
    print(finalErrorArray/errorCount)
    print('Final Motion Error is : ')
    print(finalMotionErrorArray/(errorCount*1.4))

    print('Confusion Matrix!!!')
    retConfusionMatrix = CalcConfusionMatrix(classStrList)
    print(retConfusionMatrix)
    print('Confusion Matrix printed!!!')

    # Write errorHistList To File
    with open(errorHistFilePath, 'w') as hx:
        for item in errorHistList:
            hx.write("%s\n" % item)




def __main__():

    # Load up the traj data from the Navtech folder
    folderList = os.listdir(sequence_folder)
    folderList.sort(key=lambda x: int(x.split('_')[-1]))
    idOffset = 0
    globalImageLoaded = False
    global globalRadarDisplayImage,globalRadarSmoothTraj
    averageVelocityGlobalList = []

    for eachSeq in folderList:
        print('Processing folder : ' + eachSeq)
        # Ignore junction 3 as its giving nan for the filter....
        if(eachSeq == 'junction_1_3'):
            continue
        sequence_path = os.path.join(sequence_folder, eachSeq)
        annotation_path = os.path.join(sequence_path, 'annotations', 'annotations.json')

        #Load the globalImage for trajectory visualization
        if not globalImageLoaded:
            intitalImagePath = os.path.join(sequence_path, 'Navtech_Cartesian', str_format.format(1) + '.png')
            globalRadarDisplayImage = cv2.imread(intitalImagePath)
            globalRadarSmoothTraj = cv2.imread(intitalImagePath)
            globalImageLoaded = True

        sequence = Sequence(sequence_path)
        sequence.load_sequence(sequence_path)
        sequence.load_annotations(annotation_path)
        # showedImage = sequence.play(sequence_path,idOffset)

        sequence.playByVehicle(sequence_path,idOffset)
        # averageVelocityGlobalList.extend(returnedAvgVelList)
        idOffset = idOffset + sorted([d['id'] for d in sequence.annotations])[-1]

    # Print the total number of ignored cars
    print('Total number of ignored cars : ' + str(ignoredCars) + ' !!!')

    # Prepeare the data and get the input output arrays
    xTrainArray,decoderTrainArray,YClassTrainArray,YVelTrainArray,YPoseTrainArray,xValArray,decoderValArray,YClassValArray,YVelValArray,YPoseValArray = DataPrep()

    # Check all the items in xTrainArray and decoderTrainArray are <= 1.0
    xtrainCheck = (xTrainArray <= 1.0).all()
    decoderTrainArrayCheck = (decoderTrainArray <= 1.0).all()

    if(xtrainCheck == False):
        print('All the item in the input array are not normalied to 1')
        sys.exit()

    if(decoderTrainArrayCheck == False):
        print('All the item in the decoderInput array are not normalied to 1')
        sys.exit()


    print('xTrainArray Shape : ' + str(xTrainArray.shape))
    print('decoderTrainArray Shape : ' + str(decoderTrainArray.shape))
    print('YClassTrainArray Shape : ' + str(YClassTrainArray.shape))
    print('YVelTrainArray Shape : ' + str(YVelTrainArray.shape))
    print('YPoseTrainArray Shape : ' + str(YPoseTrainArray.shape))

    print('xValArray Shape : ' + str(xValArray.shape))
    print('decoderValArray Shape : ' + str(decoderValArray.shape))
    print('YClassValArray Shape : ' + str(YClassValArray.shape))
    print('YVelValArray Shape : ' + str(YVelValArray.shape))
    print('YPoseValArray Shape : ' + str(YPoseValArray.shape))

    print('Data prep done!!!')

    # Create the model
    model,encoder_model,decoder_model = CreateModel()

    if(trainOrTest == trainStr):
        print('Starting traininig !!!')

        # Create the model and train the model
        model,encoder_model,decoder_model = FitModel(model,encoder_model,decoder_model,xTrainArray,decoderTrainArray,YClassTrainArray,YVelTrainArray,YPoseTrainArray,xValArray,decoderValArray,YClassValArray,YVelValArray,YPoseValArray)

        # Backup models in case nothing else works
        # # # # Save the model 
        # # # encoder_model.save_weights('/home/saptarshi/PythonCode/Junction/NavtechModels/AlvieWrongTurnEncoderV1.h5')
        # # # decoder_model.save_weights('/home/saptarshi/PythonCode/Junction/NavtechModels/AlvieWrongTurnDecoderV1.h5')

        encoder_model.save_weights('/home/saptarshi/PythonCode/Junction/NavtechModels/AlvieWrongTurnMoreEffortEncoderV2.h5')
        decoder_model.save_weights('/home/saptarshi/PythonCode/Junction/NavtechModels/AlvieWrongTurnMoreEffortDecoderV2.h5')


    elif(trainOrTest == testStr):
        print('Starting testing !!!')

        # Backup models in case nothing else works
        # # # # encoder_model.load_weights('/home/saptarshi/PythonCode/Junction/NavtechModels/AlvieWrongTurnEncoderV1.h5')
        # # # # decoder_model.load_weights('/home/saptarshi/PythonCode/Junction/NavtechModels/AlvieWrongTurnDecoderV1.h5')

        # Previous models
        # AlvieDecoderV2.h5 and AlvieEncoderV2.h5 are good as well in terms of wrong maneuver 
        # Its just the trajectories are not that maneuver dependent

        encoder_model.load_weights('/home/saptarshi/PythonCode/Junction/NavtechModels/AlvieWrongTurnMoreEffortEncoderV2.h5')
        decoder_model.load_weights('/home/saptarshi/PythonCode/Junction/NavtechModels/AlvieWrongTurnMoreEffortDecoderV2.h5')

        # Test the model
        TestModel(encoder_model,decoder_model)


__main__()
