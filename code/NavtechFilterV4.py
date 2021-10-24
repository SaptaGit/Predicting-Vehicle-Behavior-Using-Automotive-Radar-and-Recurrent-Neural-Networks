# Extract the labbeled bounding boxes prep the input output traj and train a basic model 
# NO surrounding car info in this one
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import math
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import PFHelper
import math
from keras.models import Model, load_model
from keras.models import model_from_json
#from keras.utils import Sequence
from keras.layers import  Input,LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dropout, MaxPooling1D, Concatenate, division, subtract, Lambda, BatchNormalization, LeakyReLU, Reshape
import matplotlib.pyplot as plt
from keras import optimizers
from keras.optimizers import RMSprop, Adam
from keras import backend as K
from keras.callbacks import LearningRateScheduler
from keras import callbacks
from keras.losses import logcosh
import tensorflow as tf
import sys
# import pptk

# sequence_folder = '/home/saptarshi/Dropbox (Heriot-Watt University Team)/RES_EPS_PathCad/datasets/pathcad_van_dataset/junction_1'
sequence_folder = '/home/saptarshi/Dropbox (Heriot-Watt University Team)/RES_EPS_PathCad/datasets/pathcad_van_dataset/SighthillJunc'
# sequence_folder = '/home/saptarshi/PythonCode/Junction/SightHillSample'
imageSaveFolder = '/home/saptarshi/PythonCode/Junction/avgAngle/'
mapImagePath = '/home/saptarshi/PythonCode/Junction/SightHillMap.png'

# Train or test the model
trainStr = 'Train'
testStr = 'Test'
trainOrTest = testStr

# Perfrom data display or not
# 1 -> Display 0 -> no display
display = 0

# # Read map image and get the dimesion
# mapImage = cv2.imread(mapImagePath)
# mapWidth = mapImage.shape[0]
# mapHeight = mapImage.shape[1]

radarImageDimension = 1152
radarImageCentre = radarImageDimension/2

# dim = (radarImageDimension, radarImageDimension)
# resize image
# resizedMap = cv2.resize(mapImage, dim, interpolation = cv2.INTER_AREA)
if(display == 1 or trainOrTest == testStr):
    globalRadarDisplayImage = np.zeros((radarImageDimension,radarImageDimension,3))
    globalRadarSmoothTraj = np.zeros((radarImageDimension,radarImageDimension,3))
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    # cv2.namedWindow('smooth',cv2.WINDOW_NORMAL)

    # Read the globaL RADAR Image 
    radarPath = sequence_folder + '/SighthillJunc_3/Navtech_Cartesian/000001.png'
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
inputFeature = 8  
decoderInputFeature = 7
leakyAlphaValue = 0.5
batchSize = 64 #128
nepochs = 400
# Main road
# laneLines = [[143,418,1104,452], [129,458,1097,486], [446,417,454,94], [464,417,470,94], [600,483,605,1015], [564,483,592,1015], [553,533,109,513], [560,553,100,533], [624,537,1071,553], [627,561,1066,573]]

# Main road
laneLines = [[140,418,456,427], [130,458,583,470], [456,427,1106,450], [583,470,1097,484], [448,427,456,89], [465,427,470,89], [600,470,605,1015], [564,470,592,1015], [567,533,112,513], [570,553,100,533], [601,537,1077,553], [601,561,1069,573]]

# Info field index
poseXIndex = 0
poseYIndex = 1
velXIndex = 2
velYIndex = 3
laneDistIndex = 4
laneNumberIndex = 5
movementIndex = 6
angelIndex = 7

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
carInfoList = []
ignoredCars = 0

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

        #color = (255, 0, 0)
        #color = np.array(color) * 255

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

        global globalRadarDisplayImage,globalRadarSmoothTraj, imageCount, resizedMap, ignoredCars

        if (self.annotations != None):
            for object in self.annotations:
                if(display == 1):
                    # reload the image for trajectory visualization 
                    intitalImagePath = os.path.join(sequence_path, 'Navtech_Cartesian', str_format.format(1) + '.png')
                    globalRadarDisplayImage = cv2.imread(intitalImagePath)
                    globalRadarSmoothTraj = cv2.imread(intitalImagePath)

                    # # Draw Lane lines
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

                currentvehicleInfoList = []

                for idx,eachBbbox in enumerate(allBoundingBoxes):
                    if (eachBbbox):
                        if ((allDeleteParam[idx] == 0) and (allVisibleParam[idx] == 'visible')):
                            # Read the current Frame
                            if(display == 1):
                                radar_id = int(self.radar_ids[idx])
                                radar_cartesian_path = os.path.join(sequence_path, 'Navtech_Cartesian', str_format.format(radar_id) + '.png')
                                radar_cartesian = cv2.imread(radar_cartesian_path)

                            bbox = eachBbbox['position']
                            angle = eachBbbox['rotation']
                            currentVehicleAngleList.append(angle)

                            cx = int(bbox[0] + bbox[2]/2)
                            cy = int(bbox[1] + bbox[3]/2)

                            centrePoses.append([cx,cy])

                            if(len(centrePoses) > 2):
                                for kdx,eachLinePose in enumerate(centrePoses[1:]):
                                    prevCx = eachLinePose[0]
                                    prevCy = eachLinePose[1]
                                    Cx = centrePoses[kdx][0]
                                    Cy = centrePoses[kdx][1]
                                    if(display == 1):
                                        radar_cartesian = cv2.line(radar_cartesian, (prevCx, prevCy), (Cx, Cy), (0,0,255), 3)

                            if(display == 1):
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
                                cv2.waitKey(1)
                                
                                imageCount = imageCount+1
                                imagePath = imageSaveFolder + str(imageCount) + '.png'
                                # cv2.imwrite(imagePath, finalImage)



                # Smooth the jerky trajectories using particle Filter
                # Initialize Particle Filter for smooth trajectory

                if (len(centrePoses) < 3):
                    continue

                smothedTraj = []
                particleCount = 500
                intialIndex = 3  # 2
                intitalCovariance = 2 #2
                pfObj = PFHelper.ParticleFilter(particleCount,[],'Classical',centrePoses[intialIndex][0],centrePoses[intialIndex][1],intitalCovariance)

                trajLength = len(centrePoses)
                for mdx in range(0,trajLength):
                    # Extract the current poses
                    currentPoseX = centrePoses[mdx][0]
                    currentPoseY = centrePoses[mdx][1]
                    # Add the first two poses 
                    if mdx<intialIndex:
                        smothedTraj.append([int(currentPoseX),int(currentPoseY)])
                        continue
                    # Extract the prev poses for average velcity calculation
                    prev1PoseX = centrePoses[mdx-1][0]
                    prev1PoseY = centrePoses[mdx-1][1]
                    prev2PoseX = centrePoses[mdx-2][0]
                    prev2PoseY = centrePoses[mdx-2][1]
                    prev3PoseX = centrePoses[mdx-3][0]
                    prev3PoseY = centrePoses[mdx-3][1]
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

                    pfObj.update([currentPoseX,currentPoseY], avgVx, avgVy)
                    filteredX = int(pfObj.particleMean[0])
                    filteredY = int(pfObj.particleMean[1])
                    smothedTraj.append([int(filteredX),int(filteredY)])

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

                # Draw the raw to check filter
                # centrePosesArray = np.array(centrePoses)
                # globalRadarDisplayImage = cv2.polylines(globalRadarDisplayImage,[centrePosesArray], False, (255,0,0), 2)
                # smothedTrajArray = np.array(smothedTraj)
                # globalRadarDisplayImage = cv2.polylines(globalRadarDisplayImage,[smothedTrajArray], False, (0,255,0), 2)


                # Draw the color coded trajectory 
                lineThickness = 3
                originLaneIndex = -1
                for trajIndex in range(3,trajLength):
                    currentPose = smothedTraj[trajIndex]
                    prev1Pose = smothedTraj[trajIndex-1]
                    prev2Pose = smothedTraj[trajIndex-2]
                    prev3Pose = smothedTraj[trajIndex-3]

                    prev1Angle = math.degrees(math.atan2((prev1Pose[1]-currentPose[1]),(prev1Pose[0]-currentPose[0])))
                    prev2Angle = math.degrees(math.atan2((prev2Pose[1]-currentPose[1]),(prev2Pose[0]-currentPose[0])))
                    prev3Angle = math.degrees(math.atan2((prev3Pose[1]-currentPose[1]),(prev3Pose[0]-currentPose[0])))

                    if prev1Angle < 0 : prev1Angle = prev1Angle + 360
                    if prev2Angle < 0 : prev2Angle = prev2Angle + 360
                    if prev3Angle < 0 : prev3Angle = prev3Angle + 360

                    avgAngle = (prev1Angle+prev2Angle+prev3Angle)/3

                    # # atan2 retunrs negetive angles as well. If negetive add 360 to convert to positive quadrant 
                    # if(avgAngle<0):
                    #     avgAngle = avgAngle + 360

                    # cv2.line(globalRadarDisplayImage, (prev1Pose[0], prev1Pose[1]), (currentPose[0], currentPose[1]), (0,255,0), lineThickness)

                    # # Based on the angle change the color
                    # # Extract last 3 angles to do avg
                    # currentAngle = currentVehicleAngleList[poseIndex]
                    # prevAngle = currentVehicleAngleList[poseIndex-1]
                    # prevToPrevAngle = currentVehicleAngleList[poseIndex-2]

                    # avgAngle = (currentAngle+prevAngle+prevToPrevAngle)/3



                    # Calculate distance from current point from each lane lines 
                    nearestLaneIndex = -1
                    lowestDist = 9999
                    for laneIndex,eacLine in enumerate(laneLines):
                        p1 = np.asarray((eacLine[0],eacLine[1]))
                        p2 = np.asarray((eacLine[2],eacLine[3]))
                        p3 = np.asarray((currentPose[0],currentPose[1]))
                        
                        # d = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
                        # d = np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)
                        # d = np.abs(np.cross(p2-p1, p3-p1))/np.linalg.norm(p2-p1)
                        d = point_to_line_dist(p3, [p1, p2])

                        if(d < lowestDist):
                            lowestDist = d
                            nearestLaneIndex = laneIndex

                    # Ignor the cars originating at 8,9,10,11 or vehicle ID is 49 (as this is a weired traj)
                    if(nearestLaneIndex == 8 or nearestLaneIndex == 9 or nearestLaneIndex == 10 or nearestLaneIndex == 11 or vehicleID == 49):
                        print('Another car ignored!!!')
                        ignoredCars = ignoredCars + 1
                        break
                    
                    # Update the origin lane index for left/right estimtion
                    if(trajIndex == 3):
                        originLaneIndex = nearestLaneIndex

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

                    if(display == 1):
                        cv2.line(mapImage, (prev3Pose[0], prev3Pose[1]), (prev2Pose[0], prev2Pose[1]), angleColor, lineThickness)
                        cv2.line(globalRadarDisplayImage, (prev3Pose[0], prev3Pose[1]), (prev2Pose[0], prev2Pose[1]), angleColor, lineThickness)

                    # Add the vehicle current information in the list
                    smoothVx = prev3Pose[0] - prev2Pose[0]
                    smoothVy = prev3Pose[1] - prev2Pose[1]
                    currentInfo = [prev2Pose[0], prev2Pose[1], smoothVx, smoothVy, lowestDist, nearestLaneIndex, movementInfoFloat, prev3Angle]
                    currentvehicleInfoList.append(currentInfo)

                    if(display == 1):
                        # # Draw Lane lines
                        for laneIndex,eacLine in enumerate(laneLines):
                            mapImage = cv2.putText(mapImage, str(laneIndex), (int((eacLine[0]+eacLine[2])/2),int((eacLine[1]+eacLine[3])/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                            if(laneIndex == nearestLaneIndex):

                                mapImage = cv2.line(mapImage, (eacLine[0], eacLine[1]), (eacLine[2], eacLine[3]), (0,255,255), 2)
                            else:
                                mapImage = cv2.line(mapImage, (eacLine[0], eacLine[1]), (eacLine[2], eacLine[3]), (255,0,0), 2)
                        

                        finalImage = np.hstack((radar_cartesian,mapImage))

                        cv2.imshow('image', finalImage)
                        cv2.waitKey(50)

                        imageCount = imageCount+1
                        imagePath = imageSaveFolder + str(imageCount) + '.png'
                        # cv2.imwrite(imagePath, finalImage)

                        # cv2.line(globalRadarDisplayImage, (prev2Pose[0], prev2Pose[1]), (prev1Pose[0], prev1Pose[1]), angleColor, lineThickness)
                        # cv2.line(globalRadarDisplayImage, (prev1Pose[0], prev1Pose[1]), (currentPose[0], currentPose[1]), angleColor, lineThickness)

                # Add the current vehicle list in the final list
                # If the "currentvehicleInfoList" is not empty then add the local list in the main list
                if(len(currentvehicleInfoList) > 0):
                    carInfoList.append(currentvehicleInfoList)
                # # Draw the traj on the map
                # mapTraj = []
                # for trajPoses in smothedTraj:

                #     # Extract the pose w.r.t to centre co-ordinate frame
                #     smoothPoseX = trajPoses[0]-radarImageCentre
                #     smoothPoseY = trajPoses[1]-radarImageCentre

                #     # Convert the poses to real meters
                #     poseXMeters = smoothPoseX*cellResolution
                #     poseYMeters = smoothPoseY*cellResolution

                #     # Convert the meter poses to map pixel location
                #     poseMapX = int((mapWidth/2) + poseXMeters/metersPerPixel)
                #     poseMapY = int((mapHeight/2) + poseYMeters/metersPerPixel)

                #     mapTraj.append([int(trajPoses[0]),int(trajPoses[1])])

                # # Draw the generated trajectory on the map image                
                # mapTrajArray = np.array(mapTraj)
                # resizedMap = cv2.polylines(resizedMap,[mapTrajArray], False, (0,0,255), 3)


                # finalImageTraj = np.hstack((globalRadarDisplayImage,mapImage))
                # imageCount = imageCount+1
                # imagePath = imageSaveFolder + str(imageCount) + '.png'
                # cv2.imwrite(imagePath, finalImageTraj)
                # cv2.imshow('traj', resizedMap)
                # # cv2.imshow('smooth', finalImageTraj)
                # cv2.waitKey(1)

# Prepare the input, decoderInput and output data
def DataPrep():

    # Access the global min max values
    # Global Min max for X/Y poses
    global globalMaxXPose, globalMaxYPose, globalMinXPose, globalMinYPose
    # Global Min max for Velocity
    global globalMaxXVelocity, globalMaxYVelocity, globalMinXVelocity, globalMinYVelocity
    # Global Min max for Lane Dist
    global globalMinLaneDist, globalMaxLaneDist

    # Initialize the input output list
    xTrainList = []
    decoderTrainList = []
    YClassTrainList = []
    YVelTrainList = []
    YPoseTrainList = []

    # For each item in the main list get the min max values for later normlization
    for eachCar in carInfoList:

        eachCarArray = np.array(eachCar)

        # Calculate the min max for the current vehicle
        # Min max for poses
        maxXPose = max(eachCarArray[:,poseXIndex])
        maxYPose = max(eachCarArray[:,poseYIndex])
        minXPose = min(eachCarArray[:,poseXIndex])
        minYPose = min(eachCarArray[:,poseYIndex])

        # Min max for Velocity
        maxXVelocity = max(eachCarArray[:,velXIndex])
        maxYVelocity = max(eachCarArray[:,velYIndex])
        minXVelocity = min(eachCarArray[:,velXIndex])
        minYVelocity = min(eachCarArray[:,velYIndex])

        minLaneDist = min(eachCarArray[:,laneDistIndex])
        maxLaneDist = max(eachCarArray[:,laneDistIndex])

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
                currentMovement = eachTrainCar[jdx][movementIndex]
                currentAngle =  eachTrainCar[jdx][angelIndex]
                # Normalize each field
                normCurrentPoseX = (currentPoseX-globalMinXPose)/(globalMaxXPose-globalMinXPose)
                normCurrentPoseY = (currentPoseY-globalMinYPose)/(globalMaxYPose-globalMinYPose)
                normCurrentVx = (currentVx-globalMinXVelocity)/(globalMaxXVelocity-globalMinXVelocity)
                normCurrentVy = (currentVy-globalMinYVelocity)/(globalMaxYVelocity-globalMinYVelocity) 
                normCurrentLaneDist = (currentLaneDist-globalMinLaneDist)/(globalMaxLaneDist-globalMinLaneDist)
                normCurrentLaneNumber = currentLaneNumber/globalMaxLanenumber 
                normCurrentMovement = currentMovement
                normAngle = currentAngle/360

                # Append the input info to local list
                localInput.append([normCurrentPoseX,normCurrentPoseY,normCurrentVx,normCurrentVy,normCurrentLaneDist,normCurrentLaneNumber,normCurrentMovement,normAngle])

            
            # Get the last input for the decoder first input (use the variable as they have the last instance)
            # Convert the movement info to class form
            movementClassInfo = MovementToClassForm(normCurrentMovement)
            firstDecoderInput = [movementClassInfo[0],movementClassInfo[1],movementClassInfo[2],normCurrentVx,normCurrentVy,normCurrentPoseX,normCurrentPoseY]
            
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
                currentMovement = eachTrainCar[kdx][movementIndex]
                # Convert the movement info to class form
                movementClassInfo = MovementToClassForm(currentMovement)
                # Append the input info to local list
                localClassOut.append([movementClassInfo[0],movementClassInfo[1],movementClassInfo[2]])
                localVelOut.append([currentVx,currentVy])
                localPoseOut.append([currentPoseX,currentPoseY])
                # normalize the fields before appending to the Decoder
                normCurrentMovement = movementClassInfo 
                normCurrentPoseX = (currentPoseX-globalMinXPose)/(globalMaxXPose-globalMinXPose)
                normCurrentPoseY = (currentPoseY-globalMinYPose)/(globalMaxYPose-globalMinYPose)
                normCurrentVx = (currentVx-globalMinXVelocity)/(globalMaxXVelocity-globalMinXVelocity)
                normCurrentVy = (currentVy-globalMinYVelocity)/(globalMaxYVelocity-globalMinYVelocity) 
                localDecoderInput.append([movementClassInfo[0],movementClassInfo[1],movementClassInfo[2],normCurrentVx,normCurrentVy,normCurrentPoseX,normCurrentPoseY])

            # Prepeare the decoder input (add the first item and remove the last item)
            localDecoderInput.insert(0,firstDecoderInput)
            localDecoderInput.pop()

            # Append each datapoint to the main list
            xTrainList.append(localInput)
            decoderTrainList.append(localDecoderInput)
            YClassTrainList.append(localClassOut)
            YVelTrainList.append(localVelOut)
            YPoseTrainList.append(localPoseOut)

    # Convert all lists to array and return.
    xTrainArray = np.array(xTrainList)
    decoderTrainArray = np.array(decoderTrainList)
    YClassTrainArray = np.array(YClassTrainList)
    YVelTrainArray = np.array(YVelTrainList)
    YPoseTrainArray = np.array(YPoseTrainList)

    return xTrainArray,decoderTrainArray,YClassTrainArray,YVelTrainArray,YPoseTrainArray

# Define the Custome learing rate decays
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 3.0
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

# Create the model
# def TrainModel(XTrain,decoderTrainInput,YClassTrain,YVelTrain,YPoseTrain):
def CreateModel():    

    n_units = 256

    # define training encoder
    encoder_inputs = Input(shape=(None, inputFeature))
    # First Encoder LSTM Layer
    encoder1 = LSTM(n_units, return_state=True, return_sequences=True)
    encoder_output, state_h1, state_c1 = encoder1(encoder_inputs)
    encoder_states1 = [state_h1, state_c1]
    # Second Encoder LSTM Layer
    encoder2 = LSTM(n_units, return_state=True)
    encoder_output, state_h2, state_c2 = encoder2(encoder_output)
    encoder_states2 = [state_h2, state_c2]
	# define training decoder
    decoder_inputs = Input(shape=(None, decoderInputFeature))
    # First Decoder LSTM Layer
    decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm1(decoder_inputs, initial_state=encoder_states1)
    # Second Decoder LSTM Layer
    decoder_lstm2 = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm2(decoder_outputs, initial_state=encoder_states2)

    encoder_states = [state_h1, state_c1, state_h2, state_c2]

    # Decoder for ClassOut
    decoder_dense10a = Dense(1024)
    decoder_Leaky10a = LeakyReLU(alpha=leakyAlphaValue)
    decoder_output1 = decoder_dense10a(decoder_outputs)
    decoder_output1 = decoder_Leaky10a(decoder_output1)
    decoder_dense10 = Dense(512)
    decoder_Leaky10 = LeakyReLU(alpha=leakyAlphaValue)
    decoder_output1 = decoder_dense10(decoder_output1)
    decoder_output1 = decoder_Leaky10(decoder_output1)
    decoder_dense11 = Dense(256)
    decoder_Leaky11 = LeakyReLU(alpha=leakyAlphaValue)
    decoder_output1 = decoder_dense11(decoder_output1)
    decoder_output1 = decoder_Leaky11(decoder_output1)
    decoder_dense12 = Dense(128)
    decoder_Leaky12 = LeakyReLU(alpha=leakyAlphaValue)
    decoder_output1 = decoder_dense12(decoder_output1)
    decoder_output1 = decoder_Leaky12(decoder_output1)
    decoder_dense13 = Dense(64)
    decoder_Leaky13 = LeakyReLU(alpha=leakyAlphaValue)
    decoder_output1 = decoder_dense13(decoder_output1)
    decoder_output1 = decoder_Leaky13(decoder_output1)
    decoder_dense14 = Dense(32)
    decoder_Leaky14 = LeakyReLU(alpha=leakyAlphaValue)
    decoder_output1 = decoder_dense14(decoder_output1)
    decoder_output1 = decoder_Leaky14(decoder_output1)
    decoder_dense15 = Dense(3, activation='softmax', name='Class')
    classOut = decoder_dense15(decoder_output1)

    # Decoder for Velocity Out
    decoder2_concat = Concatenate()
    decoder_output2 = decoder2_concat([decoder_outputs,classOut])
    decoder_dense20a = Dense(1024)
    decoder_Leaky20a = LeakyReLU(alpha=leakyAlphaValue)
    decoder_output2 = decoder_dense20a(decoder_output2)
    decoder_output2 = decoder_Leaky20a(decoder_output2)
    decoder_dense20 = Dense(512)
    decoder_Leaky20 = LeakyReLU(alpha=leakyAlphaValue)
    decoder_output2 = decoder_dense20(decoder_output2)
    decoder_output2 = decoder_Leaky20(decoder_output2)
    decoder_dense21 = Dense(256)
    decoder_Leaky21 = LeakyReLU(alpha=leakyAlphaValue)
    decoder_output2 = decoder_dense21(decoder_output2)
    decoder_output2 = decoder_Leaky21(decoder_output2)
    decoder_dense22 = Dense(128)
    decoder_Leaky22 = LeakyReLU(alpha=leakyAlphaValue)
    decoder_output2 = decoder_dense22(decoder_output2)
    decoder_output2 = decoder_Leaky22(decoder_output2)
    decoder_dense23 = Dense(64)
    decoder_Leaky23 = LeakyReLU(alpha=leakyAlphaValue)
    decoder_output2 = decoder_dense23(decoder_output2)
    decoder_output2 = decoder_Leaky23(decoder_output2)
    decoder_dense24 = Dense(32)
    decoder_Leaky24 = LeakyReLU(alpha=leakyAlphaValue)
    decoder_output2 = decoder_dense24(decoder_output2)
    decoder_output2 = decoder_Leaky24(decoder_output2)
    decoder_dense25 = Dense(2, activation='linear', name='Velcoity')
    velocityOut = decoder_dense25(decoder_output2)

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
    # decoder_output3 = decoder3_concat([decoder_outputs,classOut])
    decoder_output3 = decoder3_concat([decoder_outputs,classOut,velocityConcatX,velocityConcatY])
    decoder_dense30b = Dense(2048)
    decoder_Leaky30b = LeakyReLU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense30b(decoder_output3)
    decoder_output3 = decoder_Leaky30b(decoder_output3)
    decoder_dense30a = Dense(1024)
    decoder_Leaky30a = LeakyReLU(alpha=leakyAlphaValue)
    decoder_output3 = decoder_dense30a(decoder_output3)
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

    # Inference decoder for Class out

    decoder_output1 = decoder_dense10a(decoder_outputs)
    decoder_output1 = decoder_Leaky10a(decoder_output1)
    decoder_output1 = decoder_dense10(decoder_output1)
    decoder_output1 = decoder_Leaky10(decoder_output1)
    decoder_output1 = decoder_dense11(decoder_output1)
    decoder_output1 = decoder_Leaky11(decoder_output1)
    decoder_output1 = decoder_dense12(decoder_output1)
    decoder_output1 = decoder_Leaky12(decoder_output1)
    decoder_output1 = decoder_dense13(decoder_output1)
    decoder_output1 = decoder_Leaky13(decoder_output1)
    decoder_output1 = decoder_dense14(decoder_output1)
    decoder_output1 = decoder_Leaky14(decoder_output1)
    classOut = decoder_dense15(decoder_output1)

    # Inference Decoder for Velocity Out
    decoder_output2 = decoder2_concat([decoder_outputs,classOut])
    decoder_output2 = decoder_dense20a(decoder_output2)
    decoder_output2 = decoder_Leaky20a(decoder_output2)
    decoder_output2 = decoder_dense20(decoder_output2)
    decoder_output2 = decoder_Leaky20(decoder_output2)
    decoder_output2 = decoder_dense21(decoder_output2)
    decoder_output2 = decoder_Leaky21(decoder_output2)
    decoder_output2 = decoder_dense22(decoder_output2)
    decoder_output2 = decoder_Leaky22(decoder_output2)
    decoder_output2 = decoder_dense23(decoder_output2)
    decoder_output2 = decoder_Leaky23(decoder_output2)
    decoder_output2 = decoder_dense24(decoder_output2)
    decoder_output2 = decoder_Leaky24(decoder_output2)
    velocityOut = decoder_dense25(decoder_output2)

    # Inference Decoder Velocity Normalizer
    velocityOutX = velocityExtractX(velocityOut)
    velocityOutY = velocityExtractY(velocityOut)

    velocityConcatX = velocityXNormalized(velocityOutX)
    velocityConcatY = velocityYNormalized(velocityOutY)

    #Inference  Decoder for position out
    decoder_output3 = decoder3_concat([decoder_outputs,classOut,velocityConcatX,velocityConcatY])
    decoder_output3 = decoder_dense30b(decoder_output3)
    decoder_output3 = decoder_Leaky30b(decoder_output3)
    decoder_output3 = decoder_dense30a(decoder_output3)
    decoder_output3 = decoder_Leaky30a(decoder_output3)
    decoder_output3 = decoder_dense30(decoder_output3)
    decoder_output3 = decoder_Leaky30(decoder_output3)
    decoder_output3 = decoder_dense31(decoder_output3)
    decoder_output3 = decoder_Leaky31(decoder_output3)
    decoder_output3 = decoder_dense32(decoder_output3)
    decoder_output3 = decoder_Leaky32(decoder_output3)
    decoder_output3 = decoder_dense33(decoder_output3)
    decoder_output3 = decoder_Leaky33(decoder_output3)
    decoder_output3 = decoder_dense34(decoder_output3)
    decoder_output3 = decoder_Leaky34(decoder_output3)
    positionOut = decoder_dense35(decoder_output3)

    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [classOut, velocityOut, positionOut] + decoder_states)

    opt = RMSprop()

    model.compile(optimizer=opt, loss=['categorical_crossentropy', 'mean_squared_error', euclidean_distance_loss])
    # model.compile(optimizer=opt, loss=['categorical_crossentropy', logcosh, euclidean_distance_loss])
    model.summary()

    # Return the encoder and decoder model
    return model,encoder_model,decoder_model


# Fit the model
def FitModel(model, encoder_model, decoder_model, XTrain,decoderTrainInput,YClassTrain,YVelTrain,YPoseTrain):

    # Custom decay rates
    # loss_history = LossHistory()
    lrate = LearningRateScheduler(step_decay)
    esObj = EarlyStoppingByLossVal()
    callbacks_list = [esObj,lrate]   #[loss_history, lrate]
    

    print('XTrain Shape : ' + str(XTrain.shape))
    print('DecoderInput Shape : ' + str(decoderTrainInput.shape))
    print('YClassTrain Shape : ' + str(YClassTrain.shape))
    print('YVelTrain Shape : ' + str(YVelTrain.shape))
    print('YPoseTrain Shape : ' + str(YPoseTrain.shape))
    # print('XVal Shape : ' + str(XVal.shape))
    # print('decoderValInput Shape : ' + str(decoderValInput.shape))
    # print('YClassVal Shape : ' + str(YClassVal.shape))
    # print('YVelVal Shape : ' + str(YVelVal.shape))
    # print('YPoseVal Shape : ' + str(YPoseVal.shape))

    model.fit([XTrain,decoderTrainInput], [YClassTrain,YVelTrain,YPoseTrain], batch_size=batchSize, epochs=nepochs, verbose=1, validation_data=([XTrain,decoderTrainInput],[YClassTrain,YVelTrain,YPoseTrain]), callbacks=callbacks_list)

    return model,encoder_model,decoder_model

# Test the model
def TestModel(encoder_model,decoder_model):

    lineThickness = 3

    for eachTrainCar in carInfoList:
        currentVehicleLength = len(eachTrainCar)
        for idx in range(historyTemporal,currentVehicleLength-futureTemporal,historyTemporal):
            # Load the radar and map image and draw the lane lines for visualization
            resultRadarImage = cv2.imread(radarPath)
            # resultRadarImage = cv2.cvtColor(resultRadarImage, cv2.COLOR_GRAY2BGR)
            resultMapImage = cv2.imread(mapImagePath)
            for laneIndex,eacLine in enumerate(laneLines):
                resultMapImage = cv2.line(resultMapImage, (eacLine[0], eacLine[1]), (eacLine[2], eacLine[3]), (255,0,0), 2)
            
            # Add the color code for input, groundtruth and predicted trajectory
            resultMapImage = cv2.putText(resultMapImage, 'Red : Input Traj', (750,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            resultMapImage = cv2.putText(resultMapImage, 'Green : Predicted', (750,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            resultMapImage = cv2.putText(resultMapImage, 'Blue : Ground Truth', (750,140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

            # Prepeare the input
            localInput = []
            for jdx in range(idx-historyTemporal,idx):
                currentPoseX = eachTrainCar[jdx][poseXIndex]
                currentPoseY = eachTrainCar[jdx][poseYIndex]
                currentVx = eachTrainCar[jdx][velXIndex]
                currentVy = eachTrainCar[jdx][velYIndex]
                currentLaneDist = eachTrainCar[jdx][laneDistIndex]
                currentLaneNumber = eachTrainCar[jdx][laneNumberIndex]
                currentMovement = eachTrainCar[jdx][movementIndex]
                currentAngle = eachTrainCar[jdx][angelIndex]
                # Normalize each field
                normCurrentPoseX = (currentPoseX-globalMinXPose)/(globalMaxXPose-globalMinXPose)
                normCurrentPoseY = (currentPoseY-globalMinYPose)/(globalMaxYPose-globalMinYPose)
                normCurrentVx = (currentVx-globalMinXVelocity)/(globalMaxXVelocity-globalMinXVelocity)
                normCurrentVy = (currentVy-globalMinYVelocity)/(globalMaxYVelocity-globalMinYVelocity) 
                normCurrentLaneDist = (currentLaneDist-globalMinLaneDist)/(globalMaxLaneDist-globalMinLaneDist)
                normCurrentLaneNumber = currentLaneNumber/globalMaxLanenumber 
                normCurrentMovement = currentMovement 
                normCurrnetAngle = currentAngle/360

                # Append the input info to local list
                localInput.append([normCurrentPoseX,normCurrentPoseY,normCurrentVx,normCurrentVy,normCurrentLaneDist,normCurrentLaneNumber,normCurrentMovement,normCurrnetAngle])

                # Draw the input traj using poses and movement
                if(jdx > 0):
                    prevPoseX = eachTrainCar[jdx-1][poseXIndex]
                    prevPoseY = eachTrainCar[jdx-1][poseYIndex]
                    angleColor = (0,0,255)
                    # Check the movement info and decide the traj color
                    if(currentMovement == 0):
                        angleColor = (0,0,255)
                    resultRadarImage = cv2.line(resultRadarImage, (prevPoseX, prevPoseY), (currentPoseX, currentPoseY), angleColor, lineThickness)
                    resultMapImage = cv2.line(resultMapImage, (prevPoseX, prevPoseY), (currentPoseX, currentPoseY), angleColor, lineThickness)

                # Concat the images for visualitzation
                finalImage = np.hstack((resultRadarImage,resultMapImage))
                cv2.imshow('image', finalImage)
                cv2.waitKey(0)

            
            # Get the last input for the decoder first input (use the variable as they have the last instance)
            normMovementInfo = MovementToClassForm(normCurrentMovement)
            firstDecoderInput = [normMovementInfo[0],normMovementInfo[1],normMovementInfo[2],normCurrentVx,normCurrentVy,normCurrentPoseX,normCurrentPoseY]

            # Gather the ground truth pose array
            groundTruthPose = []
            for kdx in range(idx,idx+futureTemporal):
                currentPoseX = eachTrainCar[kdx][poseXIndex]
                currentPoseY = eachTrainCar[kdx][poseYIndex]
                currentVx = eachTrainCar[kdx][velXIndex]
                currentVy = eachTrainCar[kdx][velYIndex]
                currentMovement = eachTrainCar[kdx][movementIndex]
                # Append the ground truth info to local list
                groundTruthPose.append([currentPoseX,currentPoseY])

            # Format the input array for prediction
            localInputArray = np.array(localInput).reshape(1,historyTemporal,inputFeature)
            target_seq = np.array(firstDecoderInput).reshape(1,1,decoderInputFeature)
            
            # Start the prediction
            # Predict the encoder state
            state = encoder_model.predict(localInputArray)

            # Declare list to gather the predicted pose and the calculated error
            outputPredPose = []
            currentError = []

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

                # update state
                state = [h1, c1, h2, c2]

                # update target sequence
                # Update the target sequence till second last frame. At the last frame no need to update the seq as it will not be used
                if(t<(futureTemporal-1)):
                    targetDecoder = [classPred[0][0][0],classPred[0][0][1],classPred[0][0][2],normalizedPredVelocityX,normalizedPredVelocityY,normalizedPredPoseX,normalizedPredPoseY]
                    target_seq = np.array(targetDecoder).reshape(1,1,decoderInputFeature)

                # Calculate the Euclidian Error
                truePoseX = groundTruthPose[t][0]
                truePoseY = groundTruthPose[t][1]

                predPoseX = outputPredPose[t][0]
                predPoseY = outputPredPose[t][1]

                euclidianError = math.sqrt(((truePoseX-predPoseX)**2) + ((truePoseY-predPoseY)**2))
                euclidianErrorMeter = euclidianError*cellResolution

                currentError.append(euclidianErrorMeter)

            # Draw the predicted and ground truth trajetory on the map and radar image
            # Check the length of ground truth and predicted pose length same
            predPoseLength = len(outputPredPose)
            groundTruthLength = len(groundTruthPose)
            if(predPoseLength != groundTruthLength):
                print('predpose and groundTruthPose are not equal!!!')
                print('predPose Length = ' + str(predPoseLength))
                print('groundTruthPose Length = ' + str(groundTruthLength))
                sys.exit()

            trueColor = (255,0,0) # blue
            predColor = (0,255,0) # green
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

                # Draw the true and predicted on the radar image
                resultRadarImage = cv2.line(resultRadarImage, (prevPredPoseX, prevPredPoseY), (currentPredPoseX, currentPredPoseY), predColor, lineThickness)
                resultRadarImage = cv2.line(resultRadarImage, (prevTruePoseX, prevTruePoseY), (currentTruePoseX, currentTruePoseY), trueColor, lineThickness)

                # Draw the true and predicted on the map image
                resultMapImage = cv2.line(resultMapImage, (prevPredPoseX, prevPredPoseY), (currentPredPoseX, currentPredPoseY), predColor, lineThickness)
                resultMapImage = cv2.line(resultMapImage, (prevTruePoseX, prevTruePoseY), (currentTruePoseX, currentTruePoseY), trueColor, lineThickness)

                # Concat the images for visualitzation
                finalImage = np.hstack((resultRadarImage,resultMapImage))
                cv2.imshow('image', finalImage)
                cv2.waitKey(0)


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
    print('Totoal number of ignored cars : ' + str(ignoredCars) + ' !!!')

    # Prepeare the data and get the input output arrays
    xTrainArray,decoderTrainArray,YClassTrainArray,YVelTrainArray,YPoseTrainArray = DataPrep() 

    print('XTrain Shape : ' + str(xTrainArray.shape))
    print('decoderTrainArray Shape : ' + str(decoderTrainArray.shape))
    print('YClassTrainArray Shape : ' + str(YClassTrainArray.shape))
    print('YVelTrainArray Shape : ' + str(YVelTrainArray.shape))
    print('YPoseTrainArray Shape : ' + str(YPoseTrainArray.shape))
    print('Data prep done!!!')

    if(trainOrTest == trainStr):
        print('Starting traininig !!!')

        # Create the model and train the model
        model,encoder_model,decoder_model = CreateModel()
        model,encoder_model,decoder_model = FitModel(model,encoder_model,decoder_model,xTrainArray,decoderTrainArray,YClassTrainArray,YVelTrainArray,YPoseTrainArray)

        # Save the model
        encoder_model.save_weights('./NavtechModels/encoder.h5')
        decoder_model.save_weights('./NavtechModels/decoder.h5')
 
       
    elif(trainOrTest == testStr):
        print('Starting testing !!!')

        model, encoder_model, decoder_model = CreateModel()
        encoder_model.load_weights('/home/saptarshi/PythonCode/Junction/NavtechModels/encoder.h5')
        decoder_model.load_weights('/home/saptarshi/PythonCode/Junction/NavtechModels/decoder.h5')

        # Test the model
        TestModel(encoder_model,decoder_model)


    # normalizeIndexList = [poseXIndex,poseYIndex,velXIndex,velYIndex,laneDistIndex,laneNumberIndex,movementIndex]

    
__main__()
