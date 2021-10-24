# Extract the labbeled bounding boxes and create OGMs from training.
# This to draw al the trajectories on both radar and map image
import cv2
import os
import numpy as np
# import matplotlib.pyplot as plt
import json
import PFHelper
# import pandas as pd
import math
import scipy.interpolate as interp
# import matplotlib.pyplot as plt
# import pptk

sequence_folder = '/home/saptarshi/Dropbox (Heriot-Watt University Team)/RES_EPS_PathCad/datasets/pathcad_van_dataset/junction_1'
# sequence_folder = '/home/saptarshi/PythonCode/Junction/NavtechSample'
radarImagePath = '/home/saptarshi/Dropbox (Heriot-Watt University Team)/RES_EPS_PathCad/datasets/pathcad_van_dataset/junction_1/junction_1_0/Navtech_Cartesian/000001.png'
mapImagePath = '/home/saptarshi/PythonCode/Junction/AlvieMap.png'
globalRadarImage = cv2.imread(radarImagePath)
globalMapImage = cv2.imread(mapImagePath)

saveImageFilePath = '/home/saptarshi/PythonCode/Junction/CaseStudy/AlvieCount.png'

cv2.namedWindow('image',cv2.WINDOW_NORMAL)

# Main road
laneLines =   [[537,287,519,1021], [537,287,690,286], [537,287,522,63]]

globalLeftTurn = 0
globallRightTrun = 0
globalStraight = 0

turn = 0
straight = 0
processedObjectIds = []
str_format = '{:06d}'
straightStr = 'Straight'
turnStr = 'Turn'
highestFrameCount = 0
straightAngles = [0,90,180,270,360]


# Smooth the trajectories using particle filter
def SmoothTrajcetoryGeneration(poseList):

    smothedTraj = []
    particleCount = 500
    intialIndex = 3  # 2
    intitalCovariance = 2 #2
    pfObj = PFHelper.ParticleFilter(particleCount,[],'Classical',poseList[intialIndex][0],poseList[intialIndex][1],intitalCovariance)

    trajLength = len(poseList)
    for mdx in range(0,trajLength):
        # Extract the current poses
        currentPoseX = poseList[mdx][0]
        currentPoseY = poseList[mdx][1]
        # Add the first two poses
        if mdx<intialIndex:
            smothedTraj.append([int(currentPoseX),int(currentPoseY)])
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

        pfObj.update([currentPoseX,currentPoseY], avgVx, avgVy)
        filteredX = int(pfObj.particleMean[0])
        filteredY = int(pfObj.particleMean[1])
        smothedTraj.append([int(filteredX),int(filteredY)])

    # Return the smoothed trajectory
    return smothedTraj

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
                                radar_cartesian = cv2.putText(radar_cartesian, str(object['id']+idOffset), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA) 
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
                                        globalRadarDisplayImage = cv2.line(globalRadarDisplayImage, (cx,cy), (prevCx,prevCy), (0,0,255), 2)
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

            finalImage = np.hstack((radar_cartesian,globalRadarDisplayImage))

            cv2.imshow('image', finalImage)
            cv2.waitKey(10)

        return finalImage


    def playByVehicle(self,sequence_path,idOffset):

        global globalRadarImage,globalMapImage,highestFrameCount,globalLeftTurn,globallRightTrun,globalStraight

        if (self.annotations != None):
            for object in self.annotations:

                # Read the trajectory from the annotation files
                centrePoses = []
                allBoundingBoxes = object['bboxes']
                allVisibleParam = object['visible']
                allDeleteParam = object['deleted']

                for idx,eachBbbox in enumerate(allBoundingBoxes):
                    if (eachBbbox):
                        if ((allDeleteParam[idx] == 0) and (allVisibleParam[idx] == 'visible')):

                            bbox = eachBbbox['position']
                            angle = eachBbbox['rotation']

                            cx = int(bbox[0] + bbox[2]/2)
                            cy = int(bbox[1] + bbox[3]/2)
                            centrePoses.append((cx,cy))

                # Smoothe the trajectory
                poseLen = len(centrePoses)
                if (poseLen < 3):
                    continue

                smoothTraj = SmoothTrajcetoryGeneration(centrePoses)

                smoothTrajLength = len(smoothTraj)
                maneuverList = []

                # Draw and visualize the trajectories
                for trajIndex in range(3,smoothTrajLength):
                    # Extract the current and prev poses
                    currentPose = smoothTraj[trajIndex]
                    prev2Pose = smoothTraj[trajIndex-2]
                    prev3Pose = smoothTraj[trajIndex-3]
                    # Calculate the tangent for angle
                    prev3Angle = math.degrees(math.atan2((prev3Pose[1]-currentPose[1]),(prev3Pose[0]-currentPose[0])))
                    if prev3Angle < 0 : prev3Angle = prev3Angle + 360

                    # Estimate the nearest lane index and calculate the distance
                    nearestLaneIndex,lowestDist = CalculateNearestLaneAndDist(currentPose)

                    # Update the origin lane index for left/right estimtion
                    if(trajIndex == 3):
                        originLaneIndex = nearestLaneIndex

                    # Estimate the Turn or straight movement based on the angle
                    movementInfo,movementInfoFloat = CalculateManeuverInfo(prev3Angle,originLaneIndex)

                    # Append the maneuver list
                    maneuverList.append(movementInfoFloat)

                # Check the main maneuver 0.5 -> left turn and 1.0 -> right turn and update the global count
                # Set color code Straight Green, left turn red , right turn yellow
                leftTurnCount = maneuverList.count(0.5)
                rightTurnCount = maneuverList.count(1.0)
                mainManeuver = 0
                trajColor = (0,255,0)
                # Check if there is occurance of turn
                if(leftTurnCount > 5 or rightTurnCount > 5):
                    if(leftTurnCount > rightTurnCount):
                        mainManeuver = 0.5
                        trajColor = (0,255,255)
                    else:
                        mainManeuver = 1.0
                        trajColor = (0,0,255)

                # Update the global count
                if(mainManeuver == 0):
                    globalStraight = globalStraight + 1
                elif(mainManeuver == 0.5):
                    globalLeftTurn = globalLeftTurn + 1
                elif(mainManeuver == 1.0):
                    globallRightTrun = globallRightTrun + 1
                    continue
                else:
                    print('Unknown main maneuver!!!')
                    sys.exit()   

                # Ignore oopposite side car... 
                firstLocY = smoothTraj[0][1]
                if(firstLocY < 600):
                    continue 

                # Draw the trajectory with color code Straight Green, left turn red , right turn yellow
                lineThickness = 2
                for drawIndex in range(1,smoothTrajLength):

                    currentPoseX = smoothTraj[drawIndex][0]
                    currentPoseY = smoothTraj[drawIndex][1]
                    prevPoseX = smoothTraj[drawIndex-1][0]
                    prevPoseY = smoothTraj[drawIndex-1][1]

                    # Draw on Radar and Map image
                    globalRadarImage = cv2.line(globalRadarImage, (prevPoseX, prevPoseY), (currentPoseX, currentPoseY), trajColor, lineThickness)
                    globalMapImage = cv2.line(globalMapImage, (prevPoseX, prevPoseY), (currentPoseX, currentPoseY), trajColor, lineThickness)

                    finalImageTraj = np.hstack((globalRadarImage,globalMapImage))

                    cv2.imshow('image', finalImageTraj)
                    cv2.waitKey(1)

def __main__():

    global globalRadarImage,globalMapImage

    folderList = os.listdir(sequence_folder)
    folderList.sort(key=lambda x: int(x.split('_')[-1]))
    idOffset = 0
    # globalImageLoaded = False
    # global globalRadarDisplayImage,globalRadarSmoothTraj
    # averageVelocityGlobalList = []
    # velocityProfileGlobal = []

    for eachSeq in folderList:
        print('Processing folder : ' + eachSeq)
        # Ignore junction 3 as its giving nan for the filter....
        if(eachSeq == 'junction_1_3'):
            continue
        sequence_path = os.path.join(sequence_folder, eachSeq)
        annotation_path = os.path.join(sequence_path, 'annotations', 'annotations.json')


        sequence = Sequence(sequence_path)
        sequence.load_sequence(sequence_path)
        sequence.load_annotations(annotation_path)
        sequence.playByVehicle(sequence_path,idOffset)
        idOffset = idOffset + sorted([d['id'] for d in sequence.annotations])[-1]

    # Add the maeuver info on the image with same color code
    # Set color code Straight Green, left turn red , right turn yellow
    globalMapImage = cv2.putText(globalMapImage, 'Straight Cars:' + str(globalStraight), (700,50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 2, cv2.LINE_AA)
    globalMapImage = cv2.putText(globalMapImage, 'Right Cars:' + str(globalLeftTurn), (700,95), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,255), 2, cv2.LINE_AA)
    # globalMapImage = cv2.putText(globalMapImage, 'Right Cars:' + str(globallRightTrun), (700,140), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,255), 2, cv2.LINE_AA)
    # globalMapImage = cv2.putText(globalMapImage, 'Total Cars:' + str(globalStraight+globalLeftTurn+globallRightTrun), (700,185), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,20,147), 2, cv2.LINE_AA)

    globalRadarImage = cv2.putText(globalRadarImage, 'Straight Cars:' + str(globalStraight), (700,50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 2, cv2.LINE_AA)
    globalRadarImage = cv2.putText(globalRadarImage, 'Right Cars:' + str(globalLeftTurn), (700,95), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,255), 2, cv2.LINE_AA)
    # globalRadarImage = cv2.putText(globalRadarImage, 'Right Cars: ' + str(globallRightTrun), (700,140), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,255), 2, cv2.LINE_AA)
    # globalRadarImage = cv2.putText(globalRadarImage, 'Total Cars:' + str(globalStraight+globalLeftTurn+globallRightTrun), (700,185), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,20,147), 2, cv2.LINE_AA)

    # Stack the image side by side and save the image
    saveImage = np.hstack((globalRadarImage,globalMapImage))

    cv2.imshow('image', saveImage)
    cv2.waitKey(10)

    cv2.imwrite(saveImageFilePath,saveImage)




    
    # # # #cv2.imwrite('./countImage.png', showedImage)
    # # # # print(averageVelocityGlobalList)
    # # # print(len(averageVelocityGlobalList))

    # # # # Write the average velocities in a file 
    # # # with open('AverageVelocity.txt', 'w') as f:
    # # #     for item in averageVelocityGlobalList:
    # # #         f.write("%s\n" % item)

    # # # with open('profile.txt','w') as f:
    # # #     for sublist in velocityProfileGlobal:
    # # #         for item in sublist:
    # # #             f.write(str(item) + ',')
    # # #         f.write('\n')
    # # #     f.write('Highest Frame Length:' + str(highestFrameCount))

    # Seprate the turn and straight vehicles velocities 
    # turnVelocityList = []
    # straightVelocityList = []

    # for eachVelocity in averageVelocityGlobalList:
    #     if(eachVelocity[0] == straightStr):
    #         straightVelocityList.append(eachVelocity[1])
    #     elif(eachVelocity[0] == turnStr):
    #         turnVelocityList.append(eachVelocity[1])
    #     else:
    #         print('Unknown maneuver string')

    
__main__()
