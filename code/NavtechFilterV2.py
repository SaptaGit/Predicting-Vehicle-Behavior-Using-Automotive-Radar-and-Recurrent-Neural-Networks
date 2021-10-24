# Extract the labbeled bounding boxes and create OGMs from training.
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
# import pptk

# sequence_folder = '/home/saptarshi/Dropbox (Heriot-Watt University Team)/RES_EPS_PathCad/datasets/pathcad_van_dataset/junction_1'
sequence_folder = '/home/saptarshi/Dropbox (Heriot-Watt University Team)/RES_EPS_PathCad/datasets/pathcad_van_dataset/SighthillJunc'
globalRadarDisplayImage = np.zeros((1152,1152,3))
globalRadarSmoothTraj = np.zeros((1152,1152,3))
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
# cv2.namedWindow('traj',cv2.WINDOW_NORMAL)
turn = 0
straight = 0
processedObjectIds = []
str_format = '{:06d}'
straightStr = 'Straight'
turnStr = 'Turn'
imageCount = 0
straightAngles = [0,90,180,270,360]

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
                                        globalRadarDisplayImage = cv2.line(globalRadarDisplayImage, (cx,cy), (prevCx,prevCy), (0,255,0), 1)
                                        if (object['id']+idOffset) not in processedObjectIds:
                                            processedObjectIds.append((object['id']+idOffset))
                                            turn = turn + 1
                                    else:
                                        globalRadarDisplayImage = cv2.line(globalRadarDisplayImage, (cx,cy), (prevCx,prevCy), (0,255,0), 1)
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
            cv2.waitKey(1)

        return finalImage

    def playByVehicle(self,sequence_path,idOffset):

        global globalRadarDisplayImage,globalRadarSmoothTraj, imageCount

        if (self.annotations != None):
            for object in self.annotations:
                # reload the image for trajectory visualization 
                intitalImagePath = os.path.join(sequence_path, 'Navtech_Cartesian', str_format.format(1) + '.png')
                globalRadarDisplayImage = cv2.imread(intitalImagePath)
                globalRadarSmoothTraj = cv2.imread(intitalImagePath)

                centrePoses = []
                allBoundingBoxes = object['bboxes']
                allVisibleParam = object['visible']
                allDeleteParam = object['deleted']
                vehicleID = object['id'] + idOffset
                currentVehicleAngleList = []

                for idx,eachBbbox in enumerate(allBoundingBoxes):
                    if (eachBbbox):
                        if ((allDeleteParam[idx] == 0) and (allVisibleParam[idx] == 'visible')):
                            # Read the current Frame
                            radar_id = int(self.radar_ids[idx])
                            radar_cartesian_path = os.path.join(sequence_path, 'Navtech_Cartesian', str_format.format(radar_id) + '.png')
                            radar_cartesian = cv2.imread(radar_cartesian_path)

                            bbox = eachBbbox['position']
                            angle = eachBbbox['rotation']
                            currentVehicleAngleList.append(angle)

                            cx = int(bbox[0] + bbox[2]/2)
                            cy = int(bbox[1] + bbox[3]/2)
                            centrePoses.append([cx,cy])

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
                            radar_cartesian = cv2.putText(radar_cartesian, str(angle), (int(cx+20),int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA) 

                            cv2.imshow('image', radar_cartesian)
                            # cv2.imshow('smooth', finalImageTraj)
                            cv2.waitKey(1)



                # Smooth the jerky trajectories using particle Filter
                # Initialize Particle Filter for smooth trajectory

                if (len(centrePoses) < 3):
                    continue

                smothedTraj = []
                particleCount = 1000
                intialIndex = 2
                intitalCovariance = 4 #2
                pfObj = PFHelper.ParticleFilter(particleCount,[],'Classical',centrePoses[intialIndex][0],centrePoses[intialIndex][1],intitalCovariance)

                trajLength = len(centrePoses)
                for mdx in range(intialIndex,trajLength):
                    currentPoseX = centrePoses[mdx][0]
                    currentPoseY = centrePoses[mdx][1]
                    prevPoseX = centrePoses[mdx-1][0]
                    prevPoseY = centrePoses[mdx-1][1]
                    prevToPrevPoseX = centrePoses[mdx-2][0]
                    prevToPrevPoseY = centrePoses[mdx-2][1]
                    # currentVx = (prevPoseX-prevToPrevPoseX)/16
                    currentVx = (prevPoseX-prevToPrevPoseX)
                    currentVy = prevPoseY-prevToPrevPoseY
                    pfObj.update([currentPoseX,currentPoseY], currentVx, currentVy)
                    filteredX = int(pfObj.particleMean[0])
                    filteredY = int(pfObj.particleMean[1])
                    smothedTraj.append([int(filteredX),int(filteredY)])

                # If the length of center poses is zero no need to write the image and write the vehicle ID
                if(len(centrePoses) == 0):
                    print('Zero traj lenght vehicle ID ' + str(vehicleID))
                    continue


                # Draw the smoothed trajectory and write the vehicle ID for ref purpose
                centrePosesArray = np.array(centrePoses)
                # globalRadarDisplayImage = cv2.polylines(globalRadarDisplayImage,[centrePosesArray], False, (0,255,0), 2)
                globalRadarDisplayImage = cv2.putText(globalRadarDisplayImage, str(vehicleID), (800,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)

                # Draw the color coded trajectory 
                lineThickness = 2
                for poseIndex, eachCurrentPose in enumerate(centrePoses[2:]):
                    prevPose = centrePoses[poseIndex]

                    # Based on the angle change the color
                    # Extract last 3 angles to do avg
                    currentAngle = currentVehicleAngleList[poseIndex]
                    prevAngle = currentVehicleAngleList[poseIndex-1]
                    prevToPrevAngle = currentVehicleAngleList[poseIndex-2]

                    avgAngle = (currentAngle+prevAngle+prevToPrevAngle)/3

                    angleColor = (255,0,0)
                    # Check the current angle with each angle with margin
                    for eachStraightAngle in straightAngles:
                        if(avgAngle > eachStraightAngle-3 and avgAngle < eachStraightAngle+3):
                            angleColor = (0,255,0)
                            break
                    cv2.line(globalRadarDisplayImage, (prevPose[0], prevPose[1]), (eachCurrentPose[0], eachCurrentPose[1]), angleColor, lineThickness)


                # Calculate the tangent of the trajectory and display
                # currentTrajTangentInfo, fittedTraj = CalculateTangent(centrePosesArray)
                # fittedTrajArray = np.array(fittedTraj)

                # Check the fitted Trajectory
                # globalRadarDisplayImage = cv2.polylines(globalRadarDisplayImage,[fittedTrajArray], False, (0,0,255), 2)

                # for eachTanInfo in currentTrajTangentInfo:
                #     minXTan = int(eachTanInfo[0])
                #     maxXTan = int(eachTanInfo[1])
                #     lowYTan = int(eachTanInfo[2])
                #     highYTan = int(eachTanInfo[3])

                    # globalRadarDisplayImage = cv2.line(globalRadarDisplayImage, (minXTan,lowYTan), (maxXTan,highYTan), (0,0,255), 1) 

                    # cv2.imshow('traj', globalRadarDisplayImage)
                    # cv2.waitKey(1)


                # Smoothed traj is giving nan error. So ignore at the moment and insted of using smothedTraj use raw poses 
                # globalRadarDisplayImage = cv2.polylines(globalRadarDisplayImage,[np.array(smothedTraj)], False, (0,255,0), 2)

                finalImageTraj = np.hstack((globalRadarDisplayImage,globalRadarSmoothTraj))
                imageCount = imageCount+1
                imagePath = '/home/saptarshi/PythonCode/Junction/avgAngle/' + str(imageCount) + '.png'
                # cv2.imwrite(imagePath, globalRadarDisplayImage)
                # cv2.imshow('image', radar_cartesian)
                # # cv2.imshow('smooth', finalImageTraj)
                # cv2.waitKey(1)


def __main__():

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
        showedImage = sequence.play(sequence_path,idOffset)

        # sequence.playByVehicle(sequence_path,idOffset)
        # averageVelocityGlobalList.extend(returnedAvgVelList)
        idOffset = idOffset + sorted([d['id'] for d in sequence.annotations])[-1]
    
    cv2.imwrite('./countImage.png', showedImage)
    # print(averageVelocityGlobalList)
    print(len(averageVelocityGlobalList))

    # Write the average velocities in a file 
    with open('AverageVelocity.txt', 'w') as f:
        for item in averageVelocityGlobalList:
            f.write("%s\n" % item)

    # Seprate the turn and straight vehicles velocities 
    turnVelocityList = []
    straightVelocityList = []

    for eachVelocity in averageVelocityGlobalList:
        if(eachVelocity[0] == straightStr):
            straightVelocityList.append(eachVelocity[1])
        elif(eachVelocity[0] == turnStr):
            turnVelocityList.append(eachVelocity[1])
        else:
            print('Unknown maneuver string')

    
__main__()
