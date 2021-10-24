# Extract the labbeled bounding boxes and create OGMs from training.
import cv2
import os
import numpy as np
# import matplotlib.pyplot as plt
import json
# import pandas as pd
import math
import scipy.interpolate as interp
# import matplotlib.pyplot as plt
# import pptk

sequence_folder = '/home/saptarshi/Dropbox (Heriot-Watt University Team)/RES_EPS_PathCad/datasets/pathcad_van_dataset/junction_1'
# sequence_folder = '/home/saptarshi/PythonCode/Junction/NavtechSample'
globalRadarDisplayImage = np.zeros((1152,1152,3))
globalRadarSmoothTraj = np.zeros((1152,1152,3))
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.namedWindow('smooth',cv2.WINDOW_NORMAL)
turn = 0
straight = 0
processedObjectIds = []
str_format = '{:06d}'
straightStr = 'Straight'
turnStr = 'Turn'
highestFrameCount = 0

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
        annotateAngle = poseList[mdx][2]
        # Add the first two poses
        if mdx<intialIndex:
            smothedTraj.append([int(currentPoseX),int(currentPoseY),annotateAngle])
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
        smothedTraj.append([int(filteredX),int(filteredY),annotateAngle])

    # Return the smoothed trajectory
    return smothedTraj

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

        global globalRadarDisplayImage,globalRadarSmoothTraj,highestFrameCount
        junctionlocationY = 325
        averageVelocityLocalList = []
        velocityProfileList = []

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
                instanteniousVelocityList = []
                averageVelocityCalculated = False

                # Travers the list from begining to pick the first location and decide direction
                firstLocation = 0
                for ldx in (allBoundingBoxes):
                    if(ldx):
                        firstLocation = ldx['position'][1]
                        break
                if firstLocation < 700:
                    continue

                # Travers the list from last to pick the last location and decide turn ot straight
                lastLocation = 0
                maneuverString = ''
                for kdx in reversed(allBoundingBoxes):
                    if(kdx):
                        lastLocation = kdx['position'][0]
                        break
                
                if (lastLocation>580):
                    maneuverString = turnStr
                else:
                    maneuverString = straightStr

                for idx,eachBbbox in enumerate(allBoundingBoxes):
                    if (eachBbbox):
                        if ((allDeleteParam[idx] == 0) and (allVisibleParam[idx] == 'visible')):
                            # Read the current Frame
                            radar_id = int(self.radar_ids[idx])
                            radar_cartesian_path = os.path.join(sequence_path, 'Navtech_Cartesian', str_format.format(radar_id) + '.png')
                            radar_cartesian = cv2.imread(radar_cartesian_path)

                            bbox = eachBbbox['position']
                            angle = eachBbbox['rotation']
                            # ignore vehicle in other directions
                            # if ((angle > 100) and (angle < 200)):
                            #     continue
                            cx = int(bbox[0] + bbox[2]/2)
                            cy = int(bbox[1] + bbox[3]/2)
                            centrePoses.append((cx,cy))
                            radar_cartesian = self.draw_boundingbox_rot(radar_cartesian, bbox, angle, (0,255,0))

                            # Check if the previous box exists if yes calculate instantenious velocity
                            if(allBoundingBoxes[idx-1]):
                                prevBox = allBoundingBoxes[idx-1]['position']
                                prevCx = int(prevBox[0] + prevBox[2]/2)
                                prevCy = int(prevBox[1] + prevBox[3]/2)
                                globalRadarDisplayImage = cv2.line(globalRadarDisplayImage, (cx,cy), (prevCx,prevCy), (0,0,255), 2)
                                instanteniousVelocity = (prevCy-cy)
                                instanteniousVelocityList.append(instanteniousVelocity)
                                if ((cy < junctionlocationY) and (not averageVelocityCalculated)):
                                    # For just the last 15 frames from junction
                                    #releventVelocityList = instanteniousVelocityList[-15:]

                                    # For just the last all frames from junction
                                    releventVelocityList = instanteniousVelocityList[:]

                                    frameLength = len(releventVelocityList)
                                    if(frameLength > highestFrameCount):
                                        highestFrameCount = frameLength

                                    averageVelociy = sum(releventVelocityList)/len(releventVelocityList)
                                    averageVelocityLocalList.append([maneuverString,averageVelociy])
                                    releventVelocityList.append(maneuverString)
                                    velocityProfileList.append(releventVelocityList)
                                    averageVelocityCalculated = True

                                #radar_cartesian = cv2.putText(radar_cartesian, str(instanteniousVeloity), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA) 


                            finalImageTraj = np.hstack((globalRadarDisplayImage,globalRadarSmoothTraj))

                            cv2.imshow('image', radar_cartesian)
                            cv2.imshow('smooth', finalImageTraj)
                            cv2.waitKey(1)

                # Smooth the jerky trajectories
                # # smothedTraj = interpolate_polyline(np.array(centrePoses),len(centrePoses))
                # # for jdx in range(1,len(smothedTraj)):
                # #     currentX = int(smothedTraj[jdx][0])
                # #     currentY = int(smothedTraj[jdx][1])
                # #     prevX = int(smothedTraj[jdx-1][0])
                # #     prevY = int(smothedTraj[jdx-1][1])
                # #     globalRadarSmoothTraj = cv2.line(globalRadarSmoothTraj, (currentX,currentY), (prevX,prevY), (0,0,255), 2)
                # #     finalImageTraj = np.hstack((globalRadarDisplayImage,globalRadarSmoothTraj))
                # #     cv2.imshow('image', radar_cartesian)
                # #     cv2.imshow('smooth', finalImageTraj)
                # #     cv2.waitKey(100)

        return averageVelocityLocalList, velocityProfileList


def __main__():

    folderList = os.listdir(sequence_folder)
    folderList.sort(key=lambda x: int(x.split('_')[-1]))
    idOffset = 0
    globalImageLoaded = False
    global globalRadarDisplayImage,globalRadarSmoothTraj
    averageVelocityGlobalList = []
    velocityProfileGlobal = []

    for eachSeq in folderList:
        print('Processing folder : ' + eachSeq)
        # Ignore junction 3 as its giving nan for the filter....
        if(eachSeq == 'junction_1_3'):
            continue
        sequence_path = os.path.join(sequence_folder, eachSeq)
        annotation_path = os.path.join(sequence_path, 'annotations', 'annotations.json')

        # #Load the globalImage for trajectory visualization
        # if not globalImageLoaded:
        #     intitalImagePath = os.path.join(sequence_path, 'Navtech_Cartesian', str_format.format(1) + '.png')
        #     globalRadarDisplayImage = cv2.imread(intitalImagePath)
        #     globalRadarSmoothTraj = cv2.imread(intitalImagePath)
        #     globalImageLoaded = True

        sequence = Sequence(sequence_path)
        sequence.load_sequence(sequence_path)
        sequence.load_annotations(annotation_path)
        returnedAvgVelList, returnedVelocityProfile = sequence.playByVehicle(sequence_path,idOffset)
        averageVelocityGlobalList.extend(returnedAvgVelList)
        velocityProfileGlobal.extend(returnedVelocityProfile)
        idOffset = idOffset + sorted([d['id'] for d in sequence.annotations])[-1]
    
    #cv2.imwrite('./countImage.png', showedImage)
    # print(averageVelocityGlobalList)
    print(len(averageVelocityGlobalList))

    # Write the average velocities in a file 
    with open('AverageVelocity.txt', 'w') as f:
        for item in averageVelocityGlobalList:
            f.write("%s\n" % item)

    with open('profile.txt','w') as f:
        for sublist in velocityProfileGlobal:
            for item in sublist:
                f.write(str(item) + ',')
            f.write('\n')
        f.write('Highest Frame Length:' + str(highestFrameCount))

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
