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
import PFHelper
# import pptk

sequence_folder = '/home/saptarshi/Dropbox (Heriot-Watt University Team)/RES_EPS_PathCad/datasets/pathcad_van_dataset/junction_1'
# sequence_folder = '/home/saptarshi/Dropbox (Heriot-Watt University Team)/RES_EPS_PathCad/datasets/pathcad_van_dataset/SighthillJunc'
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
imageCount = 0

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

            finalImage = np.hstack((radar_cartesian,globalRadarDisplayImage))

            cv2.imshow('image', finalImage)
            cv2.waitKey(10)

        return finalImage

    def playByVehicle(self,sequence_path,idOffset):

        global globalRadarDisplayImage,globalRadarSmoothTraj, imageCount
        junctionlocationY = 325
        averageVelocityLocalList = []

        if (self.annotations != None):
            for object in self.annotations:
                # reload the image for trajectory visualization 
                intitalImagePath = os.path.join(sequence_path, 'Navtech_Cartesian', str_format.format(1) + '.png')
                # globalRadarDisplayImage = cv2.imread(intitalImagePath)
                # globalRadarSmoothTraj = cv2.imread(intitalImagePath)

                centrePoses = []
                allBoundingBoxes = object['bboxes']
                allVisibleParam = object['visible']
                allDeleteParam = object['deleted']
                instanteniousVelocityList = []
                averageVelocityCalculated = False

                # Travers the list from begining to pick the first location and decide direction
                firstLocationX = 0
                firstLocationY = 0
                for ldx in (allBoundingBoxes):
                    if(ldx):
                        firstLocationX = ldx['position'][0]
                        firstLocationY = ldx['position'][1]
                        break
                if firstLocationY < 700:
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
                            centrePoses.append([cx,cy])
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
                                    releventVelocityList = instanteniousVelocityList[-5:]
                                    averageVelociy = sum(releventVelocityList)/len(releventVelocityList)
                                    averageVelocityLocalList.append([maneuverString,averageVelociy])
                                    averageVelocityCalculated = True

                                #radar_cartesian = cv2.putText(radar_cartesian, str(instanteniousVeloity), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA) 


                            finalImageTraj = np.hstack((globalRadarDisplayImage,globalRadarSmoothTraj))

                            cv2.imshow('image', radar_cartesian)
                            cv2.imshow('smooth', finalImageTraj)
                            cv2.waitKey(1)

                # Smooth the jerky trajectories using particle Filter
                # Initialize Particle Filter for smooth trajectory

                # if (imageCount == 21):
                #     imageCount = imageCount + 1
                #     imagePath = '/home/saptarshi/PythonCode/Junction/smooth/' + str(imageCount) + '.png'
                #     cv2.imwrite(imagePath, globalRadarDisplayImage)
                #     continue

                if (len(centrePoses) < 3):
                    continue

                smothedTraj = []
                particleCount = 1000
                intialIndex = 2
                intitalCovariance = 2
                pfObj = PFHelper.ParticleFilter(particleCount,[],'Classical',centrePoses[intialIndex][0],centrePoses[intialIndex][1],intitalCovariance)

                trajLength = len(centrePoses)
                for mdx in range(intialIndex,trajLength):
                    currentPoseX = centrePoses[mdx][0]
                    currentPoseY = centrePoses[mdx][1]
                    prevPoseX = centrePoses[mdx-1][0]
                    prevPoseY = centrePoses[mdx-1][1]
                    prevToPrevPoseX = centrePoses[mdx-2][0]
                    prevToPrevPoseY = centrePoses[mdx-2][1]
                    currentVx = (prevPoseX-prevToPrevPoseX)/16
                    currentVy = prevPoseY-prevToPrevPoseY
                    pfObj.update([currentPoseX,currentPoseY], currentVx, currentVy)
                    filteredX = int(pfObj.particleMean[0])
                    filteredY = int(pfObj.particleMean[1])
                    smothedTraj.append([int(filteredX),int(filteredY)])

                # #PolyFit 
                # poseArray = np.array(centrePoses)
                # xPoints = poseArray[:,1]
                # yPoints = poseArray[:,0]

                # # calculate polynomial
                # z = np.polyfit(xPoints, yPoints, 3)
                # f = np.poly1d(z)

                # x_new = np.linspace(xPoints[0], xPoints[-1], 60)
                # y_new = f(x_new)

                # smothedTraj = []
                # for odx in range(0,len(x_new)):
                #     smothedTraj.append([int(y_new[odx]+5),int(x_new[odx])])

                # Draw the smoothed trajectory
                globalRadarDisplayImage = cv2.polylines(globalRadarDisplayImage,[np.array(smothedTraj)], False, (0,255,0), 2)
                finalImageTraj = np.hstack((globalRadarDisplayImage,globalRadarSmoothTraj))
                imageCount = imageCount+1
                imagePath = '/home/saptarshi/PythonCode/Junction/smooth/' + str(imageCount) + '.png'
                # cv2.imwrite(imagePath, globalRadarDisplayImage)
                cv2.imshow('image', radar_cartesian)
                cv2.imshow('smooth', finalImageTraj)
                cv2.waitKey(1)

        return averageVelocityLocalList


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
        # showedImage = sequence.play(sequence_path,idOffset)

        returnedAvgVelList = sequence.playByVehicle(sequence_path,idOffset)
        # averageVelocityGlobalList.extend(returnedAvgVelList)
        idOffset = idOffset + sorted([d['id'] for d in sequence.annotations])[-1]
    
    # cv2.imwrite('./countImage.png', showedImage)
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
