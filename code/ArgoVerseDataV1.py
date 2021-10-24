from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.visualization.visualize_sequences import viz_sequence
import numpy as np
import cv2
import sys
from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt

cv2.namedWindow('image',cv2.WINDOW_NORMAL)

# # #set root_dir to the correct path to your dataset folder
# rootTrainDir = '/home/saptarshi/Downloads/train/data/'
# rootValDir = '/home/saptarshi/Downloads/val/data/'
# rootTestDir = '/home/saptarshi/Downloads/test_obs/data/'
# imageFolder = '/home/saptarshi/PythonCode/Junction/ArgData/'

# imageWidth = 1000
# imageHeight = 1000
# imageCount = 0

# avm = ArgoverseMap()
# aflTrain = ArgoverseForecastingLoader(rootTrainDir)
# aflVal = ArgoverseForecastingLoader(rootValDir)
# aflTest = ArgoverseForecastingLoader(rootTestDir)

# trainSeqCount = len(aflTrain)
# print('Number of Train samples ' + str(trainSeqCount))

# valSeqCount = len(aflVal)
# print('Number of Validation samples ' + str(valSeqCount))

# testSeqCount = len(aflTest)
# print('Number of Test samples ' + str(testSeqCount))


# # Scatter plot of each minX and minY from each csv between training and validation to check the range. (Training)
# trainMinXList = []
# trainMaxXList = []
# trainMinYList = []
# trainMaxYList = []

# for jdx in range(0,trainSeqCount):
#     print('Processing Train sample ' + str(jdx) + ' of total sample ' + str(trainSeqCount))
#     agenTraj = aflTrain[jdx].agent_traj
#     minX = min(agenTraj[:,0])
#     maxX = max(agenTraj[:,0])
#     minY = min(agenTraj[:,1])
#     maxY = max(agenTraj[:,1])
#     trainMinXList.append(minX)
#     trainMaxXList.append(maxX)
#     trainMinYList.append(minY)
#     trainMaxYList.append(maxY)

# trainMinXList.extend(trainMaxXList)
# trainMinYList.extend(trainMaxYList)

# # Scatter plot of each minX and minY from each csv between training and validation to check the range. Validation
# valMinXList = []
# valMaxXList = []
# valMinYList = []
# valMaxYList = []

# for kdx in range(0,valSeqCount):
#     print('Processing Validation sample ' + str(kdx) + ' of total sample ' + str(valSeqCount))
#     agenTraj = aflVal[kdx].agent_traj
#     minX = min(agenTraj[:,0])
#     maxX = max(agenTraj[:,0])
#     minY = min(agenTraj[:,1])+1500
#     maxY = max(agenTraj[:,1])+1500
#     valMinXList.append(minX)
#     valMaxXList.append(maxX)
#     valMinYList.append(minY)
#     valMaxYList.append(maxY)

# valMinXList.extend(valMaxXList)
# valMinYList.extend(valMaxYList)


# # Scatter plot of each minX and minY from each csv between training and validation to check the range. Test
# testMinXList = []
# testMaxXList = []
# testMinYList = []
# testMaxYList = []

# for ldx in range(0,testSeqCount):
#     print('Processing Test sample ' + str(ldx) + ' of total sample ' + str(testSeqCount))
#     agenTraj = aflTest[ldx].agent_traj
#     minX = min(agenTraj[:,0])
#     maxX = max(agenTraj[:,0])
#     minY = min(agenTraj[:,1])
#     maxY = max(agenTraj[:,1])
#     testMinXList.append(minX)
#     testMaxXList.append(maxX)
#     testMinYList.append(minY)
#     testMaxYList.append(maxY)

# testMinXList.extend(testMaxXList)
# testMinYList.extend(testMaxYList)

# plt.scatter(np.array(valMinXList), np.array(valMinYList), label='Validation Region')
# plt.scatter(np.array(trainMinXList), np.array(trainMinYList), label='Training Region')
# plt.scatter(np.array(testMinXList), np.array(testMinYList), label='Test Region')
# plt.legend()
# plt.show()


# sys.exit()



# Plot each CSV file as trajectory over a defined image frame. 


afl = ArgoverseForecastingLoader('/home/saptarshi/Downloads/forecasting_sample/data/')
avm = ArgoverseMap()

imageWidth = 1000
imageHeight = 1000
imageCount = 0

seqCount = len(afl)

for jdx in range(0,seqCount):

    dataDf = afl[jdx].seq_df

    uniqueTimeStamps = sorted(np.unique(dataDf['TIMESTAMP'].values))

    minXVal = dataDf['X'].min()
    maxXVal = dataDf['X'].max()
    minYVal = dataDf['Y'].min()
    maxYVal = dataDf['Y'].max()
    cityName = dataDf['CITY_NAME'].values[0]

    localLanePolys = avm.find_local_lane_polygons([minXVal,maxXVal,minYVal,maxYVal], cityName)
    allLanePoly = []
    for eachLanePoly in localLanePolys:
        currentLnePoly = []
        for eachPoint in eachLanePoly:
            laneX = eachPoint[0]
            laneY = eachPoint[1]
            normLaneX = int(((laneX-minXVal)/(maxXVal-minXVal))*1000)
            normLaneY = int(((laneY-minYVal)/(maxYVal-minYVal))*1000)
            currentLnePoly.append([normLaneX,normLaneY])
        allLanePoly.append(np.array(currentLnePoly))

    # dispImage = np.zeros((imageHeight,imageWidth,3))
    # dispImage.fill(255)
    # dispImage = cv2.polylines(dispImage, allLanePoly,  False,  (0, 0, 0), 1)

    for eachUniqueTime in uniqueTimeStamps:
        selectedRows = dataDf.loc[dataDf['TIMESTAMP'] == eachUniqueTime]

        dispImage = np.zeros((imageHeight,imageWidth,3))
        dispImage.fill(255)
        dispImage = cv2.polylines(dispImage, allLanePoly,  False,  (0, 0, 0), 1)

        for idx,row in selectedRows.iterrows():
            color = (0,0,0)
            objectType = row['OBJECT_TYPE']
            xVal = row['X']
            yVal = row['Y']
            cityName = dataDf['CITY_NAME'].values[0]
            # laneDirection = avm.get_lane_direction(np.array([xVal,yVal]),cityName)
            laneSemanticLabel = avm.get_nearest_centerline(np.array([xVal,yVal]),cityName)
            turnDirection = laneSemanticLabel[0].turn_direction

            if(objectType == 'AGENT'):
                color = (0,0,255)
            elif(objectType == 'OTHERS'):
                color = (0,255,0)
            elif(objectType == 'AV'):
                color = (255,0,0)
            else:
                print('Unknow type')
                sys.exit()
            
            # Draw the cars
            normXVal = int(((xVal-minXVal)/(maxXVal-minXVal))*1000)
            normYVal = int(((yVal-minYVal)/(maxYVal-minYVal))*1000)
            dispImage = cv2.circle(dispImage, (normXVal,normYVal), 6, color, -1)

            # Draw the arrows
            # dx = laneDirection[0][0]*100
            # dy = laneDirection[0][1]*100
            # pt1 = (normXVal,normYVal)
            # pt2 = (int(normXVal+dx),int(normYVal+dy))
            # dispImage = cv2.arrowedLine(dispImage, pt1, pt2, (255,0,0), 5)

            # Put the lane direction
            dispImage = cv2.putText(dispImage, turnDirection, (normXVal,normYVal), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2, cv2.LINE_AA)


            # Put the legends and infos
            dispImage = cv2.circle(dispImage, (640,50), 6, (255,0,0), -1)
            dispImage = cv2.circle(dispImage, (640,100), 6, (0,0,255), -1)
            dispImage = cv2.circle(dispImage, (640,150), 6, (0,255,0), -1)
            dispImage = cv2.putText(dispImage, 'Ego Vehicle', (660,55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
            dispImage = cv2.putText(dispImage, 'Target Vehicle', (660,105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
            dispImage = cv2.putText(dispImage, 'Other Vehicles/Pedestrians', (660,155), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)




        cv2.imshow('image', dispImage)
        cv2.waitKey(1000)

        imageCount = imageCount + 1
        # imagePath = imageFolder + str(imageCount) + '.png'
        # cv2.imwrite(imagePath, dispImage)
