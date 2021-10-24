
import numpy as np
import sys
from random import randrange
import random

futureTemporal = 50
numberOfSelectedItems = 73034   # (can go down upto 3.03....)

# For 3 by 3 
# Caluclate the class based Accuracy
def classBasedAcc(inputConfMat):
    straightAcc  = (inputConfMat[0,0]+inputConfMat[1,1]+inputConfMat[1,2]+inputConfMat[2,1]+inputConfMat[2,2])/(sum(sum(inputConfMat)))
    leftTurnAcc  = (inputConfMat[1,1]+inputConfMat[0,0]+inputConfMat[0,2]+inputConfMat[2,0]+inputConfMat[2,2])/(sum(sum(inputConfMat)))
    rightTurnAcc = (inputConfMat[2,2]+inputConfMat[0,0]+inputConfMat[0,1]+inputConfMat[1,0]+inputConfMat[1,1])/(sum(sum(inputConfMat)))
    return round(straightAcc,3),round(leftTurnAcc,3),round(rightTurnAcc,3)

# Calculate the Precision for each class
def classBasedPrecision(inputConfMat):
    columnBasedSum = sum(inputConfMat)
    straightPrec = inputConfMat[0,0]/columnBasedSum[0]
    leftTurnPrec = inputConfMat[1,1]/columnBasedSum[1]
    rightTurnPrec = inputConfMat[2,2]/columnBasedSum[2]
    return round(straightPrec,3),round(leftTurnPrec,3),round(rightTurnPrec,3)

# For 2 by 2 
# # # Caluclate the class based Accuracy
# # def classBasedAcc(inputConfMat):
# #     straightAcc  = (inputConfMat[0,0]+inputConfMat[1,1])/(sum(sum(inputConfMat)))
# #     leftTurnAcc  = (inputConfMat[1,1]+inputConfMat[0,0])/(sum(sum(inputConfMat)))
# #     return round(straightAcc,3),round(leftTurnAcc,3)

# # # Calculate the Precision for each class
# # def classBasedPrecision(inputConfMat):
# #     columnBasedSum = sum(inputConfMat)
# #     straightPrec = inputConfMat[0,0]/columnBasedSum[0]
# #     leftTurnPrec = inputConfMat[1,1]/columnBasedSum[1]
# #     return round(straightPrec,3),round(leftTurnPrec,3)




if __name__ == '__main__':

    # # # sum = (3476082 + 24312 + 20900 + 33080 + 232172 + 2958 + 5791 + 729 + 70676)/50

    # Read the RMSE error file
    f = open("/home/saptarshi/PythonCode/Junction/models/SenServerModels/V18RelativeDecoderJuncDistEligibilityCheck91.txt", "r")
    # f = open("/home/saptarshi/PythonCode/Junction/models/SenServerModels/V22ModidiedDecoder96.txt", "r")
    # numberOfSamples = len(f.readlines())
    # print('Number of samples = ' + str(numberOfSamples))
    allSamples = f.readlines()

    # Collect all the error items and convert to float
    totalErrorItems = []
    for eachLine in allSamples :   # f.readlines():
        currentErrorStr = eachLine[1:-2]
        currentErrorItems = currentErrorStr.split(',')
        errorFloat = []
        for eachItem in currentErrorItems:
            errorFloat.append(float(eachItem))

        # Check the length of each list should be 
        eachItemLen = len(errorFloat)
        if(eachItemLen != futureTemporal):
            print('Expected and original prediction line mis-match!!!')
            print('Expected Length : ' + str(futureTemporal))
            print('Orginal Length  : ' + str(eachItemLen))
            sys.exit()

        # If the check satisfied add it to the final list
        totalErrorItems.append(errorFloat)

    print('Number of samples = ' + str(len(totalErrorItems)))

    # Sort the list based on the last error item and convert to array 
    sortedList = sorted(totalErrorItems,key=lambda x: x[0])
    errorArray = np.array(sortedList)

    # Calculate mean and variance
    errorMean = np.mean(errorArray[0:numberOfSelectedItems,:], axis=-2)   # axis -2 for row wise
    print('Error mean : ')
    print(errorMean)
    errorVariance = np.std(errorArray[0:numberOfSelectedItems,:], axis=-2)   # axis -2 for row wise
    print('Error Variance : ')
    print(errorVariance)

    # randIndex = random.sample(range(1, numberOfSelectedItems), 10000)

    # finalError = np.zeros((futureTemporal))

    # for eachIndex in randIndex:
    #     finalError = finalError + errorArray[int(eachIndex)]

    # print('New error!!!')
    # print(finalError/10000)








    # # # # NGSIM frame divded...
    # # # confMatrix = np.array([[46921,786,715],[1161,19243,459],[615,414,7013]])

    # # # straightAccuracy,leftTurnAccuracy,rightTurnAccuracy = classBasedAcc(confMatrix)
    # # # straightPrecision,leftTurnPrecision,rightTurnPrecision = classBasedPrecision(confMatrix)

    # # # print('Confusion Matrix .....')
    # # # print(confMatrix)
    # # # print('Class Based Accuracy .....')
    # # # print(straightAccuracy,leftTurnAccuracy,rightTurnAccuracy)
    # # # print('Class Based Precision .......')
    # # # print(straightPrecision,leftTurnPrecision,rightTurnPrecision)
    # # # print('All predicted Done!!!!!')

    # NGSIM
    # confMatrix = np.array([[2346050,39300,35750],[58050,962150,22950],[30750,20700,350650]])

    # Sighthill
    # confMatrix = np.array([[2800,12,18],[21,1440,6],[46,8,1320]])

    # # Alvie
    # confMatrix = np.array([[4755,6,0],[4,3335,0],[0,0,0]])

    # US101/I80 conf matrix
    # confMatrix = np.array([[11027052, 137872, 36850], [25630, 84442, 1212], [169383, 22562, 24347]])
    # confMatrix = np.array([[4907377, 154612, 41661],[177018, 116453, 3007],[39309, 5916, 25247]])   # 11026952
    confMatrix = np.array([[5027420, 160342, 42181],[895, 103105, 7],[95389, 13534, 27727]])


    # Calculate the fractional confutsion matrix
    confFracMatrix = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            columnSum = sum(confMatrix[:,j])
            confFracMatrix[i,j] = round(confMatrix[i,j]/columnSum,3)


    # for 3 by 3
    straightAccuracy,leftTurnAccuracy,rightTurnAccuracy = classBasedAcc(confMatrix)
    straightPrecision,leftTurnPrecision,rightTurnPrecision = classBasedPrecision(confMatrix)

    # # # for 2 by 2
    # # straightAccuracy,leftTurnAccuracy = classBasedAcc(confMatrix)
    # # straightPrecision,leftTurnPrecision = classBasedPrecision(confMatrix)


    # For 2 by 2
    # # print('Confusion Matrix .....')
    # # print(confMatrix)
    # # print('Class Based Accuracy .....')
    # # print(straightAccuracy,leftTurnAccuracy)
    # # print('Class Based Precision .......')
    # # print(straightPrecision,leftTurnPrecision)
    # # print('All predicted Done!!!!!')

    # For 3 by 3
    print('Confusion Matrix .....')
    print(confMatrix)
    print('Confusion fraction Matrix .....')
    print(confFracMatrix)
    print('Class Based Accuracy .....')
    print(straightAccuracy,leftTurnAccuracy,rightTurnAccuracy)
    print('Class Based Precision .......')
    print(straightPrecision,leftTurnPrecision,rightTurnPrecision)
    print('All predicted Done!!!!!')



# # # [[3476082   24312   20900]
# # #  [  33080  232172    2958]
# # #  [   5791     729   70676]]



# # # # # NGSIM 

# # # # # [2346050,39300,35750]
# # # # # [58050,962150,22950]   done
# # # # # [30750,20700,350650]

# # # Class Based Accuracy .....
# # # 0.958 0.964 0.972    done
# # # Class Based Precision .......
# # # 0.964 0.941 0.857


# # # navtech sighthill 

# # # Confusion Matrix!!!
# # # [[2800    14    20]
# # #  [   22 1440    8]   done
# # #  [   46    10 1320]]

# # # Class Based Accuracy .....
# # # 0.983 0.992 0.986
# # # Class Based Precision .......
# # # 0.977 0.986 0.982


# # # alvie

# # # Confusion Matrix!!!
# # # [[4755    6    0]
# # #  [   4 3335    0]   done
# # #  [   0    0    0]]

# # # Class Based Accuracy .....
# # # 0.999 0.999   done
# # # Class Based Precision .......
# # # 0.999 0.998

 

# # # sighthill
# # # Final Error is : 
# # # [0.72065772 0.74655208 0.83934793 0.95919884 1.05205762 1.21008557
# # #  1.3619371  1.54008445 1.708441   1.88501992 2.06700054 2.25169894
# # #  2.46426486 2.65070589 2.84105846 3.06205112 3.28857904 3.53240096
# # #  3.8037846  4.05145579]


# # # alvie
# # # Final Error is : 
# # # [0.45643026 0.60791206 0.73338413 0.84927809 0.94081189 1.09778269
# # #  1.26468539 1.45076165 1.60310786 1.76780287 1.97754917 2.166085
# # #  2.35061262 2.53913141 2.77982892 2.92261095 3.13736986 3.3041186
# # #  3.51148733 3.69188805]

# # # # I80/US101 manuever classification early 0.25 
# # # # This one is frame count, not sample
# divide by 50 for number of samples
# # # #############################################
# # # Final confusion matrix!!!
# # # [[11030398   138402    40266]
# # #  [  159230   103768     2119]
# # #  [   32437     2706    20024]]
# # # #############################################

# # # # #############################################
# # # # Final confusion matrix!!!
# # # # [[11027052   137872    36850]
# # # #  [   25630    84442     1212]
# # # #  [  169383    22562    24347]]
# # # # #############################################

# # # [[11026952   137699    36849]
# # #  [   14582    60796      173]
# # #  [  180531    46381    25387]]

# # [[4907377  154612   41661]
# #  [ 177018  116453    3007]
# #  [  39309    5916   25247]]





# number of samples for all NGSIM sighthill and lavie
#            Train    VAl   TEST
# NGSIM     307360  80000  77327
# Sighthill  2880     440    282       done
# Alvie     3540      550    405   



