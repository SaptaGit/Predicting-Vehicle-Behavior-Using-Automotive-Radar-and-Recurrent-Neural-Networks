import numpy as np
# import matplotlib.pyplot as plt
import math
import scipy.stats as stats
from scipy.ndimage.filters import gaussian_filter1d
# import matplotlib.lines as mlines

straightStr = 'Straight'
turnStr = 'Turn'

def Histplot():
    f = open("/home/saptarshi/PythonCode/Junction/AverageVelocity.txt", "r")
    lines = f.readlines()

    # Seprate the turn and straight vehicles velocities 
    turnVelocityList = []
    straightVelocityList = []

    for line in lines:
        info = line[1:-2].split(',')
        maneuverInfo = info[0][1:-1]
        velocityInfo = float(info[1][1:-2])
        if(maneuverInfo == straightStr):
            straightVelocityList.append(velocityInfo)
        elif(maneuverInfo == turnStr):
            turnVelocityList.append(velocityInfo)
        else:
            print('Unknown maneuver string')

    straightMean = np.mean(straightVelocityList)
    straightVar = np.var(straightVelocityList)
    straightSigma = math.sqrt(straightVar)

    turnMean = np.mean(turnVelocityList)
    turnVar = np.var(turnVelocityList)
    turnSigma = math.sqrt(turnVar)


    # straightDist = np.linspace(straightMean - 3*straightSigma, straightMean + 3*straightSigma, 100)
    # turnDist = np.linspace(turnMean - 3*turnSigma, turnMean + 3*turnSigma, 100)
    # plt.plot(straightDist, stats.norm.pdf(straightDist, straightMean, straightSigma), label='Straight', linewidth=2)
    # plt.plot(turnDist, stats.norm.pdf(turnDist, turnMean, turnSigma), label='Turn', linewidth=2)

    plt.hist(straightVelocityList,30, label='Straight')
    plt.hist(turnVelocityList,30, label='Turn')
    plt.xlabel('Velocity', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(prop={'size': 18})
    plt.show()

def ProfilePlot():
    f = open("/home/saptarshi/PythonCode/Junction/infos/profile1.txt", "r")
    writeFile = open("/home/saptarshi/PythonCode/Junction/infos/smoothText.txt", "w")
    lines = f.readlines()

    # Seprate the turn and straight vehicles velocity Profiles 
    turnVelocityList = []
    straightVelocityList = []

    # indexList = [42,44,57,50, 62] # 
    # indexList = [42] # 57

    highesFrameInfo = lines.pop(-1)

    highestFrame = int(highesFrameInfo.split(':')[-1])

    for idx,line in enumerate(lines):
        # if(idx not in indexList):
        #     continue
        maneuverInfo = line.split(',')[-2]
        velocityInfo = line.split(',')[0:-2]
        velocityInfo = [float(i) for i in velocityInfo]
        velocitySmoothed = gaussian_filter1d(velocityInfo, sigma=2)

        for item in velocitySmoothed:
            writeFile.write("%s," % str(item))
        writeFile.write("%s\n" % maneuverInfo)

        currentLength = len(velocitySmoothed)
        lengthDiff = highestFrame-currentLength
        xAxisVal = range(lengthDiff,highestFrame,1)

        # # if(maneuverInfo == straightStr):
        # #     plt.plot(xAxisVal,velocitySmoothed, 'g--', linewidth=3)
        # # elif(maneuverInfo == turnStr):
        # #     plt.plot(xAxisVal,velocitySmoothed, 'r-.', linewidth=3)
        # # else:
        # #     print('Unknown manuever info')
        # #     sys.exit()

    writeFile.close()

    # plt.hist(straightVelocityList,30, label='Straight')
    # plt.hist(turnVelocityList,30, label='Turn')
    patchList = []
    greenLine = mlines.Line2D([], [], color='green', linestyle='--',linewidth=3, label='Straight')    
    patchList.append(greenLine)
    redLine = mlines.Line2D([], [], color='red', linestyle='-.', linewidth=3, label='Turn')    
    patchList.append(redLine)


    plt.xlabel('Frame', fontsize=20)
    plt.ylabel('Velocity', fontsize=20)
    plt.xticks(np.arange(0,highestFrame,4), np.arange(-highestFrame,0,4), fontsize=18)
    plt.yticks(fontsize=16)
    plt.legend(handles=patchList, prop={'size': 20}, loc='upper left')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.show()


def __main__():

    # Histplot()
    
    ProfilePlot()

__main__()