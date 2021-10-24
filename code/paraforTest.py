# from multiprocessing import Process, Manager
# import time


# def yourfunction(j):
#     for kdx in range(0,1000,1):
#         time.sleep(2)
#     print('End process ' + str(j))


# if __name__ == '__main__':
#     processes = []
#     d = {}

#     for j in range(0, 10000):
#         d[j] = Manager().Value('j',0)
#         p = Process(target=yourfunction, args=(d[j],))
#         p.start()
#         processes.append(p)
    
#     for q in processes:
#         q.join()



# import multiprocessing as mp
# import time

# def yourfunction(j):
#     a = 0
#     b = 0
#     for kdx in range(0,1000,1):
#         time.sleep(2)
#     print('End process ' + str(j))
#     return a,b


# if __name__ == '__main__':
#     processes = []
#     d = {}

#     pool = mp.Pool(5)
#     a,b = pool.map(yourfunction, range(0,1000,1))
#     print(a)

#     print(b)

import numpy as np

a = [1,2,3,4,5,6,7,8,9,10]
for idx,eacha in enumerate(a):
    if (eacha == 5):
        a.pop(idx)



check = open('/home/saptarshi/PythonCode/Junction/Lankershim/validation.txt', "r")

loadedData = check.readlines()
validationVehicleList = []

for eachValVehicle in loadedData:
    validationVehicleList.append(float(eachValVehicle.rstrip()))


# Enumarate 30 shift jump to get loadeddata [0:30] = sample
totalLines = len(loadedData)
for idx in range(0,totalLines,50):
    loadedSample = loadedData[idx:idx+50]
    sampleList = []
    for eachLoadedSample in loadedSample:
        currentSample = eachLoadedSample[1:-2].split(',')   
        currentSampleFloat = [float(i) for i in currentSample]
        sampleList.append(currentSampleFloat) 
    finalXTrain.append(sampleList)
    





loadedDataArray = np.array(finalXTrain)

print('Read file shape : ')
print(loadedDataArray.shape)
