import numpy, matplotlib
import matplotlib.pyplot as plt
import sys

# xData = numpy.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0])
# yData = numpy.array([21.05, 21.21, 20.76, 20.34, 20.27, 20.78, 20.60, 20.55, 19.95, 19.23, 19.64, 19.92, 19.91, 19.56, 19.39, 19.31, 19.35, 18.97, 18.69, 19.00, 19.15, 19.08, 18.97, 19.26, 19.52, 19.56, 19.28, 19.47, 19.85, 19.77])

# xData = numpy.arange(0,10,0.1)
# yData = numpy.sin(xData)

newPoseArray = numpy.array([[279, 428],
[288, 428],
[298, 427],
[311, 427],
[320, 426],
[335, 428],
[345, 429],
[357, 431],
[367, 431],
[378, 432],
[389, 432],
[398, 433],
[411, 435],
[419, 436],
[434, 435],
[443, 438],
[452, 440],
[457, 442],
[470, 442],
[479, 442],
[490, 445],
[496, 446],
[502, 447],
[509, 447],
[515, 447],
[525, 449],
[528, 449],
[533, 449],
[533, 449],
[542, 451],
[544, 451],
[548, 449],
[553, 449],
[553, 449],
[553, 449],
[557, 450],
[559, 450],
[559, 450],
[560, 450],
[565, 451],
[568, 451],
[568, 451],
[568, 451],
[575, 453],
[580, 454],
[580, 454],
[585, 460],
[588, 463],
[591, 468],
[594, 471],
[596, 473],
[597, 480],
[598, 485],
[599, 491],
[599, 499],
[599, 503],
[599, 510],
[599, 516],
[600, 526],
[599, 531],
[599, 541],
[599, 548],
[599, 555],
[599, 565],
[603, 587],
[603, 587],
[601, 601],
[600, 613],
[599, 623],
[599, 634],
[599, 642],
[599, 654],
[599, 665],
[597, 672],
[597, 683],
[597, 693],
[597, 705],
[598, 719],
[598, 730],
[598, 739],
[599, 750],
[598, 760],
[597, 771],
[598, 782],
[598, 792],
[598, 804],
[596, 812],
[596, 820],
[597, 834],
[597, 842],
[599, 848],
[598, 858],
[597, 867],
[597, 878],
[598, 887],
[599, 901],
[599, 911],
[599, 922],
[598, 932],
[599, 942],
[599, 952],
[603, 965],
[603, 977],
[606, 987],
[610, 999],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[608, 1012],
[607, 1022],
[609, 1032],
[609, 1042],
[612, 1052],
[612, 1065]])

xData = newPoseArray[:,0]
yData = newPoseArray[:,1]

# polynomial curve fit the test data
fittedParameters = numpy.polyfit(xData, yData, 20)

# polynomial derivative from numpy
deriv = numpy.polyder(fittedParameters)

# create data for the fitted equation plot
xModel = numpy.linspace(min(xData), max(xData))
yModel = numpy.polyval(fittedParameters, xModel)


f = plt.figure(figsize=(20, 20))
axes = f.add_axes([0, 0, 1, 1])

for eachVal in xData:

    # axes.set_xlim([-10,60])
    # axes.set_ylim([-1.2,1.2])

    axes.plot(xModel, yModel)
    axes.plot(xData, yData)

    minX = eachVal - 10
    maxX = eachVal + 10

    y_value_at_point = numpy.polyval(fittedParameters, eachVal)
    slope_at_point = numpy.polyval(deriv, eachVal)

    ylow = (minX - eachVal) * slope_at_point + y_value_at_point
    yhigh = (maxX - eachVal) * slope_at_point + y_value_at_point

    # now the tangent as a line plot
    axes.plot([minX, maxX], [ylow, yhigh])

    plt.pause(0.05)

    plt.cla()

plt.show()