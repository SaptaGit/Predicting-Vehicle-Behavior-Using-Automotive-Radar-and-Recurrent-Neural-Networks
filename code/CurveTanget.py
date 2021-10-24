"""
===============
Rain simulation
===============

Simulates rain drops on a surface by animating the scale and opacity
of 50 scatter points.

Author: Nicolas P. Rougier
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import interpolate


# Create new Figure and an Axes which fills it.
fig = plt.figure(figsize=(20, 10))
ax = fig.add_axes([0, 0, 1, 1], frameon=False)
ax.set_xlim(0, 12), ax.set_xticks([])
ax.set_ylim(-2, 2), ax.set_yticks([])

# Generate the pose list using sine
x = np.arange(0,40,0.01)
y = np.sin(x)

# Interpolate the curve
curve = interpolate.splrep(x,y)

# Interpolated curve poses
interpolatedX = []
interpolatedY = []
tangentList = []


for kdx,eachX in  enumerate(x):
    x0 = eachX 
    y0 = y[kdx]
    dydx = interpolate.splev(x0,curve, der=1)
    interpolatedX.append(x0)
    interpolatedY.append(dydx)
    tngnt = lambda x: dydx*x + (y0-dydx*x0)
    tangentItem = tngnt(x)
    tangentList.append(tangentItem)
    


# plt.plot(x0,y0, "or")
# plt.plot(x,tngnt(x), label="tangent")


ax.plot(x,y)
# ax.plot(interpolatedX,interpolatedY)
# Construct the scatter which we will update during animation
# as the raindrops develop.
scat = ax.scatter(x, y,  s=70, c='g', marker='o')
linePlot = ax.plot(x,tangentList[0])



def update(frame_number):

    currentData = np.array([interpolatedX[frame_number],interpolatedY[frame_number]])
    scat.set_offsets(currentData)
    linePlot.set_data(x,tangentList[frame_number])
    return linePlot


# Construct the animation, using the update function as the animation
# director.
animation = FuncAnimation(fig, update, frames=len(x))
plt.show()




# from scipy import interpolate
# import matplotlib.pyplot as plt
# import numpy as np
# from time import sleep
# import matplotlib.pyplot as plt 
# import matplotlib.animation as animation 


# x = np.linspace(-15,15,1000)
# y = np.sin(x)
# tck = interpolate.splrep(x,y)


# fig = plt.figure() 
# # ax = plt.axes(xlim=(-20, 20), ylim=(-100, 100)) 
# ax = plt.axes() 
# line, = ax.plot([], [], lw=2) 
# scat = plt.scatter(x, y)

# # x0 = x[i]
# # y0 = interpolate.splev(x0,tck)


# # # initialization function 
# def init():
#     # creating an empty plot/frame 
#     line.set_data([], [])
#     return line, 

# # animation function 
# def animate(i): 
#     # dydx = interpolate.splev(x0,tck,der=1)

#     # tngnt = lambda x: dydx*x + (y0-dydx*x0)

#     # plt.plot(x0,y0, "or")
#     # plt.plot(x,tngnt(x), label="tangent")


# 	# appending new points to x, y axes points list 
# 	# xdata.append(x) 
# 	# ydata.append(y) 
# 	# line.set_data(x, tngnt(x)) 
#     # line.set_data(x0, y0) 
#     scat.set_data(x[i], y[i])
#     return scat,

    
# anim = animation.FuncAnimation(fig, animate, init_func=init, frames=500, interval=20, blit=True) 
# plt.show()

# # plt.plot(x,y)

# # for eachX in x:
# #     x0 = eachX
# #     y0 = interpolate.splev(x0,tck)
# #     dydx = interpolate.splev(x0,tck,der=1)

# #     tngnt = lambda x: dydx*x + (y0-dydx*x0)

# #     plt.plot(x0,y0, "or")
# #     plt.plot(x,tngnt(x), label="tangent")

# #     plt.legend()
# #     plt.show()

# #     sleep(1)