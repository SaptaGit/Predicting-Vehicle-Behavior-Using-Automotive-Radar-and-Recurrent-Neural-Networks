import math
from math import log, exp, tan, atan, pi, ceil
from PIL import Image
import urllib
import urllib.request
from io import StringIO
import io
import cv2
import numpy as np
import imutils


EARTH_RADIUS = 6378137
EQUATOR_CIRCUMFERENCE = 2 * math.pi * EARTH_RADIUS
INITIAL_RESOLUTION = EQUATOR_CIRCUMFERENCE / 256.0
ORIGIN_SHIFT = EQUATOR_CIRCUMFERENCE / 2.0

def latlontopixels(lat, lon, zoom):
    mx = (lon * ORIGIN_SHIFT) / 180.0
    my = log(tan((90 + lat) * pi/360.0))/(pi/180.0)
    my = (my * ORIGIN_SHIFT) /180.0
    res = INITIAL_RESOLUTION / (2**zoom)
    px = (mx + ORIGIN_SHIFT) / res
    py = (my + ORIGIN_SHIFT) / res
    return px, py

def pixelstolatlon(px, py, zoom):
    res = INITIAL_RESOLUTION / (2**zoom)
    mx = px * res - ORIGIN_SHIFT
    my = py * res - ORIGIN_SHIFT
    lat = (my / ORIGIN_SHIFT) * 180.0
    lat = 180 / pi * (2*atan(exp(lat*pi/180.0)) - pi/2.0)
    lon = (mx / ORIGIN_SHIFT) * 180.0
    return lat, lon


# Position, decimal degrees 55.929897749495112,-3.272431262552133   

# # # Sighthill junction localtion 55.929910999748550,-3.272424913293455
# # lat = 55.929910999748550
# # lon = -3.272424913293455

# # Alvie T junction  location  57.082478622181213,-4.041574860382791
# # lat = 57.082478622181213
# # lon = -4.041574860382791

# Alvie round-about junction location 57.191864153802513,-3.829045207652034
# lat = 57.191864153802513
# lon = -3.829045207652034

# # # Single Lane Currie junction localtion 55.929910999748550,-3.272424913293455
# # lat = 55.888990102183953
# # lon = -3.335993885880295

# # # # Double lane Highway location 55.899465773681563,-3.229875722229063
# # # lat = 55.899465773681563
# # # lon = -3.229875722229063

# Triple lane highway location 55.891702601553675,-3.180328821893259
lat = 55.891702601553675
lon = -3.180328821893259

# # # # offsets in meters for sighthill
# # # dn = 85
# # # de = 75

# offsets in meters for alvie T junction
# # dn = 85
# # de = 85

# offsets in meters for alvie round about junction 
dn = 85
de = 85

# Coordinate offsets in radians
dLat = dn/EARTH_RADIUS
dLon = de/(EARTH_RADIUS*math.cos(math.pi*lat/180))

# OffsetPosition, decimal degrees
lat0 = lat + dLat * 180/math.pi
lon0 = lon + dLon * 180/math.pi

# # # # offsets in meters
# # # dn = -85
# # # de = -75

# offsets in meters
dn = -100
de = -100

# Coordinate offsets in radians
dLat = dn/EARTH_RADIUS
dLon = de/(EARTH_RADIUS*math.cos(math.pi*lat/180))

# OffsetPosition, decimal degrees
lat1 = lat + dLat * 180/math.pi
lon1 = lon + dLon * 180/math.pi

# a neighbourhood in Lajeado, Brazil:

upperleft =  str(lat0) + ',' + str(lon1) 
lowerright = str(lat1) + ',' + str(lon0)

# upperleft =  '-29.44,-52.0'  
# lowerright = '-29.45,-51.98'

# upperleft = '37.8466,-122.2987'
# lowerright =  '37.8385,-122.2962'  

zoom = 20  # 19 works best   # be careful not to get too many images!
#metersPerPx = 156543.03392 * Math.cos(latLng.lat() * Math.PI / 180) / Math.pow(2, zoom) (from google [https://gis.stackexchange.com/questions/7430/what-ratio-scales-do-google-maps-zoom-levels-correspond-to])

############################################

ullat, ullon = map(float, upperleft.split(','))
lrlat, lrlon = map(float, lowerright.split(','))

# Set some important parameters
scale = 1
maxsize = 640

# convert all these coordinates to pixels
ulx, uly = latlontopixels(ullat, ullon, zoom)
lrx, lry = latlontopixels(lrlat, lrlon, zoom)

# calculate total pixel dimensions of final image
dx, dy = lrx - ulx, uly - lry

# calculate rows and columns
cols, rows = int(ceil(dx/maxsize)), int(ceil(dy/maxsize))

# calculate pixel dimensions of each small image
bottom = 120
largura = int(ceil(dx/cols))
altura = int(ceil(dy/rows))
alturaplus = altura + bottom


final = Image.new("RGB", (int(dx), int(dy)))
for x in range(cols):
    for y in range(rows):
        dxn = largura * (0.5 + x)
        dyn = altura * (0.5 + y)
        latn, lonn = pixelstolatlon(ulx + dxn, uly - dyn - bottom/2, zoom)
        position = ','.join((str(latn), str(lonn)))
        #print x, y, position
        urlparams = urllib.parse.urlencode({'center': position,
                                      'zoom': str(zoom),
                                      'size': '%dx%d' % (largura, alturaplus),
                                      'maptype': 'satellite',
                                      'sensor': 'false',
                                      'scale': scale,
                                      'key':'AIzaSyBuYwhrUegFSvy2UaLDXd52CxiuXlMsLm4'})
        url = 'http://maps.google.com/maps/api/staticmap?' + urlparams
        f=urllib.request.urlopen(url)
        #buffer = StringIO(f.read())
        im=Image.open(io.BytesIO(f.read()))
        final.paste(im, (int(x*largura), int(y*altura)))

# final.show()
cv2.namedWindow('test', cv2.WINDOW_NORMAL)
opencvImage = np.array(final)
# cv2.imshow('test', opencvImage)
# cv2.waitKey(0)
# Rotation 110 deg for Sighthill junction
# Rotation 300 deg for Alvie junction ...
# Rotation 220 deg for Alvie Round-about ...
rotated = imutils.rotate_bound(opencvImage,260) 
# cv2.imwrite('/home/saptarshi/PythonCode/Junction/SightHillMap.png', rotated)
# cv2.namedWindow('test', cv2.WINDOW_NORMAL)
cv2.imshow('test',rotated)
cv2.waitKey(0)

# Add the lane lines
print('Add lane lines!!!')
dim = (1152, 1152)
resizedMap = cv2.resize(rotated, dim, interpolation = cv2.INTER_AREA)

# Main road
# resizedMap = cv2.line(resizedMap, (140,418), (456,427), (255,0,0), 2) 
# resizedMap = cv2.line(resizedMap, (130,458), (583,470), (255,0,0), 2) 

# # resizedMap = cv2.line(resizedMap, (456,427), (1106,450), (255,0,0), 2) 
# # resizedMap = cv2.line(resizedMap, (583,470), (1097,484), (255,0,0), 2) 

# #Side road on top
# resizedMap = cv2.line(resizedMap, (448,427), (456,89), (255,0,0), 2) 
# resizedMap = cv2.line(resizedMap, (465,427), (470,89), (255,0,0), 2) 

# # Side road below
# resizedMap = cv2.line(resizedMap, (600,470), (605,1015), (255,0,0), 2) 
# resizedMap = cv2.line(resizedMap, (564,470), (592,1015), (255,0,0), 2) 

# # Parallel road below left
# resizedMap = cv2.line(resizedMap, (567,533), (112,513), (255,0,0), 2)
# resizedMap = cv2.line(resizedMap, (570,553), (100,533), (255,0,0), 2) 

# # Parallel road below Right
# resizedMap = cv2.line(resizedMap, (601,537), (1077,553), (255,0,0), 2)
# resizedMap = cv2.line(resizedMap, (601,561), (1069,573), (255,0,0), 2) 

cv2.imshow('test', resizedMap)
cv2.waitKey(0)

cv2.imwrite('/home/saptarshi/PythonCode/Junction/Maps/TripleLaneRaw.png', resizedMap)

