import numpy as np
import numpy.ma as ma
import math
from scipy.stats import norm
from matplotlib.patches import Ellipse
import pickle

particleRegenerateCovariance = 2

# Calulate the yaw (theta) of the target car using the front and rear co-ordinate of the car
def EstimateAngle(x1,y1,x2,y2):
    perpendicular = y2-y1
    base = x2-x1
    angle = math.atan(perpendicular/base)
    return angle

# Error ellispe plot for Particle Covariance
def error_ellipse(ax, xc, yc, cov, sigma=1, **kwargs):
    '''
    Plot an error ellipse contour over your data.
    Inputs:
    ax : matplotlib Axes() object
    xc : x-coordinate of ellipse center
    yc : x-coordinate of ellipse center
    cov : covariance matrix
    sigma : # sigma to plot (default 1)
    additional kwargs passed to matplotlib.patches.Ellipse()
    '''
    w, v = np.linalg.eigh(cov) # assumes symmetric matrix
    order = w.argsort()[::-1]
    w, v = w[order], v[:,order]
    theta = np.degrees(np.arctan2(*v[:,0][::-1]))
    ellipse = Ellipse(xy=(xc,yc),
                    width=2.*sigma*np.sqrt(w[0]),
                    height=2.*sigma*np.sqrt(w[1]),
                    angle=theta, **kwargs)
    ellipse.set_facecolor('none')
    ax.add_artist(ellipse)

# Estimate the mean square w.r.t ground truth trajectory
def MeanSquareError(xValues, yValues, predictedXValues, predictedYValues):
    if(xValues.shape == yValues.shape == predictedXValues.shape == predictedYValues.shape):
        size = xValues.shape[0]
        meanSqaureError = []
        for idx in range(0,size):
            error = ((xValues[idx] - predictedXValues[idx]) ** 2) + ((yValues[idx] - predictedYValues[idx]) ** 2)
            error = math.sqrt(error)
            meanSqaureError.append(error)
        meanSqaureErrorArray = np.array(meanSqaureError)
        return meanSqaureErrorArray
    else:
        print('Value miss match in the X,Y corodinate counts')
        return None

# Calculate the distance from the nearest junction
def JunctionDistance(x, y):
	Junction_x = 120
	Junction_y = 45
	distance = np.sqrt(np.square(x-Junction_x) + np.square(y-Junction_y))
	return distance

def PredictVelocity(xTest):
	vxFileName = '/home/saptarshi/PythonCode/pfilter-master/ModifiedParticleFilter/vxModelParam.sav'
	vyFileName = '/home/saptarshi/PythonCode/pfilter-master/ModifiedParticleFilter/vyModelParam.sav'
	vxModel = pickle.load(open(vxFileName, 'rb'))
	vyModel = pickle.load(open(vyFileName, 'rb'))
	vx = vxModel.predict(xTest.reshape(1, -1))
	vy = vyModel.predict(xTest.reshape(1, -1))
	return (vx,vy)

# Particle Filter class initilize with number of particles
class ParticleFilter(object):

    def __init__(self,numberOfParticles,road,pfType,meanX,meanY,var):

        self.numberOfParticles = numberOfParticles
        #self.particles = self.initializeParticles(self.numberOfParticles)
        self.particles = self.generateParticles(meanX,meanY,var)
        self.weights = np.full(numberOfParticles,1/numberOfParticles)
        self.road = road
        self.pfType = pfType

    # Update function
    def update(self, observed, vx, vy):
        # Propagate the particles using the dynamics function
        propagatedParticles = self.velocity(self.particles, vx, vy)

        if(observed != None):
            # Extract the current observation and update the weights. (Switch for with or without noise in measurement)
            # currentValue = self.noiseFunction(np.array([observed[0], observed[1]]))
            currentValue = np.array([observed[0], observed[1]])

            # Update the weights based on the observation and normalize the weights
            updatedWeights = self.calculateWeights(np.array(propagatedParticles), currentValue)
            normalizedWeights = updatedWeights / np.sum(updatedWeights)
        else:
            if(self.pfType == 'Classical'):
                #Classical Particle Filter Method
                # normalizedWeights = self.weights   this was already there
                # No observation all equal weights
                normalizedWeights  =  np.full(self.numberOfParticles,1/self.numberOfParticles)
            elif (self.pfType == 'Modified'):
                #Modified Particle Filter Method
                updatedWeights = self.modifiedWeightCalculation(np.array(propagatedParticles))
                normalizedWeights = updatedWeights / np.sum(updatedWeights)
            else:
                print('Wrong Filter Type...')
        
        # Estimate the weight efficiency and resample the particles if needed. 
        n_eff = (1.0 / np.sum(normalizedWeights ** 2)) / self.numberOfParticles
        resampledParticles = []
        if(n_eff<1.0):
            resampledIndexes = self.resampleWeights(normalizedWeights)
            for j in range(0,self.numberOfParticles):
                resampledParticles.append(propagatedParticles[resampledIndexes[j]])
            self.particles = np.array(resampledParticles)
            self.weights = np.full(self.numberOfParticles, (1/self.numberOfParticles))
        else:
            self.particles = propagatedParticles
            self.weights = normalizedWeights

        # Estimate the mean from the resapmled particles
        self.particleMean = np.array([0,0])
        for k in range(0,self.numberOfParticles):
            self.particleMean = self.particleMean + (self.particles[k]*self.weights[k])

        # Estimate the Covariance using the Python APIs
        self.particleCov = np.cov(self.particles, rowvar=False, aweights=self.weights)
        
        # Generate particles based on the last mean
        self.particles = self.generateParticles(self.particleMean[0], self.particleMean[1], particleRegenerateCovariance)

    # Caluculate the weights based on the current obsevation
    def calculateWeights(self, particles, currentValue):
        xVals = np.array(particles[:,0])
        yVals = np.array(particles[:,1])
        currentXVal = currentValue[0]
        currentYVal = currentValue[1]
        sigma = 2
        weights = []
        for idx,val in enumerate(xVals):
            subx = (xVals[idx]-currentXVal) ** 2
            subx = subx/(2*sigma*sigma)
            subx = math.exp(-subx)
            suby = (yVals[idx]-currentYVal) ** 2
            suby = suby/(2*sigma*sigma)
            suby = math.exp(-suby)
            weight = subx*suby*self.weights[idx]
            weights.append(weight)
        return weights

    # Caluculate the weights based on the current obsevation
    def modifiedWeightCalculation(self, particles):
        xVals = np.array(particles[:,0])
        yVals = np.array(particles[:,1])
        sigma = 2
        weights = []
        for idx,val in enumerate(xVals):
            particleX = xVals[idx]
            particleY = yVals[idx]
            # Estimate Nearest Point on the road map w.r.t current particle
            #and use that point for weight update
            roadX,roadY = self.EstimateNearestPointOnRoadMap(particleX,particleY)
            subx = (particleX-roadX) ** 2
            subx = subx/(2*sigma*sigma)
            subx = math.exp(-subx)
            suby = (particleY-roadY) ** 2
            suby = suby/(2*sigma*sigma)
            suby = math.exp(-suby)
            weight = self.weights[idx]*subx*suby
            weights.append(weight)
        return weights

    # Resample the weights
    def resampleWeights(self, weights):
        Q = np.cumsum(weights)
        tList = []
        for i in range(0,self.numberOfParticles):
            tList.append(np.random.uniform(0,1))
        t = np.array(tList)
        TList = list(np.sort(t))
        TList.append(1)
        T = np.array(TList)
        i = 0
        j = 0
        Index = []
        while(i<self.numberOfParticles):
            if(T[i]<Q[j]):
                Index.append(j)
                i = i+1
            else:
                j = j+1
        return Index

    # Generate particles based on the provided mean and the covariance. 
    def generateParticles(self, x, y, cov):
        a = norm(x, cov).rvs(size=self.numberOfParticles)
        b = norm(y, cov).rvs(size=self.numberOfParticles)
        res = np.vstack((a,b)).T
        return res

    # Dynamic function to propagate the particles using the vx and vy to the next time stamp
    #Later on this function will be modified to take into account the other features like Lane number, Indicator light etc
    #OR the predicted future state using the state transition model 
    def velocity(self, x, vx, vy):
        newX = x[:,0] + vx
        newY = x[:,1] + vy
        newXY =  np.transpose(np.vstack((newX,newY)))
        return newXY

    # Add noise to the measurement values
    def noiseFunction(self, x):
        noise = np.random.normal(0, 2, x.shape)
        noisyX = x + noise
        return noisyX
    
    # For each particle find the nearest point on the road map and return the nearest point
    # for weight calculation
    def EstimateNearestPointOnRoadMap(self, x, y):
        nearestRoadX = 0
        nearestRoadY = 0
        minDist = 100000
        for val in self.road:
            dist = ((val[0] - x) ** 2) + ((val[1] - y) ** 2)
            dist = math.sqrt(dist)
            if  dist < minDist:
                minDist = dist
                nearestRoadX = val[0]
                nearestRoadY = val[1]
        return nearestRoadX,nearestRoadY