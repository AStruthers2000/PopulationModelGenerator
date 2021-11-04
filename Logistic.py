import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("error")
import math

data = pd.read_csv('logisticDataBelgium.csv')

xVals = data['X'].values
yVals = data['Y'].values

def LinearEquation(m, b, tVals):
    line = []
    for t in tVals:
        line.append(round(m*t+b,10))
    return line

def QuadraticEquation(a, b, c, tVals):
    line = []
    for t in tVals:
        line.append(a*t**2+b*t+c)
    return line

def ExponentialEquation(p_0, r, tVals):
   line = []
   for t in tVals:
       line.append(p_0*math.exp(r*t))
   return line

def LogisticEquation(p_0, k, r, tVals):
    line = []
    for t in tVals:
        line.append(p_0*(1/((p_0/k)+(1-(p_0/k))*math.exp(-r*t))))
    return line



def LinearFit(xVals, yVals):
    points = zip(xVals, yVals)
    
    xSquare = sum([x**2 for x in xVals])
    xy = sum([x*y for x, y in points])
    
    numerator = len(xVals)*xy-sum(xVals)*sum(yVals)
    denominator = len(xVals)*xSquare-sum(xVals)**2
    
    m = numerator/denominator
    b = (sum(yVals)-m*sum(xVals))/len(xVals)

    #print("m: {0}\nb: {1}".format(m, b))
    
    points = LinearEquation(m, b, xVals)
    return points

def ExponentialFit(xVals, yVals):
    lnP = [math.log(y) for y in yVals]
    prime_lnP = []
    for x in range(1, len(lnP)):
        prime_lnP.append(lnP[x]-lnP[x-1])
    r = sum(prime_lnP)/len(prime_lnP)

    line = ExponentialEquation(yVals[0], r, xVals)
    return line


def LogisticLinearFit(xVals, yVals, pPrime):
    pPrime_P = []
    for x in range(len(pPrime)):
        pPrime_P.append(pPrime[x]/yVals[x+1])

    xBar = 0
    yBar = 0
    n = len(pPrime_P)
    for i in range(0,n):
        x,y=yVals[i+1], pPrime_P[i]
        xBar += x
        yBar += y
    xBar /= n
    yBar /= n

    m = 0
    numerator = 0
    denominator = 0
    for i in range(0,n):
        numerator += (yVals[i]-xBar)*(pPrime_P[i]-yBar)
        denominator += (yVals[i]-xBar)**2
    m = numerator/denominator
    b = yBar - m*xBar

    line = LinearEquation(m, b, yVals)
    plot2 = plt.figure(2)
    plt.title("P'(t)/P(t) vs. P(t) for Linear Estimate")
    plt.ylabel("P'(t)/P(t)")
    plt.xlabel("Population of Belgium (10 billion people)")
    plt.scatter(np.delete(yVals, 0), pPrime_P)
    plt.plot(yVals, line)
    r = b
    k = -b/m
    #print("Logistic linear r value: {0}".format(r))
    #print("Logistic linear K value: {0}".format(k))
    points = LogisticEquation(yVals[0], k, r, xVals)
    return points

def LogisticQuadraticFit(xVals, yVals, pPrime):
    def ReplaceCol(base, sub, colNum):
        for row in range(len(base[0])):
            base[row][colNum] = sub[row]
        return base

    needsScalar = False
    exitStatus = False

    while not exitStatus:
        scalar = 1e7
        
        if needsScalar:
            xList = [y/scalar for y in yVals]
        else:
            xList = yVals
            
        yList = pPrime
        xList = np.delete(xList, 0)

        coeffList = []


        for matrixCount in range(3):
            xMatrix = [[0 for x in range(3)] for x in range(3)]
            try:
                for x in range(3):
                    for y in range(3):
                        xMatrix[x][y] = round(sum([pow(point, x+y) for point in xList]), 5)
                #print("This worked")
                exitStatus = True
            except RuntimeWarning:
                #print("Needs scalars, trying again")
                needsScalar = True
                break

            xMatrix[0][0] = len(xList)
            
            detMain = np.linalg.det(np.array(xMatrix))

            yMatrix = []
            for x in range(3):
                yMatrix.append(round(sum([yList[index]*pow(xList[index], x) for index in range(len(xList))]), 5))

            subMat = ReplaceCol(xMatrix, yMatrix, matrixCount)

            detSub = np.linalg.det(np.array(subMat))

            coeff = float(detSub)/float(detMain)
            coeffList.append(coeff)

    if needsScalar:
        coeffList[1] /= scalar
        coeffList[2] /= scalar**2

    line = QuadraticEquation(coeffList[2], coeffList[1], coeffList[0], yVals)
    plot3 = plt.figure(3)
    plt.title("P'(t) vs. P(t) for Quadratic Estimate")
    plt.ylabel("P'(t)")
    plt.xlabel("Population of Belgium (10 billion people)")
    plt.scatter(np.delete(yVals, 0), pPrime)
    plt.plot(yVals, line)

    r = coeffList[1]
    k = -coeffList[1]/coeffList[2]
    points = LogisticEquation(yVals[0], k, r, xVals)
    return points
    

def CalculateMSE(line, data):
    diff = []
    diffSquared = []
    for i in range(len(line)):
        diff.append(line[i]-data[i])
    for x in diff:
        diffSquared.append(x**2)
    mean = sum(diffSquared)/len(diffSquared)
    return mean










minMSE = float('inf')
bestModel = ""

pPrime = []
for x in range(1, len(yVals)):
    pPrime.append((yVals[x]-yVals[x-1])/(xVals[x]-xVals[x-1]))

plot1 = plt.figure(1)

plt.scatter(xVals, yVals)
yMin, yMax = plt.gca().get_ylim()

#linear fit
points = LinearFit(xVals, yVals)
modelName = "linear fit"
mse = CalculateMSE(points, yVals)
if mse < minMSE:
    minMSE = mse
    bestModel = modelName
print("Mean-Square Error for {0}: {1}".format(modelName, mse))
plt.figure(1)
plt.plot(xVals, points,'red')


#exponential fit
points = ExponentialFit(xVals, yVals)
modelName = "exponential fit"
mse = CalculateMSE(points, yVals)
if mse < minMSE:
    minMSE = mse
    bestModel = modelName
print("Mean-Square Error for {0}: {1}".format(modelName, mse))
plt.figure(1)
plt.plot(xVals, points, 'green')


#logistic linear fit
points = LogisticLinearFit(xVals, yVals, pPrime)
modelName = "logistic linear fit"
mse = CalculateMSE(points, yVals)
if mse < minMSE:
    minMSE = mse
    bestModel = modelName
print("Mean-Square Error for {0}: {1}".format(modelName, mse))
plt.figure(1)
plt.plot(xVals, points, 'orange')


#logistic quadratic fit
points = LogisticQuadraticFit(xVals, yVals, pPrime)
modelName = "logistic quadratic fit"
mse = CalculateMSE(points, yVals)
if mse < minMSE:
    minMSE = mse
    bestModel = modelName
print("Mean-Square Error for {0}: {1}".format(modelName, mse))
plt.figure(1)
plt.plot(xVals, points, 'magenta')


print("====="*5+"\nThe best model for this data is the {0}, with a Mean-Square Error value of {1}".format(bestModel, minMSE))

plt.title("Population of Belgium vs. Various Models")
plt.ylabel("Population of Belgium (10 million people)")
plt.xlabel("Time (year since 1960)")
plt.legend(["Linear", "Exponential", "Logistic Linear", "Logistic Quadratic", "Population"])
plt.ylim(yMin, yMax)

plot4 = plt.figure(4)
plt.scatter(xVals, yVals)
plt.title("Population of Belgium from 1960 to 2020")
plt.ylabel("Population of Belgium (10 million people)")
plt.xlabel("Time (year since 1960)")
plt.legend(["Population"])

plt.show()


