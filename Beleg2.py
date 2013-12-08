
from docutils.utils.math.latex2mathml import munder
import numpy as np
import matplotlib .pyplot as plt
import pylab
from sqlalchemy.sql.functions import coalesce


def createTrainData(tettas, size):
    dataX1 = np.arange(0.0, 1.0, 1.0/size).reshape(size,1)
    dataX2 = np.arange(0.0, 1.0, 1.0/size).reshape(size,1)
    gauss = np.random.normal(0,0.1,size).reshape(size,1)
    dataY = (tettas[0]+tettas[1]*dataX1+tettas[2]*dataX2)+gauss
    value1 = np.ones(size).reshape(size,1)
    features = np.concatenate((np.concatenate((value1,dataX1),axis=1),dataX2),axis=1)
    return features, dataY

def scale(data):
    """Skaliert die Daten auf -1 < x < +1"""
    mu = np.sum(data) / len(data)
    std = np.sqrt(np.sum(np.square(data - mu) / (len(data) - 1)))
    return (data - mu) / (2 * std)
    #return (data - min(data))/(max(data)-min(data))


def hypothesis(tettas, featureVektor):
    """Die multivariate lineare Regression"""
    return np.dot(tettas, featureVektor)

def costs(tettas, xMatrix, yVektor):
    """Kostenfunktion"""
    summe = 0.0
    factor = 1.0 / (2 * len(yVektor))
    for i in range(len(xMatrix)):
        summe += np.sum(np.square(hypothesis(tettas.T, np.array([xMatrix[0][i],xMatrix[1][i],xMatrix[2][i]])) - yVektor))
    return factor * summe

def logisticFunction(tettas, xVektor):
    """Logistische Funktion"""
    return 1.0/ (1+np.exp(-hypothesis(tettas,xVektor)))


def gradiantDesc(tettas, features, dataY, tettaJOld, dataCosts, learnrate):
    """Gradienten-Abstiegs-Verfahren"""
    factor = 1/len(dataY)
    summe = 0.0
    for i in range(len(features)):
        summe += np.dot(np.square(hypothesis(tettas.T, np.array([features[0][i],features[1][i],features[2][i]])) - dataY[i])[0],np.array([features[0][i],features[1][i],features[2][i]]))
    print summe
    tettaJTmp = learnrate*factor*summe
    #return tettaJTmp

if __name__ == "__main__":
    tettas = np.array([1,2,3])
    #tettas1 = np.array([[1,2,3],[3,4,3],[5,6,3],[5,6,3]]).T
    #print tettas1
    [features,dataY] = createTrainData(tettas,10)
    gradiantDesc(tettas, features.T, dataY, 0, [], 0.5)

