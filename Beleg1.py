import numpy as np
import matplotlib .pyplot as plt
import pylab

def createTrainData(tetta0, tetta1, size):
    """Erzeugt die Trainingsdaten"""
    dataX = np.arange(0.0, 1.0, 1.0/size)
    gauss = np.random.normal(0, 0.1, size)
    dataY = (tetta0 + tetta1 * dataX) + gauss
    return dataX, dataY

def scale(data):
    """Skaliert die Daten auf -1 < x < +1"""
    mu = np.sum(data) / len(data)
    std = np.sqrt(np.sum(np.square(data - mu) / (len(data) - 1)))
    return (data - mu) / (2 * std)
    #return (data - min(data))/(max(data)-min(data))

def hypothesis(tetta0, tetta1, dataX):
    """Die Hypothese"""
    return tetta0 + tetta1 * dataX

def costs(dataX, dataY, tetta0, tetta1):
    """Kostenfunktion"""
    factor = 1.0 / (2 * len(dataX))
    return factor * np.sum(np.square(hypothesis(tetta0, tetta1, dataX) - dataY))

def gradiantDesc(tetta0, tetta1, dataX, dataY, learnrate, dataCosts, tettas0, tettas1):
    """Gradienten-Abstiegs-Verfahren"""
    factor = 1.0 / len(dataX)
    sumX = np.sum(tetta0 + tetta1 * dataX - dataY)
    sumY = np.sum((tetta0 + tetta1 * dataX - dataY) * dataX)
    temp0 = tetta0 - (learnrate * (factor * sumX))
    temp1 = tetta1 - (learnrate * (factor * sumY))
    tettas0.append(temp0)
    tettas1.append(temp1)
    dataCosts.append(costs(dataX, dataY, tetta0, tetta1))
    if(costs(dataX, dataY, tetta0, tetta1) - costs(dataX, dataY, temp0, temp1) > 0.0001):
        return gradiantDesc(temp0, temp1, dataX, dataY, learnrate, dataCosts, tettas0, tettas1)
    else:
        return temp0, temp1, dataCosts, tettas0, tettas1

if __name__ == "__main__":
    [x,y] = createTrainData(0, 1, 50)
    x = scale(x)
    #y = scale(y)
    [tetta0, tetta1, dataCosts, tettas0, tettas1]=  gradiantDesc(-1,4,x,y,0.1,[], [], [])
    t = np.arange(min(x), max(x), 0.1)
    g = tetta0 + tetta1 * t

    plt.subplot(131)
    plt.title("Berechnete Gerade und Testdaten")
    plt.xlabel("x $\in$ $N$")
    plt.ylabel("y $\in$ $N$")
    plt.plot(t, g)
    plt.plot(x, y, 'ro')

    plt.subplot(132)
    plt.title("Gradient Descent")
    plt.ylabel("Kosten $\in$ $R$")
    plt.xlabel("Anzahl der Iterationen $\in$ $N$")
    plt.plot(dataCosts)

    plt.subplot(133)
    plt.title("Countourplot")
    contourData = []
    for tetta0 in tettas0:
        contourDataLine = []
        for tetta1 in tettas1:
            contourDataLine.append(costs(x, y, tetta0, tetta1))
        contourData.append(contourDataLine)
    plt.contour(tettas0 , tettas1, contourData)
    plt.show()