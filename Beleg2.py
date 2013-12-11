
from docutils.utils.math.latex2mathml import munder
import numpy as np
import matplotlib .pyplot as plt
import pylab
from sqlalchemy.sql.functions import coalesce
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def createTrainData(tettas, size):
    """Erzeugt die Trainingsdaten"""
    dataX1 = np.random.normal(0, 1, size).reshape(size,1)
    dataX2 = np.random.normal(0, 1, size).reshape(size,1)
    gauss = np.random.normal(0, 0.5, size).reshape(size,1)
    dataY = (tettas[0] + tettas[1] * dataX1 + tettas[2] * dataX2) + gauss
    value1 = np.ones(size).reshape(size,1)
    features = np.concatenate((np.concatenate((value1, dataX1),axis=1), dataX2),axis=1)

    return features, dataY

def scale(data):
    """Skaliert die Daten auf -1 < x < +1"""
    mu = np.sum(data) / len(data)
    std = np.sqrt(np.sum(np.square(data - mu) / (len(data) - 1)))
    return (data - mu) / (2 * std)
    #return (data - min(data))/(max(data)-min(data))


def hypothesis(tettas, featureVektor):
    """Die multivariate lineare Regression"""
    return np.dot(tettas.T, featureVektor)

def costs(tettas, features, dataY):
    """Kostenfunktion"""
    summe = 0.0
    factor = 1.0 / (2 * len(dataY))
    for i in range(len(dataY)):
        summe += np.sum(np.square(hypothesis(tettas, features.T[i]) - dataY))
    return factor * summe

def logisticFunction(tettas, xVektor):
    """Logistische Funktion"""
    return 1.0/ (1+np.exp(-hypothesis(tettas,xVektor)))


def gradiantDesc(tettas, features, dataY, learnrate, dataCosts):
    """Gradienten-Abstiegs-Verfahren"""
    factor = 1.0/len(dataY)
    summe = 0.0
    dataCostsOld = costs(tettas,features.T,dataY)
    for i in range(len(features)):
        summe += np.dot(hypothesis(tettas, features[i]) - dataY[i][0],features[i])
    tettasTmp = tettas - (learnrate * factor * summe)
    dataCostsNew = costs(tettasTmp,features.T,dataY)
    dataCosts.append(dataCostsNew)
    if (dataCostsOld - dataCostsNew > 0.1):
        return gradiantDesc(tettasTmp, features, dataY, learnrate, dataCosts)
    else:
        return tettasTmp, dataCosts

def logisticRegression(tettas, features):
    z = []
    for i in range(len(features)):
       # print hypothesis(tettas.T, features[i])
        z.append(logisticFunction(tettas.T, features[i]))
    return z

if __name__ == "__main__":
    [features, dataY] = createTrainData(np.array([0,1,1]),100)
    [tettas, dataCosts] = gradiantDesc(np.array([2,1,4]), features, dataY, 0.1, [])
    [tettas2, dataCosts2] = gradiantDesc(np.array([2,1,4]), features, dataY, 0.2, [])
    [tettas3, dataCosts3] = gradiantDesc(np.array([2,1,4]), features, dataY, 0.4, [])
    [tettas4, dataCosts4] = gradiantDesc(np.array([2,1,4]), features, dataY, 0.6, [])
    [tettas5, dataCosts5] = gradiantDesc(np.array([2,1,4]), features, dataY, 0.8, [])
    [tettas6, dataCosts6] = gradiantDesc(np.array([2,1,4]), features, dataY, 1.0, [])


    logisticRegression(tettas, features)

    plt.title("3D Data")
    fig1 = plt.figure(1)
    ax = fig1.gca(projection='3d')
    ax.scatter(features[:,2], features[:,1], dataY[:,0], s=20, c='r', label='DataPoints')

    x = np.linspace(-2,2,100)
    y = x
    x, y = np.meshgrid(x,y)
    z = tettas[0] + tettas[1] * x + tettas[2] * y
    ax.plot_surface(x,y,z)
    ax.legend()

    fig2 = plt.figure(2)
    plt.subplot(212)
    plt.title("Costs")
    plt.plot(np.arange(len(dataCosts)),dataCosts, label = "Learnrate 0.1")
    plt.plot(np.arange(len(dataCosts2)),dataCosts2, label = "Learnrate 0.2")
    plt.plot(np.arange(len(dataCosts3)),dataCosts3, label = "Learnrate 0.4")
    plt.plot(np.arange(len(dataCosts4)),dataCosts4, label = "Learnrate 0.6")
    plt.plot(np.arange(len(dataCosts5)),dataCosts5, label = "Learnrate 0.8")
    plt.plot(np.arange(len(dataCosts6)),dataCosts6, label = "Learnrate 1.0")
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.ylabel('Costs')

    y = (logisticRegression(tettas,features))
    y.sort()
    plt.subplot(222)
    plt.title("Logistic")
    plt.plot(y)


    z = logisticRegression(tettas,features)
    #z1 = tettas[0] + z * x + z*y
    fig3 = plt.figure()
    ax = fig3.gca(projection='3d')
    ax.scatter(features[:,2], features[:,1], z, s=20, c='b', label='DataPoints')
    #bx.plot_surface(x, y ,z1)
    ax.legend()

    plt.show()

