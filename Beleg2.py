
from docutils.utils.math.latex2mathml import munder
import numpy as np
import matplotlib .pyplot as plt
import pylab
from sqlalchemy.sql.functions import coalesce
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#========================================================================================================

def createTrainData(tettas, size):
    """Erzeugt die Trainingsdaten"""
    dataX1 = np.random.normal(0, 1, size).reshape(size,1)
    dataX2 = np.random.normal(0, 1, size).reshape(size,1)
    gauss = np.random.normal(0, 1.0, size).reshape(size,1)
    dataY = (-6 + 3 * dataX1 + 3 * dataX2) + gauss
    value1 = np.ones(size).reshape(size,1)
    features = np.concatenate((np.concatenate((value1, dataX1),axis=1), dataX2),axis=1)

    return features, dataY

#========================================================================================================

def scale(data):
    """Skaliert die Daten auf -1 < x < +1"""
    mu = np.sum(data) / len(data)
    std = np.sqrt(np.sum(np.square(data - mu) / (len(data) - 1)))
    return (data - mu) / (2 * std)
    #return (data - min(data))/(max(data)-min(data))

#========================================================================================================

def hypothesis(tettas, featureVektor):
    """Die multivariate lineare Regression"""
    return np.dot(tettas.T, featureVektor)

#========================================================================================================

def costs(tettas, features, dataY):
    """Kostenfunktion"""
    summe = 0.0
    factor = 1.0 / (2 * len(dataY))
    for i in range(len(dataY)):
        summe += np.sum(np.square(hypothesis(tettas, features.T[i]) - dataY))
    return factor * summe

#========================================================================================================

def logisticFunction(tettas, xVektor):
    """Logistische Funktion"""
    return 1.0/ (1+np.exp(-hypothesis(tettas,xVektor)))

#========================================================================================================

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

#========================================================================================================

def logisticRegression(tettas, features):
    z = []
    featureSum = []
    for i in range(len(features)):
        featureSum.append((np.sum(features[i]),logisticFunction(tettas.T, features[i])))
        z.append(logisticFunction(tettas.T, features[i]))
    return z, featureSum

#========================================================================================================

if __name__ == "__main__":
    [features, dataY] = createTrainData(np.array([0,1,1]),100)

    [tettas, dataCosts] = gradiantDesc(np.array([2,1,4]), features, dataY, 0.1, [])
    [tettas2, dataCosts2] = gradiantDesc(np.array([2,1,4]), features, dataY, 0.2, [])
    [tettas3, dataCosts3] = gradiantDesc(np.array([2,1,4]), features, dataY, 0.4, [])
    [tettas4, dataCosts4] = gradiantDesc(np.array([2,1,4]), features, dataY, 0.6, [])
    [tettas5, dataCosts5] = gradiantDesc(np.array([2,1,4]), features, dataY, 0.8, [])
    [tettas6, dataCosts6] = gradiantDesc(np.array([2,1,4]), features, dataY, 1.0, [])

#========================================================================================================

    plt.title("3D Data")
    fig1 = plt.figure(1)
    ax1 = fig1.gca(projection='3d')
    ax1.scatter(features[:,2], features[:,1], dataY[:,0], s=20, c='r', label='DataPoints')
    x1 = np.linspace(-2,2,100)
    y1 = x1
    x1, y1 = np.meshgrid(x1,y1)
    z1 = tettas[0] + tettas[1] * x1 + tettas[2] * y1
    ax1.plot_surface(x1,y1,z1)
    ax1.legend()

    fig2 = plt.figure(2)
    plt.title("Costs")
    plt.plot(np.arange(len(dataCosts)),dataCosts, label = "Learnrate 0.1")
    plt.plot(np.arange(len(dataCosts2)),dataCosts2, label = "Learnrate 0.2")
    plt.plot(np.arange(len(dataCosts3)),dataCosts3, label = "Learnrate 0.4")
    plt.plot(np.arange(len(dataCosts4)),dataCosts4, label = "Learnrate 0.6")
    plt.plot(np.arange(len(dataCosts5)),dataCosts5, label = "Learnrate 0.8")
    plt.plot(np.arange(len(dataCosts6)),dataCosts6, label = "Learnrate 1.0")
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.xlabel('Iterations')
    plt.ylabel('Costs')

#========================================================================================================

    fig3 = plt.figure(3)

    plt.subplot(221)
    cov0 = np.array([[5,-4],[-4,3]])
    mean0 = np.array([2.,3])
    m0 = 100
    r0 = np.random.multivariate_normal(mean0, cov0, m0)

    cov1 = np.array([[5,-3],[-3,3]])
    mean1 = np.array([1.,1])
    m1 = 100
    r1 = np.random.multivariate_normal(mean1, cov1, m1)

    value2 = np.ones(m0+m1).reshape(m0+m1,1)

    x31 = r0.T[0]
    x32 = r1.T[0]
    x33 = scale(np.concatenate([x31,x32]).reshape(m0+m1,1))
    y31 = r0.T[1]
    y32 = r1.T[1]

    y3 = scale(np.concatenate([y31,y32]).reshape(m0+m1,1))
    x3 = np.concatenate((value2, x33),axis=1)

    [tettas_log, dataCosts_log] = gradiantDesc(np.array([10,-8]), x3, y3, 0.1, [])
    [tettas_log2, dataCosts_log2] = gradiantDesc(np.array([10,-8]), x3, y3, 0.2, [])
    [tettas_log3, dataCosts_log3] = gradiantDesc(np.array([10,-8]), x3, y3, 0.4, [])
    [tettas_log4, dataCosts_log4] = gradiantDesc(np.array([10,-8]), x3, y3, 0.6, [])
    [tettas_log5, dataCosts_log5] = gradiantDesc(np.array([10,-8]), x3, y3, 0.8, [])
    [tettas_log6, dataCosts_log6] = gradiantDesc(np.array([10,-8]), x3, y3, 1.0, [])

    plt.scatter(x33[:100],y3[:100], c='b', marker='o')
    plt.scatter(x33[100:],y3[100:], c='r', marker='x')
    x4 = np.linspace(-2,2,100) # 100 linearly spaced numbers
    y4 = tettas_log[0] + tettas_log[1] * x4
    plt.plot(x4,y4)
    plt.xlabel('x1')
    plt.ylabel('x2')

    ax2 = fig3.add_subplot(224)
    x5 = np.concatenate([x31,x32]).reshape(m0+m1,1)
    x5 = np.concatenate((value2, x5),axis=1)
    [y,featureSum] = (logisticRegression(tettas_log,x5))
    featureSum.sort()
    ax2.scatter(*zip(*featureSum))
    ax2.set_ylim(-0.1, 1.1)

    plt.subplot(223)
    plt.title("Costs")
    plt.plot(np.arange(len(dataCosts_log)),dataCosts_log, label = "Learnrate 0.01")
    plt.plot(np.arange(len(dataCosts_log2)),dataCosts_log2, label = "Learnrate 0.02")
    plt.plot(np.arange(len(dataCosts_log3)),dataCosts_log3, label = "Learnrate 0.04")
    plt.plot(np.arange(len(dataCosts_log4)),dataCosts_log4, label = "Learnrate 0.06")
    plt.plot(np.arange(len(dataCosts_log5)),dataCosts_log5, label = "Learnrate 0.08")
    plt.plot(np.arange(len(dataCosts_log6)),dataCosts_log6, label = "Learnrate 0.1")
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.xlabel('Iterations')
    plt.ylabel('Costs')

    plt.subplot(222)
    plt.title("Logistic")
    x2 = np.linspace(-4,10,100) # 100 linearly spaced numbers
    y2 = 1.0/ (1+np.exp(x2 - tettas_log[0]))
    y0 = np.ones(len(x2)) * 0.5
    plt.plot(x2,y2, label = "Log-function")
    plt.plot(x2,y0, label = "f(x) = 0,5")

    plt.vlines(tettas_log[0], -0.1, 1.1, linestyles='dashed', label = "Threshold")
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)

    plt.show()