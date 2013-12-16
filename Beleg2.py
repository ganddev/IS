
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
    dataY = (-6 + 3 * dataX1 + 3 * dataX2) + gauss
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
    featureSum = []
    for i in range(len(features)):
        featureSum.append((np.sum(features[i]),logisticFunction(tettas.T, features[i])))
        z.append(logisticFunction(tettas.T, features[i]))
    return z, featureSum

if __name__ == "__main__":
    [features, dataY] = createTrainData(np.array([0,1,1]),100)

    [tettas, dataCosts] = gradiantDesc(np.array([2,1,4]), features, dataY, 0.1, [])
    [tettas2, dataCosts2] = gradiantDesc(np.array([2,1,4]), features, dataY, 0.2, [])
    [tettas3, dataCosts3] = gradiantDesc(np.array([2,1,4]), features, dataY, 0.4, [])
    [tettas4, dataCosts4] = gradiantDesc(np.array([2,1,4]), features, dataY, 0.6, [])
    [tettas5, dataCosts5] = gradiantDesc(np.array([2,1,4]), features, dataY, 0.8, [])
    [tettas6, dataCosts6] = gradiantDesc(np.array([2,1,4]), features, dataY, 1.0, [])

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


    fig3 = plt.figure(3)


    plt.subplot(222)
    plt.title("Logistic")
    x2 = np.linspace(-15,15,100) # 100 linearly spaced numbers
    y2 = 1.0/ (1+np.exp(-x2))
    plt.plot(x2,y2)

    #[z, featureSumme] = logisticRegression(tettas,features)
    ##z1 = tettas[0] + z * x + z*y
    #fig3 = plt.figure()
    #ax3 = fig3.gca(projection='3d')
    #ax3.scatter(features[:,2], features[:,1], z, s=20, c='b', label='DataPoints')
    ##bx.plot_surface(x, y ,z1)
    #ax3.legend()


    plt.subplot(223)
    # class 0:
    # covariance matrix and mean
    cov0 = np.array([[5,-4],[-4,3]])
    mean0 = np.array([2.,3])
    # number of data points
    m0 = 100
    # generate m0 gaussian distributed data points with
    # mean0 and cov0.
    r0 = np.random.multivariate_normal(mean0, cov0, m0)

    # covariance matrix
    cov1 = np.array([[5,-3],[-3,3]])
    mean1 = np.array([1.,1])
    m1 = 100
    r1 = np.random.multivariate_normal(mean1, cov1, m1)

    x31 = r1.T[0].reshape(m0,1)
    #x32 = scale(r1.T[0])
    #x3 = np.union1d(x31,x32).reshape(2*m0,1)
    #x3.append(r1.T[0].reshape(m0,1))
    y31 = r1.T[1].reshape(m0,1)
    #y32 = scale(r1.T[1])
    #y3 = np.union1d(y31,y32).reshape(2*m0,1)
    #y3.append(r1.T[1].reshape(m0,1))

    value2 = np.ones(m0).reshape(m0,1)
    #print x3
    x3 = np.concatenate((value2, x31),axis=1)

    [tettas_log, dataCosts_log] = gradiantDesc(np.array([10,-8]), x3, y31, 0.01, [])
    print dataCosts_log
    plt.scatter(r0[...][...,0], r0[...,1], c='b', marker='o')
    plt.scatter(r1[...,0], r1[...,1], c='r', marker='x')

    x4 = np.linspace(-7,7,100) # 100 linearly spaced numbers
    y4 = tettas_log[0] + tettas_log[1] * x4
    plt.plot(x4,y4)

    plt.subplot(224)
    plt.title("Costs")
    plt.plot(np.arange(len(dataCosts_log)),dataCosts_log, label = "Learnrate 0.1")

    print x3
    [y,featureSum] = (logisticRegression(tettas_log,x3))
    featureSum.sort()
    print featureSum
    ax2 = fig3.add_subplot(221)
    ax2.scatter(*zip(*featureSum))
    ax2.set_ylim(-0.1, 1.1)

    plt.show()