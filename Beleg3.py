__author__ = 'Daniel'
#!/usr/bin/python

import numpy as np
import matplotlib .pyplot as plt


##create trainings data
def create_data():
    xValues = np.arange(0,np.pi+0.1, (np.pi / 5))
    gauss = np.random.normal(0, 0.2, 6)
    yValues = np.sin(-xValues) + gauss
    return xValues, yValues

##transform x values to feature vektor z
def transform(x):
    return np.array([x,x**2,x**3,x**4,x**5])
##
# recaling
# scaling and rescaling

# scale x by given xmean and xstd
def rescale_by(x, xmean, xstd):
  return (x - xmean) / xstd

#compute xmean and xstd
#and scale x
def rescale(x):
  xmean = x.mean(axis=0)
  xstd  = x.std(axis=0)
  x_norm = rescale_by(x, xmean, xstd)
  return x_norm, xmean, xstd

# scale back parameter
def back_scale_parameter(theta, xmean, xstd):
  t = np.ones(theta.size)
  t[0] = theta[0] - (theta[1:] * xmean / xstd).sum()
  t[1:] = theta[1:] / xstd
  return t

def scale_parameter(theta, xmean, xstd):
  t = np.ones(theta.size)
  t[0] = theta[0] + (theta[1:] * xmean).sum()
  t[1:] = theta[1:] * xstd
  return t


##### for logistic regression ####

def logistic_function(x):
  return 1. / (1. + np.exp(-x))

def logistic_hypothesis(theta):
  return lambda X: logistic_function(X.dot(theta))

def cross_entropy_loss(h, y):
  return lambda theta: y * np.log(h(theta)) + (1-y) * np.log (1. - h(theta))


##### for linear regression ####

def linear_hypothesis(theta):
  return lambda X: X.dot(theta)

def squared_error_loss(h, x, y):
  return lambda theta: 1./2. * (h(theta)(x) - y)**2


def sin_hypothesis(theta):
    return lambda X:  theta[5] + X*theta[4] + theta[3]*(X**2) + theta[2]* X**3 + theta[1]*X**4 + theta[0]*X**5
# didactic code to demonstrate gradient decent
# not optimized for speed

def cost_function(X, y, h, l, loss):
  m = len(y)
  return lambda theta: 1./(float(m)) * (loss(h,X,y)(theta).sum())

# add x_0 for all x(Feature Vectors)
def addColumnZero(x):
  m = len(x)
  n = x[0].size + 1
  X = np.ones(shape=(m,n))
  X[:,1:] = x
  return X

def compute_new_theta(X, y, theta, alpha, hypothesis, l):
  m = len(X)
  h = hypothesis(theta)(X)
  error_sum = np.sum(X.T.dot(h - y))
  # update rule for j = 0
  theta[0] = theta[0] - alpha * (1.0 / float(m)) * error_sum
  #update rule for j != 0
  theta[1:] = theta[1:]*(1-alpha*(l/float(m))) - alpha/float(m) * error_sum
  return theta


def gradient_descent(x, y, theta_, alpha, num_iters, hypothesis, l, loss):
  x_norm ,xmean, xstd = rescale(x)
  theta = scale_parameter(theta_, xmean, xstd)
  X = addColumnZero(x_norm)
  assert X[0].size == theta.size
  n = theta.size
  #history_theta = np.ones((num_iters+1, n), dtype=np.float)
  #history_cost = np.ones((num_iters+1), dtype=np.float)
  cost = cost_function(X, y, hypothesis, l, loss)
  #history_cost[0] = cost(theta)
  #history_theta[0] = back_scale_parameter(theta, xmean, xstd)
  for i in xrange(num_iters):
      theta = compute_new_theta(X, y, theta, alpha, hypothesis, l)
    #history_cost[i+1] = cost(theta)
    #history_theta[i+1] = back_scale_parameter(theta, xmean, xstd)
  #return history_theta, history_cost
  return back_scale_parameter(theta, xmean, xstd), cost(theta)

if __name__ == "__main__":
    [xValues, yValues] = create_data()
    z = np.array([transform(xValues[0])])
    for x in xValues[1:]:
        z = np.concatenate((z,np.array([transform(x)])))
    thetas = np.ones(6)
    [tettas, costs] = gradient_descent(z,yValues,thetas, 0.1, 1000, sin_hypothesis,0.001,squared_error_loss)
    [tettas1, costs1] = gradient_descent(z,yValues,thetas, 0.01, 1000, sin_hypothesis,0.001,squared_error_loss)
    [tettas2, costs2] = gradient_descent(z,yValues,thetas, 0.0001, 1000, sin_hypothesis,0.001,squared_error_loss)
    [tettas3, costs3] = gradient_descent(z,yValues,thetas, 0.00001, 1000, sin_hypothesis,1,squared_error_loss)
    print tettas
    print tettas1
    print tettas2
    fig1 = plt.figure(1)
    plt.scatter(xValues,yValues)
    x = np.linspace(-1,3.2,100)
    y = sin_hypothesis(tettas)(x)
    y1 = sin_hypothesis(tettas1)(x)
    y2= sin_hypothesis(tettas2)(x)
    y3 = sin_hypothesis(tettas3)(x)

    plt.plot(x,y)
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.plot(x,y3)
    plt.show()