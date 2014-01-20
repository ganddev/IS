import numpy as np
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

def cross_entropy_loss(h, x, y):
  return lambda theta: -y * np.log(h(theta)(x)) - (1-y) * np.log(1. - h(theta)(x))



##### for linear regression ####

def linear_hypothesis(theta):
  return lambda X: X.dot(theta)

def squared_error_loss(h, x, y):
  return lambda theta: 1./2. * (h(theta)(x) - y)**2

def sin_transform(x):
    z = x.reshape((len(x), -1))
    return np.power(z, np.arange(1, 6))


# didactic code to demonstrate gradient decent
# not optimized for speed

def cost_function(X, y, h, l, loss):
  m = len(y)
  return lambda theta: 1./(float(m)) * (loss(h,X,y)(theta).sum() + np.power(theta, 2).sum()*l)

# add x_0 for all x(Feature Vectors)
def addColumnZero(x):
  m = len(x)
  n = x[0].size + 1
  X = np.ones(shape=(m,n))
  X[:,1:] = x
  return X

def compute_new_theta(X, y, theta, alpha, hypothesis, l):
  m = len(X.T[0])
  h0 = hypothesis(theta[0])(X.T[0])
  h = hypothesis(theta[1:])(X.T[1:].T)

  theta[0] = theta[0] - alpha * (1.0 / float(m)) * np.sum(X.T[0] * (h0 - y))
  theta[1:] = theta[1:]*(1-alpha*(l/float(m))) - (alpha/float(m)) * X.T[1:].dot(h - y)

  return theta


def gradient_descent(X, y, theta_, alpha, num_iters, hypothesis, l, loss):
  x_norm ,xmean, xstd = rescale(X)
  theta = scale_parameter(theta_, xmean, xstd)
  X = addColumnZero(x_norm)
  assert X.T[0].size == theta.size

  cost = cost_function(X, y, hypothesis, l, loss)
  for i in xrange(num_iters):
    theta = compute_new_theta(X, y, theta, alpha, hypothesis, l)

  return back_scale_parameter(theta, xmean, xstd), cost(theta)


def create_multi_data(func, point_count, r, min, max):
    s = np.random.normal(0, r, point_count)
    x_values = np.random.uniform(min, max, point_count)
    y_values = func(x_values) + s
    return x_values, y_values