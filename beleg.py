import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sympy.functions.elementary.trigonometric import sin
import funcs as f

### FUNCTION ###
def sin_func(x_val):
    return -1 * np.sin(x_val)


### GLOBALS ###
start_x = -10                           # left bound for x values
end_x = 10                              # right bound for x values
pointCount = 10                         # number of data points
noise = 0.1                             # size of noise for data points
max_iter = 1000                         # maximum of iterations for gradient descent algorithm
start_t = np.array([7, 10, 10])         # start theta for gradient descent algorithm
alpha = np.arange(0.1, 0.6, 0.1)        # learning rates for gradient descent algorithm


### CALCULATING AND PLOTING ###

# test data
x, y = f.create_multi_data(sin_func, pointCount, noise, 0, np.pi)

theta = np.ones(pointCount)[:6]
theta1, costs1 = f.gradient_descent(f.sin_transform(x[:6]), y[:6], theta, 0.1, 10000, f.linear_hypothesis, 1., f.squared_error_loss)
theta2, costs2 = f.gradient_descent(f.sin_transform(x[:6]), y[:6], theta, 0.1, 10000, f.linear_hypothesis, 0.1, f.squared_error_loss)
theta3, costs3 = f.gradient_descent(f.sin_transform(x[:6]), y[:6], theta, 0.1, 10000, f.linear_hypothesis, 0.01, f.squared_error_loss)
theta4, costs4 = f.gradient_descent(f.sin_transform(x[:6]), y[:6], theta, 0.1, 10000, f.linear_hypothesis, 0.001, f.squared_error_loss)
theta5, costs5 = f.gradient_descent(f.sin_transform(x[:6]), y[:6], theta, 0.1, 10000, f.linear_hypothesis, 0.0001, f.squared_error_loss)
theta6, costs6 = f.gradient_descent(f.sin_transform(x[:6]), y[:6], theta, 0.1, 10000, f.linear_hypothesis, 0., f.squared_error_loss)

x_new = np.linspace(0,np.pi,100)
y_new1 = f.linear_hypothesis(theta1)(f.addColumnZero(f.sin_transform(x_new)))
y_new2 = f.linear_hypothesis(theta2)(f.addColumnZero(f.sin_transform(x_new)))
y_new3 = f.linear_hypothesis(theta3)(f.addColumnZero(f.sin_transform(x_new)))
y_new4 = f.linear_hypothesis(theta4)(f.addColumnZero(f.sin_transform(x_new)))
y_new5 = f.linear_hypothesis(theta5)(f.addColumnZero(f.sin_transform(x_new)))
y_new6 = f.linear_hypothesis(theta6)(f.addColumnZero(f.sin_transform(x_new)))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.title("Traindata")
ax.plot(x[:6], y[:6], 'r.', marker='o', label = "Trainingsdata")
ax.plot(x[6:], y[6:], 'b.', label = "Validierungsdaten")
ax.plot(x_new, y_new1, '-', label = "Lambda 1")
ax.plot(x_new, y_new2, '-', label = "Lambda 0.1")
ax.plot(x_new, y_new3, '-', label = "Lambda 0.01")
ax.plot(x_new, y_new4, '-', label = "Lambda 0.001")
ax.plot(x_new, y_new5, '-', label = "Lambda 0.0001")
ax.plot(x_new, y_new6, '-', label = "Lambda 0")
ax.plot(np.linspace(0,3.5,100), sin_func(np.linspace(0,3.5,100)), '-', label = "Targetfunktion = -sin(x)")
ax.legend(bbox_to_anchor=(0.0, 0.0), loc=3, borderaxespad = 0.0, prop={'size':8})
ax.set_ylim(-1.5, 0.5)
plt.show()

