
#Linear Regression from scratch
import numpy as np 
''' Uncomment if you want to see the visualisations'''
from matplotlib import pyplot as plt
from numpy.linalg import inv

data = np.loadtxt('linear_regression.txt', delimiter = ',')

x = np.c_[np.ones(len(data)),data[:,0]]
y = np.c_[data[:,1]]

#closed form solution:

theta = inv(x.T.dot(x)).dot(x.T).dot(y)

y_pred = x.dot(theta)

#plt.scatter(x[:,1],y, color = 'yellow')
#plt.plot(x[:,1],y_pred, linestyle = '--', color = 'red', label = 'Optimal Solution')


#Use Gradient Descent:
def gradient_descent(x,y,theta,learning_rate,n):
    
    theta_history = np.zeros((n,2))
    cost_history = np.zeros((n))  
    N = len(y)
    for i in range(n):
        y_hat = x.dot(theta)
        theta = theta - (1/N)*learning_rate*(x.T.dot((y_hat - y)))
        cost_history = ((1/2*N)*np.sum(np.square(y_hat-y)))
        theta_history[i,:] = theta.T
    return theta ,cost_history, theta_history

theta_g, costs, thetas = gradient_descent(x,y,([0],[1]),0.0004,1000)
#y_pred_g = x.dot(theta_g)
#plt.plot(x[:,1],y_pred_g, linestyle = '--', color = 'blue',label = 'Gradient Descent')
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.legend()
#plt.show()

def lin_cost_func(x,y,theta):
    
    y_hat = x.dot(theta)
    N = len(y)
    cost = ((1/2*N)*np.sum(np.square(y_hat-y))) 
    return(cost)

theta0, theta1 = np.meshgrid(np.linspace(-theta_g[0]*16,theta_g[0]*40,100),np.linspace(-theta_g[1]*2,theta_g[1]*2,100))

#Z = np.zeros(shape = (theta0.size,theta1.size))
Z = np.array([lin_cost_func(x,y,np.array([t1,t2])) for t1,t2 in zip(np.ravel(theta0),np.ravel(theta1))])
Z = Z.reshape(theta0.shape)

plt.contour(theta0,theta1,Z,30, cmap ='jet')
#for i,v1 in enumerate(theta0):
#    for j,v2 in enumerate(theta1):
#        t = np.array((v1,v2))
#        Z[i,j] = gradient_descent(x,y,t,1,1)
#
#plt.contour(theta0,theta1,Z)