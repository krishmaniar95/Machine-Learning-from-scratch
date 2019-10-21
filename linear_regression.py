import numpy as np
from matplotlib import pyplot as plt
import math
from mpl_toolkits.mplot3d import axes3d
from numpy.linalg import inv

# Load the data
data = np.loadtxt('linear_regression.txt', delimiter = ',')
#separate predictor from target variable
X = np.c_[np.ones(data.shape[0]), data[:,0]]
y = np.c_[data[:,1]]
 

 #First appraoch - Normal equation

def normalEquation(X,y):
    """
    Parameteres: input variables (Table) , Target vector
    Instructions: Complete the code to compute the closed form solution to linear regression and 	save the result in theta.
    Return: coefficinets 
    """
    ## Your codes go here 
    theta = inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
    return theta

theta = normalEquation(X,y)
print(theta)

## Iterative Approach - Gradient Descent 
#
#'''
#Following paramteres need to be set by you - you may need to run your code multiple times to find the best combination 
#'''
#
#
def gradient_descent(X,y,theta,n,learning_rate):
#    """
#    Paramters: input variable , Target variable, theta, number of iteration, learning_rate
#    Instructions: Complete the code to compute the iterative solution to linear regression, in each iteration you will 
#    add the cost of the iteration to a an empty list name cost_hisotry and update the theta.
#    Return: theta, cost_history 
#    """
    cost_history = np.zeros(n)
    theta_history = np.zeros((n,2))
    l = len(y)
    for i in range(n):
        pred = np.dot(X,theta)
        theta = theta - (1/l)*learning_rate*(X.T.dot((pred-y)))
        theta_history[i,:] = theta.T
        cost_history[i] = ((1/2*l)*np.sum(np.square(pred-y)))
        #print((1/2*l)*np.sum(np.square(pred-y)))
    print ('cost here', cost_history)
    return cost_history, theta , theta_history
        
cost_history,theta_grad,theta_history = gradient_descent(X,y,np.array([[-20],[2]]),1000,0.01)


# Plot the cost over number of iterations

plt.title("No. Of iterations V/S Cost")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.plot(cost_history)
plt.show()


# Plot the linear regression line for both gradient approach and normal equation in same plot

plt.title("Linear Regression Lines")
plt.xlabel("X")
plt.ylabel("y")
X_new = np.array([[0],[24]])
X_nc = np.c_[np.ones((2,1)),X_new]
y_norm = X_nc.dot(theta)
y_grad = X_nc.dot(theta_grad)
plt.plot(X[:,1],y,'b.')
l1, = plt.plot(X_new,y_norm,'g-', label = 'Closed Form Solution')
l2, = plt.plot(X_new,y_grad,'b-', label = 'Gradient Descent Solution')
plt.legend(handles=[l1,l2])
plt.show()
## Plot contour plot and 3d plot for the gradient descent approach
theta0 = theta_history[:,0]
theta1 = theta_history[:,1]
Z = np.array(theta0,theta1,cost_history)
#'''
#your plots should be similar to our plots.
#
#'''
#
#
#
#
#
