import numpy as np
import matplotlib.pyplot as plt

# Defining system's properties
# m: mass, k: stiffness, c: damping
m1, m2 = 1, 2
k1, k2 = 8, 4
c1, c2 = 1, 1


# In Runge Kutta, y' = f(y,t). So I calculated the derivatives of each state as a function of others.
# x[0] = mass 1 position, x[0] = mass 1 velocity, x[2] = mass 2 position, x[3] = mass 2 velocity
def dxdt(x, t):
    du1dt = x[1]
    du2dt = (-(c1+c2)*x[1]+c2*x[3]-(k1+k2)*x[0]+k2*x[2])/m1
    du3dt = x[3]
    du4dt = (c2*x[1]-c2*x[3]+k2*x[0]-k2*x[2])/m2
    return np.array([du1dt, du2dt, du3dt, du4dt])


# Define t
tmax = 20
h = .01
time = np.arange(0,tmax,h)


# Initial values at t = 0
x1 = 1
u1 = 0
x2 = 0
u2 = 0
pos1 = [x1]     # list of mass 1 positions over time
vel1 = [u1]     # list of mass 1 velocities over time
pos2 = [x2]     # list of mass 2 positions over time
vel2 = [u2]     # list of mass 2 velocities over time

# Dumping all initial values in a list to be used in the for loop to be accessible to the dxdt function
x = [x1, u1, x2, u2]
# Starting the for loop from the second array as the first array corresponds to t = 0 which its information is
# already given (initial values).
# Runge Kutta functions are defined at each step to calculate the updated the value of x
# each array of x is appended to their corresponding list for plotting
for t in time[1:,]:
    RK1 = h*dxdt(x,t)
    RK2 = h*dxdt(x+RK1/2, t+h/2)
    RK3 = h*dxdt(x+RK2/2, t+h/2)
    RK4 = h*dxdt(x+RK3  , t+h)

    x += (RK1+2*RK2+2*RK3+RK4)/6

    pos1.append(x[0])
    vel1.append(x[1])
    pos2.append(x[2])
    vel2.append(x[3])

# Plotting all 4 variables
plt.plot(time, pos1)
plt.plot(time, vel1)
plt.plot(time, pos2)
plt.plot(time, vel2)
legend = ['pos1','vel1','pos2','vel2']
plt.legend(legend)
plt.grid()
plt.show()
