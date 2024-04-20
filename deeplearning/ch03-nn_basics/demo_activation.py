import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    y = x > 0
    return y.astype(np.int)

x = np.arange(-5.0,5.0,0.1)
y = step_function(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0,5.0,0.1)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()

def softmax(a):
   c = np.max(a)
   exp_a = np.exp(a - c)
   sum_exp_a =  np.sum(exp_a)
   y = exp_a / sum_exp_a
   return y

a = np.array([0.3,2.9,4.0])
y = softmax(a)
print(y)
np.sum(y)

