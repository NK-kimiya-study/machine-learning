import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f,x):
    h = 0.0001
    return (f(x + h) - f(x-h)) / (2*h)

def function_1(x):
    return 0.01*x**2 + 0.1 * x

x = np.arange(0.0,20.0,0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.show()

print(numerical_diff(function_1, 5))
print(numerical_diff(function_1,10))

def function_2(x):
    return x[0] ** 2 + x[1] ** 2

def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

print(numerical_diff(function_tmp1,3.0))

#偏微分　→　一つの変数の変化率に着目
def function_tmp2(x1):
    return 3.0 **2.0 + x1*x1

print(numerical_diff(function_tmp2,4.0))

#勾配　→　変数全体を含めての変化率,ベクトルを使って表す
def numerical_gradient(f,x):
    h = 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    
    return grad

print(numerical_gradient(function_2,np.array([3.0,4.0])))
print(numerical_gradient(function_2,np.array([0.0,2.0])))
print(numerical_gradient(function_2,np.array([3.0,0.0])))

'''
勾配はベクトルで表されて、勾配ベクトルのフィールドを描画することで、
ベクトルのさす方向から、最小値を探索できる。
'''
x0 = np.arange(-2.0, 2.5, 0.25)
x1 = np.arange(-2.0, 2.5, 0.25)
X, Y = np.meshgrid(x0, x1)
X = X.flatten()
Y = Y.flatten()

grad = np.zeros((X.size, 2))
for i in range(X.size):
    grad[i] = numerical_gradient(function_2, np.array([X[i], Y[i]]))

plt.figure()
plt.quiver(X, Y, -grad[:, 0], -grad[:, 1], angles="xy",color="#666666")  # 勾配のプロット
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('x0')
plt.ylabel('x1')
plt.grid()
plt.draw()
plt.show()

#勾配降下法
def gradient_descent(f,init_x,lr=0.01,step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr * grad
    return x

#x0**2 + x1**2の最小値を勾配法で求める例
def function_2(x):
    return x[0] ** 2 + x[1] ** 2

init_x = np.array([-3.0,4.0])
print(gradient_descent(function_2,init_x=init_x,lr=0.1,step_num=100))

        #学習が大きすぎる例
init_x = np.array([-3.0,4.0])
print(gradient_descent(function_2,init_x=init_x,lr=10.0,step_num=100))
        #学習率が小さすぎる例
init_x = np.array([-3.0,4.0])
print(gradient_descent(function_2,init_x=init_x,lr=0.0000000001,step_num=100))