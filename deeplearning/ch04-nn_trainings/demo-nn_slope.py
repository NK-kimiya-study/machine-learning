import sys
import os
import numpy as np
sys.path.append(os.pardir)
from common.functions import softmax,cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    
    #重みをランダムな正規分布で初期化
    def __init__(self):
        self.W = np.random.randn(2,3)
    
    #重みと入力のドット積
    def predict(self,x):
        return np.dot(x,self.W)
    
    #入力データから活性化関数で出力したデータと正解データの損失を計算
    def loss(self,x,t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y,t)
        
        return loss
    

net = simpleNet()
print(net.W)
x=np.array([0.6,0.9])
p = net.predict(x)#次の層のニューロンへの入力値
print(p)
print(np.argmax(p))
t = np.array([0,0,1])
print(net.loss(x,t))


#重みによって、損失値を返す関数
def f(W):
    return net.loss(x,t)

#任意の重みに対して、損失関数の勾配を求める
dw = numerical_gradient(f,net.W)
'''
変化率が負の値なら、重りの値を増やすと局所的な最小値に近づく
変化率が正の値なら、重りの値を減らすと局所的な最小値に近づく
勾配降下法はこれを繰り返す
'''
print(dw)


