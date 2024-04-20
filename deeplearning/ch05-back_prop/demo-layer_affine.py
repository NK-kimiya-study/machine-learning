import numpy as np

#長さ2のランダムな実数値からなる1次元配列（ベクトル）を生成
x = np.random.rand(2)
# 2x3のランダムな実数値からなる2次元配列（行列）を生成
w = np.random.rand(2,3)
# 長さ3のランダムな実数値からなる1次元配列（ベクトル）を生成
B = np.random.rand(3)

print(x.shape)
print(w.shape)
print(B.shape)

#アフィン変換
Y = np.dot(x,w) + B
print(Y)

x_dot_w = np.array([[0,0,0],[10,10,10]])
B = np.array([1,2,3])
print(x_dot_w)
print(x_dot_w + B)

dY = np.array([[1,2,3],[4,5,6]])
print(dY)
dB = np.sum(dY,axis=0)
print(dB)

class Affine:
    def __init__(self,w,b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None
    
    def forward(self,x):
        self.x = x
        out = np.dot(x,self.w) + self.b
        
        return out
    
    def backward(self,dout):
        dx = np.dot(dout,self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout,axis=0)
        
        return dx