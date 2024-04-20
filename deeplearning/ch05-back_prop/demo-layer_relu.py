import numpy as np

class Relu:
    def __init__(self):
        self.mask = None
    
    def forward(self,x):
        #入力要素の各要素が0以下であるか確認して、真偽地の配列をself.maskに格納
        self.mask = (x <= 0)
        #入力xのコピーを作成して、変数outに格納
        out = x.copy()
        #self.maskがTurueである位置のout要素を0に置き換える
        out[self.mask] = 0
        return out
    
    def backward(self,dout):
        # 順伝播時に0以下だった入力に対応するdoutの要素を0に設定
        dout[self.mask] = 0
        # 更新されたdoutをdxに代入
        dx = dout
        return dx
    
    x = np.array([[1.0,-0.5],[-2.0,3.0]])
    print(x)
    mask = (x <= 0)
    print(mask)