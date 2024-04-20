#from ..common import functions
import sys
import os
sys.path.append(os.pardir)
from common import functions

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None#損失
        self.y = None#softmaxの出力
        self.t = None#教師データー
    
    def forward(self,x,t):
        self.t = t
        self.y = functions.softmax(x)
        self.loss = functions.cross_entropy_error(self.y,self.t)
        
        return self.loss
    
    def backward(self,dout=1):#上流（出力側）から伝わってきた勾配の値
        batch_size = self.t.shape[0]
        #平均化された勾配
        dx = (self.y - self.t) / batch_size
        
        return dx
