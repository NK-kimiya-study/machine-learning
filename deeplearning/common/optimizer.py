import numpy as np

class SGD:

    """確率的勾配降下法（Stochastic Gradient Descent）"""

    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key] 
class Momentum:
    def __init__(self,lr=0.01,momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self,params,grads):
        #最初の更新時にのみ
        if self.v is None:
            #各パラメータと同じ形状のゼロ配列を持つ辞書を作成
            #これを勾配の移動平均を格納するために使用
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        
        for key in params.keys():
            #現在の勾配に学習率を乗じ、モーメンタムを適用した過去の勾配の移動平均から引く
            #(過去全体の勾配の平均にモーメンタム係数をかけたもの) - 現在の勾配に学習率を乗じたもの
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]
            
            '''
            「モーメンタムはより滑らかな勾配で損失関数の大局的最適解に近づくことができる。」
            1バッチ目
            ・初期パラメータ：パラメータθがある初期値に設定される
            ・勾配の計算: 1バッチ目のデータを用いて誤差逆伝播を行い、パラメータθに対する勾配を計算
            ・更新量の計算: 計算された勾配に学習率を掛けたものを更新量にする初回の場合、モーメンタム項は0から）
            ・パラメータの更新: 初期パラメータにこの更新量を加えて、新しいパラメータ値を得ます。
            
            2バッチ目
            ・勾配の計算: 2バッチ目のデータを用いて、新しいパラメータθに対する勾配を計算
            ・モーメンタムの適用: 1バッチ目の更新量にモーメンタム係数を乗じ、さらに2バッチ目で計算された勾配に学習率を掛けたものを加えて(または差)、新しい更新量を計算
            ・パラメータの更新: この新しい更新量を使用してパラメータθを更新
            
            3バッチ目
            ・勾配の計算: 3バッチ目のデータに基づいて、さらに新しいパラメータθの勾配を計算
            ・モーメンタムの適用: 2バッチ目の更新量にモーメンタム係数を乗じ、3バッチ目で計算された勾配に学習率を掛けたものを加えて、次の更新量を計算
            ・パラメータの更新: 計算された更新量をパラメータθに適用して、パラメータを更新
            '''

'''
AdaGradは各パラメーターに応じた学習係数を決めれる
[1バッチ目]
・初期状態：パラメーターθに初期値が設定
・勾配の計算：データ1バッチ分でθの勾配を計算
・累積勾配の計算：この勾配の二乗を記録に加える。(累積勾配)、初回は0
・更新量の計算：勾配に対して、累積勾配の平方根で割ったもので更新量を求める
・パラメーターの更新：計算した更新量でパラメータθを更新します。

[2バッチ目以降]
・勾配の計算：新しいデータバッチでθの勾配を計算します。
・累積勾配の更新：新しい勾配の二乗を累積勾配に加える。
・更新量の計算：新しい勾配を累積勾配の平方根で割ったもので更新量を求める
・パラメータの更新: 新しい更新量でパラメータを更新します。
'''

class AdaGrad:
    def __init__(self,lr=0.01):
        self.lr = lr
        self.h = None
    
    def update(self,params,grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 0.0000001)
            
            

#モーメンタムとAdaGradの融合がAdam
#一般にSGDよりも他3津の方が速く学習ができて、最終的な精度が高くなることがある

class Adam:

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)