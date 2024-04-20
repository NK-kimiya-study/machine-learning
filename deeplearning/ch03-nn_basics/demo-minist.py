import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.minist import load_mnist
from PIL import Image
import pickle

def sigmoid(x):
    #活性化関数：シグモイド関数の計算式
    return 1 / (1 + np.exp(-x))    

def softmax(a):
   #活性化関数：ソフトマックス関数の計算式
   c = np.max(a)
   exp_a = np.exp(a - c)
   sum_exp_a =  np.sum(exp_a)
   y = exp_a / sum_exp_a
   return y

def get_data():
    #MINISTデータの読み込み
    (x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=False)
    return x_test,t_test

def init_network():
    #学習済みモデルの読み込み
    with open("sample_weight.pkl",'rb') as f:
        network = pickle.load(f)
    return network



def predict(network,x):
    #学習済みモデルの重みとバイアスを設定
    w1,w2,w3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    #データの予測 
    a1 = np.dot(x,w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,w3) + b3
    y = softmax(a3)
    return y

#データの読み込みとネットワークの初期化
x,t = get_data()
network = init_network()

#データを100ずつのバッチに分けて、推論をして、予測した100のデータのうち、
# 正解ラベルとの一致度を調べる
batch_size = 100
accuracy_cnt = 0
for i in range(0,len(x),batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch,axis=1)
    print(p)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
    print(accuracy_cnt)

#全体的のデータの予測精度を調べる      
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))


        
        