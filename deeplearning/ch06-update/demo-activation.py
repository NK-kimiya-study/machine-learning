import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.random.randn(1000,100)
node_num = 100#各隠れ層のニューロンの数
hidden_layer_size = 5#隠れ層が5層
activations = {}#ここにアクティベーションの結果を格納

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
        
    #ニューラルネットワークの重みの初期化の一種であるXavierで重みを初期化
    #層にデータが入力される前と出力される後の、ニューロン（ノード）の数に基づいて重みを適切な範囲のランダム値で初期化する
    #Xavierはシグモイド関数で使う
    #ReLu関数ではHeの初期値を使う
    w = np.random.randn(node_num, node_num) * np.sqrt(node_num)
    
    z = np.dot(x,w)
    a = sigmoid(z)
    activations[i] = a

for i, a in activations.items():
    plt.subplot(1,len(activations), i + 1)
    plt.title(str(i+1) + "-layer")
    plt.hist(a.flatten(),30,range=(0,1))
plt.show()
    