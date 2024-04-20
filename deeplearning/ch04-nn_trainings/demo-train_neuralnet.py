# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.minist import load_mnist
from demo_two_layer_net import TwoLayerNet

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000  # 繰り返しの回数を適宜設定する
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # バッチ処理の準備
    batch_mask = np.random.choice(train_size, batch_size)# ランダムにバッチを選択するためのインデックスを生成
    x_batch = x_train[batch_mask]# 訓練データからバッチに対応するデータを取得
    t_batch = t_train[batch_mask]# 訓練データからバッチに対応するラベルを取得
    
    # 勾配の計算
    grad = network.gradient(x_batch, t_batch)# 誤差逆伝播法により勾配を計算
    '''
    W1の場合、入力層に784ニューロン、隠れ層に100ニューロンあるので、
    784×100ものパラメーターがあって、各要素の勾配を求めることができる
    パラメータの行進では、勾配に基づいて784×100ものパラメータを更新する
    次のバッチで行進したパラメーターを使って、同じことを繰り返す
    '''
  
    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]# 各パラメータを更新
    
    loss = network.loss(x_batch, t_batch)# 現在のバッチに対する損失を計算
    train_loss_list.append(loss) # 損失をリストに追加
    
    # エポックごとの精度計算
    if i % iter_per_epoch == 0:# エポック毎に実行
        train_acc = network.accuracy(x_train, t_train)# 訓練データ全体に対する精度を計算
        test_acc = network.accuracy(x_test, t_test)# テストデータ全体に対する精度を計算
        train_acc_list.append(train_acc)# 訓練データの精度をリストに追加
        test_acc_list.append(test_acc)# テストデータの精度をリストに追加
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))# 精度を出力

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()