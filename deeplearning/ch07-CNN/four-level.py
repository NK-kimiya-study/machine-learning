import numpy as np

x = np.random.rand(10,1,28,28)#ランダムにデータを生成
print(x.shape)

print(x[0].shape)
print(x[1].shape)

print(x[0,0])
