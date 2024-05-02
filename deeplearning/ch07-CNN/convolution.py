import sys,os
sys.path.append(os.pardir)
from common.util import im2col
import numpy as np

#画像が1枚、チャンネル数が、画像サイズが7×7の4次元配列のデータを要素をランダムに生成
x1 = np.random.rand(1,3,7,7)
col1 = im2col(x1,5,5,stride=1,pad=0)
#9行75列の行列になる(入力データでフィルターを行う部分を二次元で表す)
print(col1.shape)

x2 = np.random.rand(10,3,7,7)
col2 = im2col(x2,5,5,stride=1,pad=0)
print(col2.shape)

