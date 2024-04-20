import numpy as np

def sum_squared_error(y,t):
    return 0.5 * np.sum((y-t) ** 2)

#例１：「２」の確率が高い場合(0.6)
y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
t = [0,0,1,0,0,0,0,0,0,0]

print(sum_squared_error(np.array(y), np.array(t)))

#例２：「７」の確率が高い場合(0.6)
y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
print(sum_squared_error(np.array(y),np.array(t)))
