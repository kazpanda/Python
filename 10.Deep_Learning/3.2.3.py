# -----------------------------------------
# 3.2.3 スッテップ関数のグラフ 
# -----------------------------------------
import numpy as np
import matplotlib.pylab as plt

# ステップ関数
def step_function(x):
    return np.array(x > 0, dtype=np.int)

# シグモイド関数
def sigmoid(x):
    return 1/ (1 + np.exp(-x))

# ReLU関数
def relu(x):
    return np.maximum(0,x)


x = np.arange(-5.0,5.0,0.1)

#y = step_function(x)
#y = sigmoid(x)
y = relu(x)

plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()