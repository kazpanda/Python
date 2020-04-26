import numpy as np

# 基底クラス
class Variable:
    def __init__(self, data):
        self.data = data

# 関数クラス
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x) # 実装はforwardメソッドにて行う
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()

# 平方根クラス
class Square(Function):
    def forward(self, x):
        return x ** 2

# 実装
x = Variable(np.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)

