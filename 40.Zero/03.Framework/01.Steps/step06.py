import numpy as np

# 基底クラス
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None # 微分値


# 関数クラス
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x) # 実装はforwardメソッドにて行う
        output = Variable(y)
        self.input = input # 入力変数
        return output

    # 順伝播
    def forward(self, x):
        raise NotImplementedError()

    # バックプロパゲーション
    def backward(self, gy):
        raise NotImplementedError()    


# 平方根クラス
class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


# Expクラス
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


# 微粉関数
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


# 実装

# インスタンス
A = Square()
B = Exp()
C = Square() 

# 順伝番
x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# 逆伝番
y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)