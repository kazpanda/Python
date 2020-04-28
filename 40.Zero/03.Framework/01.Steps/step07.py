import numpy as np

# 基底クラス
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None # 微分値
        self.creator = None 

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator    #1 関数を取得
        if f is not None:
            x = f.input     #2 関数の入力を取得
            x.grad = f.backward(self.grad) #3 関数のbackwardメソッドを呼ぶ
            x.backward()    #3 自分より1つ前の変数のbackwardメソッドを呼ぶ(再帰)  


# 関数クラス
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x) # 実装はforwardメソッドにて行う
        output = Variable(y)
        output.set_creator(self) # 出力変数の生みの親を覚えさせる
        self.input = input # 入力変数
        self.output = output # 出力も覚えさせる
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


'''
# 逆伝番
y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)
'''

'''
# 逆向きに計算グラフのノードを溯る
assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x

y.grad = np.array(1.0)

C = y.creator               #1 関数を取得
b = C.input                 #2 関数の入力を取得
b.grad = C.backward(y.grad) #3 関数のbackwardメソッドを呼ぶ

B = b.creator               #1 関数を取得
a = B.input                 #2 関数の入力を取得
a.grad = B.backward(b.grad) #3 関数のbackwardメソッドを呼ぶ

A = a.creator               #1 関数を取得
x = A.input                 #2 関数の入力を取得
x.grad = A.backward(a.grad) #3 関数のbackwardメソッドを呼ぶ

print(x.grad)
'''

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
y.backward()
print(x.grad)
