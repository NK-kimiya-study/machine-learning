class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x * y
        
        return out
    
    def backward(self, dout):
        dx = dout * self.y#xとyをひっくり返す
        dy = dout * self.x
        
        return dx,dy

apple = 100
apple_num = 2
tax = 1.1

#layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

apple_price = mul_apple_layer.forward(apple,apple_num)
price = mul_tax_layer.forward(apple_price,tax)

print(price)

#backward
dprice = 1
dapple_price,dtax = mul_tax_layer.backward(dprice)
dapple,dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple,dapple_num,dtax)

class AddLayer:
    def __init__(self):
        pass
    
    def forward(self,x,y):
        out = x + y
        return out
    
    def backward(self,dout):
        dx = dout * 1
        dy = dout * 1
        return dx,dy

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

#layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

#forward
apple_price = mul_apple_layer.forward(apple,apple_num)
orange_price = mul_orange_layer.forward(orange,orange_num)
all_price = add_apple_orange_layer.forward(apple_price,orange_price)
price = mul_tax_layer.forward(all_price,tax)

#backward
dprice = 1
dall_price,dtax = mul_tax_layer.backward(dprice)
print(dall_price,dtax)#1.1 650
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
print(dall_price,dorange_price)#1.1,1.1
dorange,dorange_num = mul_orange_layer.backward(dorange_price)
print(dorange,dorange_num)#3.3,#165
dapple,dapple_num = mul_apple_layer.backward(dall_price)
print(dapple,dapple_num)#2.2,110

print(price)

