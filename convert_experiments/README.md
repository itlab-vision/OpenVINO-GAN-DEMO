# Конвертация кастомных слоев из Pytorch в ONNX

## Конвертация функции, состоящей из одного onnx слоя 

Чтобы сконвертировать кастомный слой в ONNX, нужно создать создать функцию 
`symbolic`, в которой будет находиться реализация слоя в ONNX

Пример для функции ReLU.
- В методах forward и backward находится Pytorch реализация этой функции
- В методе symbolic находится onnx реализация этой функции  

```
class MyReLUFunction(Function):

    @staticmethod
    def symbolic(g, input):
        return g.op('Relu', input)

    @staticmethod
    def forward(ctx, input):
        ctx.input = ctx
        return input.clamp(0)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input.masked_fill_(ctx.input < 0, 0)
        return grad_input

class MyReLU(nn.Module):

    def forward(self, input):
        return MyReLUFunction.apply(input)
```

Пример запуска можно посмотреть в файле [relu_layer.py](relu_layer.py).


## Источники:

https://github.com/Russzheng/ONNX_custom_layer
https://github.com/pytorch/pytorch/blob/master/torch/onnx/symbolic_opset9.py
https://github.com/onnx/tutorials/blob/master/PyTorchCustomOperator/README.md 
