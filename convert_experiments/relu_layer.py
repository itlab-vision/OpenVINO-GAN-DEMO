"""
In this file we try to make ONNX custom relu op

"""
import io
import numpy as np
import torch
from torch import nn
import torch.onnx
from torch.nn import functional as F
from torch.autograd import Function

import onnx
import onnxruntime 

##
## Custom ReLU
##

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


##
## Test custom ReLU
##

# Make random input
N, C, H, W = 1,3,4,4
x = torch.randn(N, C, H, W)
# Inference
model = MyReLU()
y = model.forward(x)
# Output
print('Pytorch input\n',x.numpy())
print('Pytorch output\n',y.numpy())

##
## Export custom ReLU to ONNX
##

# Convert to onnx
# Export the model
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "relu.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'])      # the model's output names

##
## Test exported onnx model
##

# Check the model
onnx_model = onnx.load('relu.onnx')
onnx.checker.check_model(onnx_model)
print('The model is checked!')

# Inference
session = onnxruntime.InferenceSession('relu.onnx', None)
input_name = session.get_inputs()[0].name
print('Input tensor name :', input_name)
x = x.numpy()
outputs = session.run([], {input_name: x})[0]

# Output
print('ONNX input\n',x)
print('ONNX output\n',outputs)