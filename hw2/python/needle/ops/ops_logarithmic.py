from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return Z-(LogSumExp(axes=1).compute(Z)).reshape((Z.shape[0],1))
        ### END YOUR SOLUTION
    def gradient(self, out_grad, node):
        z, = node.inputs
        # LogSoftmax gradient: out_grad - softmax * sum(out_grad, axis=1)
        z_data = z.realize_cached_data()
        max_z = array_api.max(z_data, axis=1, keepdims=True)
        z_exp = exp(z - Tensor(max_z, requires_grad=False))
        z_sum = summation(z_exp, axes=1)
        softmax = z_exp / broadcast_to(reshape(z_sum, (z.shape[0], 1)), z.shape)
        # sum out_grad along axis=1, then broadcast
        out_sum = reshape(summation(out_grad, axes=1), (z.shape[0], 1))
        return out_grad - softmax * broadcast_to(out_sum, z.shape)


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.log(array_api.sum(array_api.exp(Z-array_api.max(Z,axis=self.axes,keepdims=True)),self.axes))+array_api.max(Z,axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        # LogSumExp gradient: softmax(z) * out_grad
        z, = node.inputs
        z_data = z.realize_cached_data()
        max_z = array_api.max(z_data, axis=self.axes, keepdims=True)
        z_shifted = z - Tensor(max_z, requires_grad=False)
        z_exp = exp(z_shifted)
        z_sum = summation(z_exp, axes=self.axes)
        # Reshape z_sum to allow broadcasting: insert size-1 dims at reduced axes
        if self.axes is not None:
            new_shape = list(z.shape)
            axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
            for ax in axes:
                new_shape[ax] = 1
            z_sum = reshape(z_sum, new_shape)
        softmax = z_exp / broadcast_to(z_sum, z.shape)
        # Reshape out_grad to match z's ndim, then broadcast
        grad_shape = list(z.shape)
        axes = self.axes if isinstance(self.axes, tuple) else (self.axes,) if self.axes is not None else range(len(z.shape))
        for ax in axes:
            grad_shape[ax] = 1
        out_grad = reshape(out_grad, grad_shape)
        return softmax * broadcast_to(out_grad, z.shape)
        ### END YOUR SOLUTION

def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)