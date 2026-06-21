"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight:Parameter=Parameter(init.kaiming_uniform(in_features,out_features))
        if bias:
            self.bias:Parameter=Parameter(init.kaiming_uniform(out_features ,1).transpose())
        ### BEGIN YOUR SOLUTION
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return X.matmul(self.weight)+self.bias.broadcast_to((X.shape[0],self.out_features))
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return X.reshape((X.shape[0],-1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x=module.forward(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        z_y=(logits*init.one_hot(logits.shape[1],y)).sum(axes=1)
        return (ops.logsumexp(logits,axes=1)-z_y).sum()/logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight=Parameter(init.ones((dim)))
        self.bias=Parameter(init.zeros((dim)))
        self.running_mean=np.zeros((dim))
        self.running_var=np.ones((dim))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N,D=x.shape
        
        w=self.weight.broadcast_to((x.shape))
        b=self.bias.broadcast_to((x.shape))

        if self.training == True:
            x_bar=ops.summation(x,0)/N
            x_bar_brd=x_bar.broadcast_to(x.shape)
            x_mbar=x-x_bar_brd
            x_bar_sq=x_mbar**2

            x_var=ops.summation(x_bar_sq,0)/N
            x_var_brd=x_var.broadcast_to(x.shape)

            self.running_mean=(1-self.momentum)*self.running_mean+self.momentum*x_bar.realize_cached_data()
            self.running_var=(1-self.momentum)*self.running_var+self.momentum*x_var.realize_cached_data()

            return w*((x-x_bar_brd)/(x_var_brd+self.eps)**0.5)+b
        
        x_bar_brd = Tensor(self.running_mean, requires_grad=False).broadcast_to(x.shape)
        x_var_brd = Tensor(self.running_var, requires_grad=False).broadcast_to(x.shape)

        return w*((x-x_bar_brd)/(x_var_brd+self.eps)**0.5)+b
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight=Parameter(init.ones((dim)))
        self.bias=Parameter(init.zeros((dim)))
        return 
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N,D=x.shape
        w=ops.broadcast_to(self.weight,x.shape)
        b=ops.broadcast_to(self.bias,x.shape)
        x_bar=ops.summation(x,1)/self.dim
        x_bar_brd=x_bar.reshape((N,1)).broadcast_to(x.shape)
        x_mbar=x-x_bar_brd
        x_mbar_sq=x_mbar**2
        x_var=ops.summation(x_mbar_sq,1)/self.dim
        x_var_brd=x_var.reshape((N,1)).broadcast_to(x.shape)
        return w*((x-x_bar_brd)/(x_var_brd+self.eps)**0.5)+b
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:              
            mask=init.randb(*x.shape,p=1-self.p)
            return mask*x/(1-self.p)
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x)+x
        ### END YOUR SOLUTION
