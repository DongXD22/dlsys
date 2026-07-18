import math
import operator
from functools import reduce
from typing import Any, Callable, Iterable, Union

import numpy as np

from . import ndarray_backend_numpy
from . import ndarray_backend_cpu  # type: ignore[attr-defined]


# math.prod not in Python 3.7
def prod(x: Iterable[int]) -> int:
    return reduce(operator.mul, x, 1)


class BackendDevice:
    """A backend device, wrapps the implementation module."""

    def __init__(self, name: str, mod: Any) -> None:
        self.name: str = name
        self.mod: Any = mod

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BackendDevice) and self.name == other.name

    def __repr__(self) -> str:
        return self.name + "()"

    def __getattr__(self, name: str) -> Any:
        return getattr(self.mod, name)

    def enabled(self) -> bool:
        return self.mod is not None

    def randn(self, *shape: int, dtype: str = "float32") -> "NDArray":
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return NDArray(np.random.randn(*shape).astype(dtype), device=self)

    def rand(self, *shape: int, dtype: str = "float32") -> "NDArray":
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return NDArray(np.random.rand(*shape).astype(dtype), device=self)

    def one_hot(self, n: int, i: int, dtype: str = "float32") -> "NDArray":
        return NDArray(np.eye(n, dtype=dtype)[i], device=self)

    def empty(self, shape: tuple[int, ...], dtype: str = "float32") -> "NDArray":
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        return NDArray.make(shape, device=self)

    def full(self, shape: tuple[int, ...], fill_value: float, dtype: str = "float32") -> "NDArray":
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        arr = self.empty(shape, dtype)
        arr.fill(fill_value)
        return arr


def cuda() -> BackendDevice:
    """Return cuda device"""
    try:
        from . import ndarray_backend_cuda  # type: ignore[attr-defined]

        return BackendDevice("cuda", ndarray_backend_cuda)
    except ImportError:
        return BackendDevice("cuda", None)


def cpu_numpy() -> BackendDevice:
    """Return numpy device"""
    return BackendDevice("cpu_numpy", ndarray_backend_numpy)


def cpu() -> BackendDevice:
    """Return cpu device"""
    return BackendDevice("cpu", ndarray_backend_cpu)


def default_device() -> BackendDevice:
    return cpu_numpy()


def all_devices() -> list[BackendDevice]:
    """return a list of all available devices"""
    return [cpu(), cuda(), cpu_numpy()]


class NDArray:
    """A generic ND array class that may contain multipe different backends
    i.e., a Numpy backend, a native CPU backend, or a GPU backend.

    This class will only contains those functions that you need to implement
    to actually get the desired functionality for the programming examples
    in the homework, and no more.

    For now, for simplicity the class only supports float32 types, though
    this can be extended if desired.
    """

    _shape: tuple[int, ...]
    _strides: tuple[int, ...]
    _offset: int
    _device: BackendDevice
    _handle: Any

    def __init__(self, other, device=None):
        """Create by copying another NDArray, or from numpy"""
        if isinstance(other, NDArray):
            # create a copy of existing NDArray
            if device is None:
                device = other.device
            self._init(other.to(device) + 0.0)  # this creates a copy
        elif isinstance(other, np.ndarray):
            # create copy from numpy array
            device = device if device is not None else default_device()
            array = self.make(other.shape, device=device)
            array.device.from_numpy(np.ascontiguousarray(other), array._handle)
            self._init(array)
        else:
            # see if we can create a numpy array from input
            array = NDArray(np.array(other), device=device)
            self._init(array)

    def _init(self, other: "NDArray") -> None:
        self._shape = other._shape
        self._strides = other._strides
        self._offset = other._offset
        self._device = other._device
        self._handle = other._handle

    @staticmethod
    def compact_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
        """Utility function to compute compact strides"""
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):
            res.append(stride)
            stride *= shape[-i]
        return tuple(res[::-1])

    @staticmethod
    def make(
        shape: tuple[int, ...],
        strides: tuple[int, ...] | None = None,
        device: BackendDevice | None = None,
        handle: Any = None,
        offset: int = 0,
    ) -> "NDArray":
        """使用给定属性创建一个新的 NDArray。如果 handle 为 None，
        这将分配内存；否则将使用现有数组的句柄。"""
        array = NDArray.__new__(NDArray)
        array._shape = tuple(shape)
        array._strides = NDArray.compact_strides(shape) if strides is None else strides
        array._offset = offset
        array._device = device if device is not None else default_device()
        if handle is None:
            array._handle = array.device.Array(prod(shape))
        else:
            array._handle = handle
        return array

    ### Properies and string representations
    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def strides(self) -> tuple[int, ...]:
        return self._strides

    @property
    def device(self) -> BackendDevice:
        return self._device

    @property
    def dtype(self) -> str:
        # only support float32 for now
        return "float32"

    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        return len(self._shape)

    @property
    def size(self) -> int:
        return prod(self._shape)

    def __repr__(self) -> str:
        return "NDArray(" + self.numpy().__str__() + f", device={self.device})"

    def __str__(self) -> str:
        return self.numpy().__str__()

    ### Basic array manipulation
    def fill(self, value: float) -> None:
        """Fill (in place) with a constant value."""
        self._device.fill(self._handle, value)

    def to(self, device: BackendDevice) -> "NDArray":
        """Convert between devices, using to/from numpy calls as the unifying bridge."""
        if device == self.device:
            return self
        else:
            return NDArray(self.numpy(), device=device)

    def numpy(self) -> np.ndarray:
        """convert to a numpy array"""
        return self.device.to_numpy(
            self._handle, self.shape, self.strides, self._offset
        )

    def is_compact(self) -> bool:
        """Return true if array is compact in memory and internal size equals product
        of the shape dimensions"""
        return (
            self._strides == self.compact_strides(self._shape)
            and prod(self.shape) == self._handle.size
        )

    def compact(self) -> "NDArray":
        """Convert a matrix to be compact"""
        if self.is_compact():
            return self
        else:
            out = NDArray.make(self.shape, device=self.device)
            self.device.compact(
                self._handle, out._handle, self.shape, self.strides, self._offset
            )
            return out

    def as_strided(self, shape: tuple[int, ...], strides: tuple[int, ...]) -> "NDArray":
        """Restride the matrix without copying memory."""
        assert len(shape) == len(strides)
        return NDArray.make(
            shape, strides=strides, device=self.device, handle=self._handle, offset=self._offset
        )

    @property
    def flat(self) -> "NDArray":
        return self.reshape((self.size,))

    def reshape(self, new_shape: tuple[int, ...]) -> "NDArray":
        """
            在不复制内存的情况下重塑矩阵。这将返回一个与重塑后的数组对应，
            但指向与原数组相同内存的矩阵。

            引发：
                如果当前形状的乘积不等于新形状的乘积，或者矩阵不是紧凑的，
                则引发 ValueError。

            参数：
                new_shape (tuple)：数组的新形状

            返回：
                NDArray：重塑后的数组；这将指向同一内存
        """

        ### BEGIN YOUR SOLUTION
        if prod(self.shape) != prod(new_shape) or (not self.is_compact()):
            raise ValueError
        return NDArray.make(new_shape,NDArray.compact_strides(new_shape),self.device,self._handle,self._offset)
        ### END YOUR SOLUTION

    def permute(self, new_axes: tuple[int, ...]) -> "NDArray":
        """
        调整维度的顺序。new_axes 描述对现有轴的一个排列，例如：
            - 如果我们有一个维度为“BHWC”的数组，那么 .permute((0,3,1,2)) 
            会将其转换为“BCHW”顺序。
            - 对于二维数组，.permute((1,0)) 会转置该数组。
        与 reshape 类似，此操作不应复制内存，
        而是通过仅调整数组的形状/步长来实现维度排列。也就是说，
        它返回一个维度已按需排列的新数组，但该数组指向与原数组相同的内存。

        参数：
            new_axes (tuple)：维度的排列顺序

        返回：
            NDarray：一个维度已排列的新 NDArray 对象，
            指向与原 NDArray 相同的内存（即仅形状和步长发生变化）。
        """

        ### BEGIN YOUR SOLUTION
        new_shape=tuple([self.shape[axe] for axe in new_axes])
        new_strides=tuple([self.strides[axe] for axe in new_axes])

        return NDArray.make(new_shape,new_strides,self.device,self._handle,self._offset)
        ### END YOUR SOLUTION

    def broadcast_to(self, new_shape: tuple[int, ...]) -> "NDArray":
        """
            将数组广播到新的形状。new_shape 的每个元素必须与原形状的对应维度一致，
            除非该数组自身维度的大小为 1（这些维度可被广播到任意大小）。
            与之前的调用类似，这不会复制内存，仅通过修改步幅来实现广播。

            抛出：断言错误，当对于所有 shape[i] != 1 的维度 i，
            new_shape[i] != shape[i] 时发生。

            参数：new_shape (tuple)：要广播到的形状

            返回：NDArray：具有新广播形状的新 NDArray 对象；
            应指向与原数组相同的内存。
        """

        ### BEGIN YOUR SOLUTION
        new_strides=[]
        for i in range(1,len(new_shape)+1):
            idx=-i
            if i<=len(self.shape):
                if self.shape[idx]!=1 and new_shape[idx]!=self.shape[idx]:
                    raise AssertionError
            if new_shape[idx]!=1 and self.shape[idx]==1:
                new_strides.append(0)
            else:
                new_strides.append(self.strides[idx])
        

        return NDArray.make(new_shape,tuple(new_strides[::-1]),self.device,self._handle,self._offset)
            
        ### END YOUR SOLUTION]

    ### Get and set elements

    def process_slice(self, sl: slice, dim: int) -> slice:
        """Convert a slice to an explicit start/stop/step"""
        start, stop, step = sl.start, sl.stop, sl.step
        if start is None:
            start = 0
        if start < 0:
            start = self.shape[dim]
        if stop is None:
            stop = self.shape[dim]
        if stop < 0:
            stop = self.shape[dim] + stop
        if step is None:
            step = 1

        # we're not gonna handle negative strides and that kind of thing
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"
        return slice(start, stop, step)

    def __getitem__(self, idxs: int | slice | tuple[int | slice, ...]) -> "NDArray":
        """
        Python 中的 __getitem__ 运算符允许我们访问数组的元素。
        当传入类似 a[1:5,:-1:2,4,:] 这样的记号时，
        Python 会将其转换为由切片和整数组成的元组
        （对于本例中的 '4' 这种单值，会转换成整数）。
        切片处理起来可能有些麻烦（它有三个属性 .start、.stop、.step），
        这些属性可能为 None 或包含负数，因此为了简单起见，
        我们编写了代码将其转换为一个总是由切片组成的元组，每个维度对应一个切片。

        对于这个切片元组，返回一个包含所需元素的子集数组。
        和之前一样，这完全可以通过为原始数组的“视图”计算新的形状（shape）、
        步长（stride）和偏移量（offset）来实现，使其指向同一内存。

        异常：
            如果切片的尺寸或步长为负，或者切片的数量与数组的维度数不相等，
            则引发 AssertionError（存根代码已引发所有这些错误）。

        参数：
            idxs (tuple): （存根代码处理后）一个切片元素的元组，
            对应于要获取的矩阵子集

        返回：
            NDArray: 一个新的 NDArray 对象，对应于所选的元素子集。
            与之前一样，这不应复制内存，而只需操作新数组的形状/步长/偏移量，
            引用与原始数组相同的数组。
        """

        # handle singleton as tuple, everything as slices
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        slices = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        assert len(slices) == self.ndim, "Need indexes equal to number of dimensions"

        ### BEGIN YOUR SOLUTION
        new_shape=[(idx.stop-idx.start+idx.step-1)//idx.step for idx in slices]
        new_strides=[idx.step*ori for idx,ori in zip(slices,self.strides)]
        new_offset=self._offset
        for idx,stride in zip(slices,self.strides):
            new_offset+=idx.start*stride
        return NDArray.make(tuple(new_shape),tuple(new_strides),self.device,self._handle,new_offset)
        ### END YOUR SOLUTION

    def __setitem__(self, idxs: int | slice | tuple[int | slice, ...], other: Union["NDArray", float]) -> None:
        """Set the values of a view into an array, using the same semantics
        as __getitem__()."""
        view = self.__getitem__(idxs)
        if isinstance(other, NDArray):
            assert prod(view.shape) == prod(other.shape)
            self.device.ewise_setitem(
                other.compact()._handle,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )
        else:
            self.device.scalar_setitem(
                prod(view.shape),
                other,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )

    ### Collection of elementwise and scalar function: add, multiply, boolean, etc

    def ewise_or_scalar(
        self,
        other: Union["NDArray", float],
        ewise_func: Callable[[Any, Any, Any], None],
        scalar_func: Callable[[Any, Any, Any], None],
    ) -> "NDArray":
        """Run either an elementwise or scalar version of a function,
        depending on whether "other" is an NDArray or scalar
        """
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            ewise_func(self.compact()._handle, other.compact()._handle, out._handle)
        else:
            scalar_func(self.compact()._handle, other, out._handle)
        return out

    def __add__(self, other: Union["NDArray", float]) -> "NDArray":
        return self.ewise_or_scalar(
            other, self.device.ewise_add, self.device.scalar_add
        )

    __radd__ = __add__

    def __sub__(self, other: Union["NDArray", float]) -> "NDArray":
        return self + (-other)

    def __rsub__(self, other: Union["NDArray", float]) -> "NDArray":
        return other + (-self)

    def __mul__(self, other: Union["NDArray", float]) -> "NDArray":
        return self.ewise_or_scalar(
            other, self.device.ewise_mul, self.device.scalar_mul
        )

    __rmul__ = __mul__

    def __truediv__(self, other: Union["NDArray", float]) -> "NDArray":
        return self.ewise_or_scalar(
            other, self.device.ewise_div, self.device.scalar_div
        )

    def __neg__(self) -> "NDArray":
        return self * (-1)

    def __pow__(self, other: float) -> "NDArray":
        out = NDArray.make(self.shape, device=self.device)
        self.device.scalar_power(self.compact()._handle, other, out._handle)
        return out

    def maximum(self, other: Union["NDArray", float]) -> "NDArray":
        return self.ewise_or_scalar(
            other, self.device.ewise_maximum, self.device.scalar_maximum
        )

    ### Binary operators all return (0.0, 1.0) floating point values, could of course be optimized
    def __eq__(self, other: Any) -> "NDArray":  # type: ignore[override]
        return self.ewise_or_scalar(other, self.device.ewise_eq, self.device.scalar_eq)

    def __ge__(self, other: Any) -> "NDArray":
        return self.ewise_or_scalar(other, self.device.ewise_ge, self.device.scalar_ge)

    def __ne__(self, other: Any) -> "NDArray":  # type: ignore[override]
        return 1 - (self == other)

    def __gt__(self, other: Any) -> "NDArray":
        return (self >= other) * (self != other)

    def __lt__(self, other: Any) -> "NDArray":
        return 1 - (self >= other)

    def __le__(self, other: Any) -> "NDArray":
        return 1 - (self > other)

    ### Elementwise functions

    def log(self) -> "NDArray":
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_log(self.compact()._handle, out._handle)
        return out

    def exp(self) -> "NDArray":
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_exp(self.compact()._handle, out._handle)
        return out

    def tanh(self) -> "NDArray":
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_tanh(self.compact()._handle, out._handle)
        return out

    ### Matrix multiplication
    def __matmul__(self, other: "NDArray") -> "NDArray":
        """Matrix multplication of two arrays.  This requires that both arrays
        be 2D (i.e., we don't handle batch matrix multiplication), and that the
        sizes match up properly for matrix multiplication.

        In the case of the CPU backend, you will implement an efficient "tiled"
        version of matrix multiplication for the case when all dimensions of
        the array are divisible by self.device.__tile_size__.  In this case,
        the code below will restride and compact the matrix into tiled form,
        and then pass to the relevant CPU backend.  For the CPU version we will
        just fall back to the naive CPU implementation if the array shape is not
        a multiple of the tile size

        The GPU (and numpy) versions don't have any tiled version (or rather,
        the GPU version will just work natively by tiling any input size).
        """

        assert self.ndim == 2 and other.ndim == 2
        assert self.shape[1] == other.shape[0]

        m, n, p = self.shape[0], self.shape[1], other.shape[1]

        # if the matrix is aligned, use tiled matrix multiplication
        if hasattr(self.device, "matmul_tiled") and all(
            d % self.device.__tile_size__ == 0 for d in (m, n, p)
        ):

            def tile(a, tile):
                return a.as_strided(
                    (a.shape[0] // tile, a.shape[1] // tile, tile, tile),
                    (a.shape[1] * tile, tile, a.shape[1], 1),
                )

            t = self.device.__tile_size__
            a = tile(self.compact(), t).compact()
            b = tile(other.compact(), t).compact()
            out = NDArray.make((a.shape[0], b.shape[1], t, t), device=self.device)
            self.device.matmul_tiled(a._handle, b._handle, out._handle, m, n, p)

            return (
                out.permute((0, 2, 1, 3))
                .compact()
                .reshape((self.shape[0], other.shape[1]))
            )

        else:
            out = NDArray.make((m, p), device=self.device)
            self.device.matmul(
                self.compact()._handle, other.compact()._handle, out._handle, m, n, p
            )
            return out

    ### Reductions, i.e., sum/max over all element or over given axis
    def reduce_view_out(self, axis: int | tuple[int, ...] | list[int] | None, keepdims: bool = False) -> tuple["NDArray", "NDArray"]:
        """ Return a view to the array set up for reduction functions and output array. """
        if isinstance(axis, tuple) and not axis:
            raise ValueError("Empty axis in reduce")

        if axis is None:
            view = self.compact().reshape((1,) * (self.ndim - 1) + (prod(self.shape),))
            #out = NDArray.make((1,) * self.ndim, device=self.device)
            out = NDArray.make((1,), device=self.device)

        else:
            if isinstance(axis, (tuple, list)):
                assert len(axis) == 1, "Only support reduction over a single axis"
                axis = axis[0]

            view = self.permute(
                tuple([a for a in range(self.ndim) if a != axis]) + (axis,)
            )
            out = NDArray.make(
                tuple([1 if i == axis else s for i, s in enumerate(self.shape)])
                if keepdims else
                tuple([s for i, s in enumerate(self.shape) if i != axis]),
                device=self.device,
            )
        return view, out

    def sum(self, axis: int | tuple[int, ...] | list[int] | None = None, keepdims: bool = False) -> "NDArray":
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_sum(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def max(self, axis: int | tuple[int, ...] | list[int] | None = None, keepdims: bool = False) -> "NDArray":
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_max(view.compact()._handle, out._handle, view.shape[-1])
        return out



def array(a: Any, dtype: str = "float32", device: BackendDevice | None = None) -> NDArray:
    """Convenience methods to match numpy a bit more closely."""
    dtype = "float32" if dtype is None else dtype
    assert dtype == "float32"
    return NDArray(a, device=device)


def empty(shape: tuple[int, ...], dtype: str = "float32", device: BackendDevice | None = None) -> NDArray:
    device = device if device is not None else default_device()
    return device.empty(shape, dtype)


def full(shape: tuple[int, ...], fill_value: float, dtype: str = "float32", device: BackendDevice | None = None) -> NDArray:
    device = device if device is not None else default_device()
    return device.full(shape, fill_value, dtype)


def broadcast_to(array: NDArray, new_shape: tuple[int, ...]) -> NDArray:
    return array.broadcast_to(new_shape)


def reshape(array: NDArray, new_shape: tuple[int, ...]) -> NDArray:
    return array.reshape(new_shape)


def maximum(a: NDArray, b: NDArray | float) -> NDArray:
    return a.maximum(b)


def log(a: NDArray) -> NDArray:
    return a.log()


def exp(a: NDArray) -> NDArray:
    return a.exp()


def tanh(a: NDArray) -> NDArray:
    return a.tanh()


def sum(a: NDArray, axis: int | tuple[int] | list[int] | None = None, keepdims: bool = False) -> NDArray:
    return a.sum(axis=axis, keepdims=keepdims)


