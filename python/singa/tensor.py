# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# =============================================================================
"""
Example usage::

    import numpy as np
    from singa import tensor
    from singa import device

    # create a tensor with shape (2,3), default CppCPU device and float32
    x = tensor.Tensor((2, 3))
    x.set_value(0.4)

    # create a tensor from a numpy array
    npy = np.zeros((3, 3), dtype=np.float32)
    y = tensor.from_numpy(npy)

    y.uniform(-1, 1)  # sample values from the uniform distribution

    z = tensor.mult(x, y)  # gemm -> z of shape (2, 3)

    x += z  # element-wise addition

    dev = device.get_default_device()
    x.to_device(dev)  # move the data to a gpu device

    r = tensor.relu(x)

    s = tensor.to_numpy(r)  # tensor -> numpy array

There are two sets of tensor functions,

Tensor member functions
    which would change the internal state of the Tensor instance.

Tensor module functions
    which accept Tensor instances as arguments and return Tensor instances.

Every Tesor instance must be initialized before reading data from it.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from builtins import object
import numpy as np
from functools import reduce

from .proto import core_pb2
from . import singa_wrap as singa
from .device import get_default_device


int32 = core_pb2.kInt
float32 = core_pb2.kFloat32
CTensor = singa.Tensor


class Tensor(object):
    '''Python Tensor, which wraps a swig converted Tensor from CPP Tensor.

    Args:
        shape (tuple<int>): a tuple of integers for the tensor shape. If shape
            is not specified, the created tensor is called a dummy tensor.
        device: a swig device. If None, the default host device is used.
        dtype: data type. currently, most operations only accept float32.
        data: a numpy array or swig tensor.
        requires_grad: boolean indicator for computing the gradient.
        stores_grad: boolean indicator for storing and returning the gradient.
                     Some intermediate tensors' gradient can be released
                     during the backward propagation. A tensor may require
                     grad but not store grad; But if a tensor stores grad
                     then it must require grad.
    '''

    def __init__(self, shape=(), device=None, dtype=float32,
                 data=None, requires_grad=True, stores_grad=False,
                 creator=None):
        if device is None:
            device = get_default_device()
        if isinstance(data, np.ndarray):
            self.data = CTensor(list(data.shape), device, dtype)
            copy_from_numpy(self.data, data)
        elif isinstance(data, CTensor):
            self.data = data
            assert data.device == device, 'not the same device'
        else:
            self.data = CTensor(list(shape), device, dtype)

        self.shape = tuple(self.data.shape())
        self.device = device
        self.dtype = self.data.data_type()
        self.requires_grad = requires_grad
        self.stores_grad = stores_grad
        if creator is None:
            from . import autograd
            self.creator = autograd.Dummy(self)
        else:
            self.creator = creator

    def ndim(self):
        '''
        Returns:
            the number of dimensions of the tensor.
        '''
        return self.data.nDim()

    def is_empty(self):
        '''
        Returns:
            True if the tensor is empty according to its shape
        '''
        return self.ndim() == 0

    def is_transpose(self):
        '''
        Returns:
            True if the internal data is transposed; otherwise False.
        '''
        return self.data.transpose()

    def size(self):  # TODO(wangwei) compute size
        '''
        Returns:
            the number of elements of the tensor.
        '''
        return self.data.Size()

    def memsize(self):
        '''
        Returns:
            the number of Bytes allocated for this tensor.
        '''
        return self.data.MemSize()

    def reshape(self, shape):
        '''Change the tensor shape.

        Args:
            shape (list<int>): new shape, which should have the same volumn as
                the original shape.
        '''
        assert product(self.shape) == product(shape), \
            'product of shape should be equal'
        self.shape = shape
        self.data.Reshape(list(shape))

    def reset_like(self, t):
        '''Reset the shape, dtype and device as the given tensor.

        Args:
            t (Tensor)
        '''
        self.data.ResetLike(t.data)
        self.shape = t.shape
        self.device = t.device
        self.dtype = t.dtype

    '''
    def as_type(self, dtype):
        Change the data type.

        Args:
            dtype:
        self.data.AsType(dtype)
    '''

    def to_device(self, device):
        '''Move the tensor data onto a given device.

        Args:
            device: a swig Device converted from CudaGPU or CppCPU or OpenclGPU
        '''
        self.data.ToDevice(device)
        self.device = device

    def to_host(self):
        '''Move the tensor data onto the default host CppCPU device.
        '''
        self.data.ToHost()
        self.device = get_default_device()

    def l2(self):
        '''
        Returns:
            the L2 norm.
        '''
        return self.data.L2()

    def l1(self):
        '''
        Returns:
            the L1 norm.
        '''
        return self.data.L1()

    def set_value(self, x):
        '''Set all elements of the tensor to be the give value.

        Args:
            x (float), a float value to be set to all elements.
        '''
        # assert type(x) == float, 'set value only accepts float input'
        # if isinstance(x, float):
        self.data.SetFloatValue(float(x))

    def copy_from_numpy(self, np_array, offset=0):
        ''' Copy the data from the numpy array.

        Args:
            np_array: source numpy array
            offset (int): destination offset
        '''
        assert np_array.size == self.size(), 'tensor shape should be the same'
        if not np_array.ndim == 1:
            np_array = np_array.flatten()
        dt = np_array.dtype
        if dt == np.float32:
            self.data.CopyFloatDataFromHostPtr(np_array)
        elif dt == np.int or dt == np.int32:
            self.data.CopyIntDataFromHostPtr(np_array)
        else:
            print('Not implemented yet for ', dt)

    def copy_data(self, t):
        '''Copy data from other Tensor instance.

        Args:
            t (Tensor): source Tensor.
        '''
        assert isinstance(t, Tensor), 't must be a singa Tensor instance'
        self.data.CopyData(t.data)

    def clone(self):
        '''
        Returns:
            a new Tensor which does deep copy of this tensor
        '''
        return _call_singa_func(self.data.Clone)

    def T(self):
        ''' shallow copy, negate the transpose field.

        Returns:
            a new Tensor which shares the underlying data memory (shallow copy)
            but is marked as a transposed version of this tensor.
        '''
        return _call_singa_func(self.data.T)

    def copy(self):
        '''shallow copy calls copy constructor of singa::Tensor
        '''
        return _call_singa_func(CTensor, self.data)

    def deepcopy(self):
        '''Same as clone().

        Returns:
            a new Tensor
        '''
        return self.clone()

    def bernoulli(self, p):
        '''Sample 0/1 for each element according to the given probability.

        Args:
            p (float): with probability p, each element is sample to 1.
        '''
        singa.Bernoulli(float(p), self.data)

    def gaussian(self, mean, std):
        '''Generate a value for each element following a Gaussian distribution.

        Args:
            mean (float): mean of the distribution
            std (float): standard variance of the distribution
        '''
        singa.Gaussian(float(mean), float(std), self.data)

    def uniform(self, low, high):
        '''Generate a value for each element following a uniform distribution.

        Args:
            low (float): the lower bound
            high (float): the hight bound
        '''
        singa.Uniform(float(low), float(high), self.data)

    def add_column(self, v):
        '''Add a tensor to each column of this tensor.

        Args:
            v (Tensor): a Tensor to be added as a column to this tensor.
        '''
        singa.AddColumn(v.data, self.data)

    def add_row(self, v):
        '''Add a tensor to each row of this tensor.

        Args:
            v (Tensor): a Tensor to be added as a row to this tensor.
        '''
        singa.AddRow(v.data, self.data)

    def div_column(self, v):
        '''Divide each column of this tensor by v.

        Args:
            v (Tensor): 1d tensor of the same length the column of self.
        '''
        singa.DivColumn(v.data, self.data)

    def div_row(self, v):
        '''Divide each row of this tensor by v.

        Args:
            v (Tensor): 1d tensor of the same length the row of self.
        '''
        singa.DivRow(v.data, self.data)

    def mult_column(self, v):
        '''Multiply each column of this tensor by v element-wisely.

        Args:
            v (Tensor): 1d tensor of the same length the column of self.
        '''
        singa.MultColumn(v.data, self.data)

    def mult_row(self, v):
        '''Multiply each row of this tensor by v element-wisely.

        Args:
            v (Tensor): 1d tensor of the same length the row of self.
        '''
        singa.MultRow(v.data, self.data)

    '''
    python operators (+=, -=, *=, /=) for singa::Tensor unary operators
    '''

    def __iadd__(self, x):
        ''' inplace element-wise addition with a tensor or a float value.

        Args:
            x (float or Tensor):
        '''
        if isinstance(x, Tensor):
            self.data += x.data
        else:
            self.data += float(x)
        return self

    def __isub__(self, x):
        ''' inplace element-wise subtraction with a tensor or a float value.

        Args:
            x (float or Tensor):
        '''

        if isinstance(x, Tensor):
            self.data -= x.data
        else:
            self.data -= float(x)
        return self

    def __imul__(self, x):
        ''' inplace element-wise multiplication with a tensor or a float value.

        Args:
            x (float or Tensor):
        '''
        if isinstance(x, Tensor):
            self.data *= x.data
        else:
            self.data *= float(x)
        return self

    def __idiv__(self, x):
        ''' inplace element-wise division by a tensor or a float value.

        Args:
            x (float or Tensor):
        '''
        if isinstance(x, Tensor):
            self.data /= x.data
        else:
            self.data /= float(x)
        return self

    '''
    python operators (+, -, *, /, <, <=, >, >=) for singa binary operators
    https://docs.python.org/2/library/operator.html#mapping-operators-to-functions
    '''

    def __add__(self, rhs):
        if isinstance(rhs, Tensor):
            return from_raw_tensor(
                singa.__add__(self.data, rhs.data))
        else:
            return _call_singa_func(singa.AddFloat,
                                    self.data, rhs)

    def __sub__(self, rhs):
        if isinstance(rhs, Tensor):
            return from_raw_tensor(
                singa.__sub__(self.data, rhs.data))
        else:
            return _call_singa_func(singa.SubFloat,
                                    self.data, rhs)

    def __mul__(self, rhs):
        if isinstance(rhs, Tensor):
            return from_raw_tensor(
                singa.__mul__(self.data, rhs.data))
        else:
            return _call_singa_func(singa.MultFloat,
                                    self.data, rhs)

    def __div__(self, rhs):
        if isinstance(rhs, Tensor):
            return from_raw_tensor(
                singa.__div__(self.data, rhs.data))
        else:
            return _call_singa_func(singa.DivFloat,
                                    self.data, rhs)

    def __truediv__(self, rhs):
        if isinstance(rhs, Tensor):
            return from_raw_tensor(
                singa.__div__(self.data, rhs.data))
        else:
            return _call_singa_func(singa.DivFloat,
                                    self.data, rhs)

    def __lt__(self, rhs):
        if isinstance(rhs, Tensor):
            return from_raw_tensor(
                singa.__lt__(self.data, rhs.data))
        else:
            return _call_singa_func(singa.LTFloat, self.data, rhs)

    def __le__(self, rhs):
        if isinstance(rhs, Tensor):
            return from_raw_tensor(
                singa.__le__(self.data, rhs.data))
        else:
            return _call_singa_func(singa.LEFloat, self.data, rhs)

    def __gt__(self, rhs):
        if isinstance(rhs, Tensor):
            return from_raw_tensor(
                singa.__gt__(self.data, rhs.data))
        else:
            return _call_singa_func(singa.GTFloat, self.data, rhs)

    def __ge__(self, rhs):
        if isinstance(rhs, Tensor):
            return from_raw_tensor(
                singa.__ge__(self.data, rhs.data))
        else:
            return _call_singa_func(singa.GEFloat, self.data, rhs)

    def __radd__(self, lhs):
        lhs = float(lhs)
        one = Tensor(self.shape, self.device, self.dtype)
        one.set_value(lhs)
        one += self
        return one

    def __rsub__(self, lhs):
        lhs = float(lhs)
        one = Tensor(self.shape, self.device, self.dtype)
        one.set_value(lhs)
        one -= self
        return one

    def __rmul__(self, lhs):
        lhs = float(lhs)
        one = Tensor(self.shape, self.device, self.dtype)
        one.set_value(lhs)
        one *= self
        return one

    def __rdiv__(self, lhs):
        lhs = float(lhs)
        one = Tensor(self.shape, self.device, self.dtype)
        one.set_value(lhs)
        one /= self
        return one

    def __rtruediv__(self, lhs):
        lhs = float(lhs)
        one = Tensor(self.shape, self.device, self.dtype)
        one.set_value(lhs)
        one /= self
        return one

''' python functions for global functions in Tensor.h
'''


def from_raw_tensor(t):
    x = Tensor(t.shape(), t.device(), t.data_type())
    x.data = t
    return x


def from_raw_tensors(tt):
    ret = []
    for t in list(tt):
        ret.append(from_raw_tensor(t))
    return ret


def product(shape):
    return reduce(lambda x, y: x * y, shape)


def sizeof(dtype):
    '''
    Returns:
        the number of bytes of the given SINGA data type defined in core.proto
    '''
    return singa.SizeOf(dtype)


def reshape(t, s):
    '''Reshape the input tensor with the given shape.

    Args:
        t (Tensor): the tensor to be changed
        s (list<int>): the new shape, which should have the same volumn as the
            old shape.

    Returns:
        the new Tensor
    '''
    return _call_singa_func(singa.Reshape, t.data, s)


def copy_data_to_from(dst, src, size, dst_offset=0, src_offset=0):
    '''Copy the data between two Tensor instances which could be on different
    devices.

    Args:
        dst (Tensor): destination Tensor
        src (Tensor): source Tensor
        size (int) : number of elements to copy
        dst_offset (int): offset in terms of elements to the start of dst
        src_offset (int): offset in terms of elements to the start of src
    '''
    singa.CopyDataToFrom(dst.data, src.data, size,
                         dst_offset, src_offset)


def from_numpy(np_array):
    '''Create a Tensor instance with the shape, dtype and values from the numpy
    array.

    Args:
        np_array: the numpy array.

    Returns:
        A Tensor instance allocated on the default CppCPU device.
    '''
    assert type(np_array) is np.ndarray, 'Must input numpy array'
    # convert to float32 array
    if np_array.dtype == np.float64 or np_array.dtype == np.float:
        np_array = np_array.astype(np.float32)

    if np_array.dtype == np.int64 or np_array.dtype == np.int:
        np_array = np_array.astype(np.int32)

    if np_array.dtype == np.float32:
        dtype = core_pb2.kFloat32
    else:
        assert np_array.dtype == np.int32, \
            'Only float and int tensors are supported'
        dtype = core_pb2.kInt
    ret = Tensor(np_array.shape, dtype=dtype)
    ret.copy_from_numpy(np_array)
    return ret


def to_host(t):
    '''Copy the data to a host tensor.
    '''
    ret = t.clone()
    ret.to_host()
    return ret


def to_numpy(t):
    '''Copy the tensor into a numpy array.

    Args:
        t (Tensor), a Tensor

    Returns:
        a numpy array
    '''
    th = to_host(t)
    if th.dtype == core_pb2.kFloat32:
        np_array = th.data.GetFloatValue(int(th.size()))
    elif th.dtype == core_pb2.kInt:
        np_array = th.data.GetIntValue(int(th.size()))
    else:
        print('Not implemented yet for ', th.dtype)
    return np_array.reshape(th.shape)


def abs(t):
    '''
    Args:
        t (Tensor): input Tensor

    Returns:
        a new Tensor whose element y = abs(x), x is an element of t
    '''
    return _call_singa_func(singa.Abs, t.data)


def exp(t):
    '''
    Args:
        t (Tensor): input Tensor

    Returns:
        a new Tensor whose element y = exp(x), x is an element of t
    '''
    return _call_singa_func(singa.Exp, t.data)


def log(t):
    '''
    Args:
        t (Tensor): input Tensor

    Returns:
        a new Tensor whose element y = log(x), x is an element of t
    '''
    return _call_singa_func(singa.Log, t.data)


def sigmoid(t):
    '''
    Args:
        t (Tensor): input Tensor

    Returns:
        a new Tensor whose element y = sigmoid(x); x is an element of t
    '''
    return _call_singa_func(singa.Sigmoid, t.data)


def sign(t):
    '''
    Args:
        t (Tensor): input Tensor

    Returns:
        a new Tensor whose element y = sign(x)
    '''
    return _call_singa_func(singa.Sign, t.data)


def sqrt(t):
    '''
    Args:
        t (Tensor): input Tensor

    Returns:
        a new Tensor whose element y = sqrt(x), x is an element of t
    '''
    return _call_singa_func(singa.Sqrt, t.data)


def square(t):
    '''
    Args:
        t (Tensor): input Tensor

    Returns:
        a new Tensor whose element y = x * x, x is an element of t
    '''
    return _call_singa_func(singa.Square, t.data)


def tanh(t):
    '''
    Args:
        t (Tensor): input Tensor

    Returns:
        a new Tensor whose element y = tanh(x), x is an element of t
    '''
    return _call_singa_func(singa.Tanh, t.data)


def sum(t, axis=None):
    '''Sum elements of the input tensor long the given axis.

    Args:
        t (Tensor): input Tensor
        axis (int, optional): if None, the summation is done over all elements;
            if axis is provided, then it is calculated along the given axis,
            e.g. 0 -- sum each column; 1 -- sum each row.

    Returns:
        a float value as the sum of all elements, or a new Tensor
    '''

    if axis is None:
        return singa.SumAsFloat(t.data)
    else:
        return _call_singa_func(singa.Sum, t.data, axis)


def pow(t, x, out=None):
    '''
    Args:
        t (Tensor): input tensor
        x (float or Tensor): y[i] = t[i]^x if x is a float value; otherwise,
            y[i]= t[i]^x[i] if x is a tensor.
        out (None or Tensor): if None, a new Tensor would be constructed to
            store the result; otherwise, the result is put into out.

    Returns:
        the result tensor.
    '''
    if out is None:
        if isinstance(x, Tensor):
            return _call_singa_func(singa.Pow, t.data, x.data)
        else:
            return _call_singa_func(singa.PowFloat, t.data, x)
    else:
        if isinstance(x, Tensor):
            singa.PowWithRet(t.data, x.data, out.data)
        else:
            singa.PowFloatWitRet(t.data, x, out.data)
        return out


def average(t, axis=None):
    '''
    Args:
        t (Tensor): input Tensor
        axis (int, optional): if None, average all elements; otherwise average
            along the given dimension. 0 for averaging each column; 1 for
            averaging each row.

    Returns:
        a float value if axis is None; otherwise, a new Tensor for the result.
    '''
    if t.ndim() > 1:
        return _call_singa_func(singa.Average, t.data, axis)
    else:
        return singa.SumAsFloat(t.data) / t.size()


def softmax(t, out=None):
    '''Apply SoftMax for each row of the Tensor.
    Args:
        t (Tensor): the input 1d or 2d tensor
        out (Tensor, optional): if not None, it is used to store the result
    Returns:
        the result Tensor
    '''
    if out is None:
        return _call_singa_func(singa.SoftMax, t.data)
    else:
        singa.SoftMax(t.data, out.data)
        return out


def lt(t, x):
    '''Elementi-wise comparison for t < x

    Args:
        t (Tensor): left hand side operand
        x (Tensor or float): right hand side operand

    Returns:
        a Tensor with each element being t[i] < x ? 1.0f:0.0f,
        or t[i] < x[i] ? 1.0f:0.0f
    '''
    return t < x


def le(t, x):
    '''Elementi-wise comparison for t <= x.

    Args:
        t (Tensor): left hand side operand
        x (Tensor or float): right hand side operand

    Returns:
        a Tensor with each element being t[i] <= x ? 1.0f:0.0f,
        or t[i] <= x[i] ? 1.0f:0.0f
    '''
    return t <= x


def gt(t, x):
    '''Elementi-wise comparison for t > x.

    Args:
        t (Tensor): left hand side operand
        x (Tensor or float): right hand side operand

    Returns:
        a Tensor with each element being t[i] > x ? 1.0f:0.0f,
        or t[i] > x[i] ? 1.0f:0.0f
    '''
    return t > x


def ge(t, x):
    '''Elementi-wise comparison for t >= x.

    Args:
        t (Tensor): left hand side operand
        x (Tensor or float): right hand side operand

    Returns:
        a Tensor with each element being t[i] >= x ? 1.0f:0.0f,
        or t[i] >= x[i] ? 1.0f:0.0f
    '''
    return t >= x


def add(lhs, rhs, ret=None):
    '''Elementi-wise addition.

    Args:
        lhs (Tensor)
        rhs (Tensor)
        ret (Tensor, optional): if not None, the result is stored in it;
            otherwise, a new Tensor would be created for the result.

    Returns:
        the result Tensor
    '''
    if ret is None:
        # call Tensor.__add__()
        return lhs + rhs
    else:
        if isinstance(rhs, Tensor):
            singa.Add(lhs.data, rhs.data, ret.data)
        else:
            singa.AddFloatWithRet(lhs.data, rhs, ret.data)
        return ret


def sub(lhs, rhs, ret=None):
    '''Elementi-wise subtraction.

    Args:
        lhs (Tensor)
        rhs (Tensor)
        ret (Tensor, optional): if not None, the result is stored in it;
            otherwise, a new Tensor would be created for the result.

    Returns:
        the result Tensor
    '''
    if ret is None:
        # call Tensor.__sub__()
        return lhs - rhs
    else:
        if isinstance(rhs, Tensor):
            singa.Sub(lhs.data, rhs.data, ret.data)
        else:
            singa.SubFloatWithRet(lhs.data, rhs, ret.data)
        return ret


def eltwise_mult(lhs, rhs, ret=None):
    '''Elementi-wise multiplication.

    Args:
        lhs (Tensor)
        rhs (Tensor)
        ret (Tensor, optional): if not None, the result is stored in it;
            otherwise, a new Tensor would be created for the result.

    Returns:
        the result Tensor
    '''

    if ret is None:
        # call Tensor.__mul__()
        return lhs * rhs
    else:
        if isinstance(rhs, Tensor):
            singa.EltwiseMult(lhs.data, rhs.data,
                              ret.data)
        else:
            singa.EltwiseMultFloatWithRet(lhs.data, rhs,
                                          ret.data)
        return ret


def mult(A, B, C=None, alpha=1.0, beta=0.0):
    '''Do matrix-matrix or matrix-vector multiplication.

    This function returns C = alpha * A * B + beta * C

    Args:
        A (Tensor): 2d Tensor
        B (Tensor): If B is a 1d Tensor, GEMV would be invoked for matrix-vector
            multiplication; otherwise GEMM would be invoked.
        C (Tensor, optional): for storing the result; If None, a new Tensor
            would be created.
        alpha (float)
        beta (float)

    Returns:
        the result Tensor
    '''
    if C is None:
        return _call_singa_func(singa.Mult, A.data, B.data)
    else:
        singa.MultWithScale(alpha, A.data, B.data,
                            beta, C.data)
        return C


def einsum(ops, *args):
    '''
    function TODO list to finish the function in cpp(just like numpy function):
    1.sum(A,axis = None)
    2.repeat(A,repeats)
    3.transpose(A,axes = None)
    Do the matrix to matrix einsum calculation according to the operands
    Warning : this function could only support two matrix' einsum calcultion
    Args:
        ops(string):
            the string specifies the subscripts for summation such as 'ki,kj->kij'
            Here all the 26 lowercase letter can be used here.
        arg(list of array_like):
                These are the tensors for the operation,but here only support two tensors.
    Returns: Singa.Tensor
        the output matirx of the einsum calculation
    The best way to understand this function is to try the examples below:
    A_ = [0,1,2,3,4,5,6,7,8,9,10,11]
    A = A_.reshape(4,3)
    B = A_.reshape(3,4)

    Here this einsum calculation is the same as normal 'mult'
    Res = einsum('ij,jk->ik',A,B)

    >>> [[ 20  23  26  29]
         [ 56  68  80  92]
         [ 92 113 134 155]
         [128 158 188 218]]

    A_ = [0,1,2,3,4,5,6,7,8,9,10,11]
    A = A_.reshape(4,3)
    B = A_.reshape(4,3)

    Here the einsum calculation is the same as normol 'eltwise_mult'
    Res = einsum('ki,ki->ki',A,B)

    >>> [[  0   1   4]
         [  9  16  25]
         [ 36  49  64]
         [ 81 100 121]]

    A = [0,1,2,3,4,5,6,7,8,9,10,11]
    A = A.reshape(4,3)

    Res = einsum('ki,kj->kij',A,A)
    >>> [[[  0   0   0]
          [  0   1   2]
          [  0   2   4]]
         [[  9  12  15]
          [ 12  16  20]
          [ 15  20  25]]
         [[ 36  42  48]
          [ 42  49  56]
          [ 48  56  64]]
         [[ 81  90  99]
          [ 90 100 110]
          [ 99 110 121]]]

    A_ = [0,1,2,3,4,5,6,7,8,9,10,11]
    A = A_.reshape(3,2,2)

    Res = einsum('kia,kja->kij',A,A)
    >>> [[[  1   3]
          [  3  13]]
         [[ 41  59]
          [ 59  85]]
         [[145 179]
          [179 221]]]
    '''

    if len(ops) == 0:
        raise ValueError("No input operands")

    if len(args) != 2:
        raise ValueError("Currently only two operands are supported")
    # to get the input and output ops
    inputops, outputops = ops.split('->')
    inputops = inputops.split(',')

    # to get the two input tensor
    A = args[0]
    B = args[1]

    if A.ndim() != len(inputops[0]) or B.ndim() != len(inputops[1]):
        raise ValueError("input dim doesn't match operands")

    # to get the indices in input but not in output
    sums = sorted(list((set(inputops[0]) | set(inputops[1])) - set(outputops)))

    # to get the indices that A and B use to broadcast to each other
    broadcast_A = sorted(list(set(inputops[1]) - set(inputops[0])))
    broadcast_B = sorted(list(set(inputops[0]) - set(inputops[1])))
    # to get all the indices in input
    outputall = sorted(list(set(inputops[0]) | set(inputops[1])))

    # Map indices to axis integers
    sums = [outputall.index(x) for x in sums]
    broadcast_idA = [inputops[1].find(x) for x in broadcast_A]
    broadcast_idB = [inputops[0].find(x) for x in broadcast_B]

    broadcast_a = [B.shape[x] for x in broadcast_idA]
    broadcast_b = [A.shape[x] for x in broadcast_idB]

    # get the the transpose and reshape parameter used in the elementwise
    # calculation
    transpose_A = [(list(inputops[0]) + broadcast_A).index(x)
                   for x in outputall]
    transpose_B = [(list(inputops[1]) + broadcast_B).index(x)
                   for x in outputall]

    reshape_A = list(A.shape) + broadcast_a
    reshape_B = list(B.shape) + broadcast_b

    A_ = to_numpy(A)
    B_ = to_numpy(B)

    mult_A = np.repeat(A_, np.product(broadcast_a)).reshape(
        reshape_A).transpose(transpose_A)
    mult_B = np.repeat(B_, np.product(broadcast_b)).reshape(
        reshape_B).transpose(transpose_B)

    if mult_A.shape != mult_B.shape:
        raise ValueError("Error: matrix dimension mismatch")
    res_ = np.multiply(mult_A, mult_B)

    # reduce the axis and find the final transpose for the output
    sum_R = sorted(sums, reverse=True)
    for i in sum_R:
        res_ = res_.sum(axis=i)
    transpose_res = [sorted(list(outputops)).index(x) for x in list(outputops)]
    res_ = res_.transpose(transpose_res)
    res = from_numpy(res_)

    return res


def div(lhs, rhs, ret=None):
    '''Elementi-wise division.

    Args:
        lhs (Tensor)
        rhs (Tensor)
        ret (Tensor, optional): if not None, the result is stored in it;
            otherwise, a new Tensor would be created for the result.

    Returns:
        the result Tensor
    '''
    if ret is None:
        # call Tensor.__div__()
        return lhs / rhs
    else:
        if isinstance(rhs, Tensor):
            singa.Div(lhs.data, rhs.data, ret.data)
        else:
            singa.DivFloatWithRet(lhs.data, rhs, ret.data)
        return ret


def axpy(alpha, x, y):
    '''Element-wise operation for y += alpha * x.

    Args:
        alpha (float)
        x (Tensor)
        y (Tensor)

    Returns:
        y
    '''
    singa.Axpy(float(alpha), x.data, y.data)
    return y


def bernoulli(p, t):
    '''Generate a binary value for each element of t.

    Args:
        p (float): each element is 1 with probability p; and 0 with 1 - p
        t (Tensor): the results are put into t

    Returns:
        t
    '''
    singa.Bernoulli(float(p), t.data)
    return t


def gaussian(mean, std, t):
    '''Generate values following a Gaussian distribution.

    Args:
        mean (float): the mean of the Gaussian distribution.
        std (float): the standard variance of the Gaussian distribution.
        t (Tensor): the results are put into t

    Returns:
        t
    '''
    singa.Gaussian(float(mean), float(std), t.data)
    return t


def uniform(low, high, t):
    '''Generate values following a Uniform distribution.

    Args:
        low (float): the lower bound
        hight (float): the higher bound
        t (Tensor): the results are put into t

    Returns:
        t
    '''
    singa.Uniform(float(low), float(high), t.data)
    return t


def add_column(alpha, v, beta, M):
    '''Add v to each column of M.

    Denote each column of M as m, m = alpha * v + beta * m

    Args:
        alpha (float)
        v (Tensor)
        beta (float)
        M (Tensor): 2d tensor
    Returns:
        M
    '''
    singa.AddColumnWithScale(float(alpha), float(beta), v.data,
                             M.data)
    return M


def add_row(alpha, v, beta, M):
    '''Add v to each row of M.

    Denote each row of M as m, m = alpha * v + beta * m

    Args:
        alpha (float)
        v (Tensor)
        beta (float)
        M (Tensor): 2d tensor
    Returns:
        M
    '''
    singa.AddRowWithScale(alpha, beta, v.data, M.data)
    return M


def sum_columns(M):
    '''Sum all columns into a single column.

    Args:
        M (Tensor): the input 2d tensor.

    Returns:
        a new Tensor as the resulted column.
    '''
    assert M.ndim() == 2, 'M.nDim() is supposed to be 2'
    ret = Tensor((M.shape[0], 1), M.data.device())
    singa.SumColumns(M.data, ret.data)
    return ret


def sum_rows(M):
    '''Sum all rows into a single row.

    Args:
        M (Tensor): the input 2d tensor.

    Returns:
        a new Tensor as the resulted row.
    '''
    assert M.ndim() == 2, 'M.nDim() is supposed to be 2'
    ret = Tensor((1, M.shape[1]), M.data.device())
    singa.SumRows(M.data, ret.data)
    return ret


''' private functions, internally used
'''


def _call_singa_func(_singa_func, *args):
    ''' this function calls singa global functions that returns Tensor
        and create new python Tensor instance
        e.g., Tensor [singa_func](args...)
    '''
    new_t = Tensor()
    new_t.data = _singa_func(*args)
    new_t.shape = tuple(new_t.data.shape())
    new_t.device = new_t.data.device()
    new_t.dtype = new_t.data.data_type()
    return new_t


def copy_from_numpy(data, np_array):
    '''
    Copy the data from the numpy array.
    '''
    assert np_array.size == data.Size(), \
        'tensor shape should be the same'
    if not np_array.ndim == 1:
        np_array = np_array.flatten()
    dt = np_array.dtype
    if dt == np.float32:
        data.CopyFloatDataFromHostPtr(np_array)
    elif dt == np.int or dt == np.int32:
        data.CopyIntDataFromHostPtr(np_array)
    else:
        print('Not implemented yet for ', dt)
