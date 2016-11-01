
import scipy
import theano
import numpy as np

from theano import tensor, gof
from theano.gradient import DisconnectedType



__all__ = ["cufft", "cuifft"]


class FFTOp(gof.Op):
    __props__ = ("_inverse",)

    def __init__(self, inverse=False):
        super(FFTOp, self).__init__()
        self._inverse = inverse

    def output_type(self, inp):
        # expect complex valued (last axis of size 2) input, returns tensor of same size
        return tensor.TensorType(inp.dtype, broadcastable=[False] * inp.type.ndim)

    def make_node(self, a):
        a = tensor.as_tensor_variable(a)
        if a.ndim < 3:
            raise TypeError('%s: input must have dimension >= 3,  with ' %
                            self.__class__.__name__ +
                            'first dimension batches and last real/imag parts')
        return gof.Apply(self, [a], [self.output_type(a)()])

    def perform(self, node, inputs, output_storage):
        a = inputs[0]

        frames_dtype = a.dtype
        if a.dtype in [np.float32, np.int32]:
            complex_dtype = np.complex64
        elif a.dtype in [np.float64, np.int64]:
            complex_dtype = np.complex128
        else:
            raise ValueError('array type must be 32/64 int/float, but is {}'.format(a.dtype))
        in_data = np.zeros(a.shape[:-1], dtype=complex_dtype)
        in_data.real = a[:, :, 0]
        in_data.imag = a[:, :, 1]

        if self._inverse:
            out = np.fft.ifft(in_data, a.shape[1])
        else:
            out = np.fft.fft(in_data, a.shape[1])
        output_storage[0][0] = np.stack([out.real, out.imag], axis=-1)

    def grad(self, inputs, output_gradients):
        gout, = output_gradients
        return [FFTOp(inverse = not self._inverse)(gout)]


fft_op, ifft_op = FFTOp(inverse=False), FFTOp(inverse=True)

cufft_op, cuifft_op = fft_op, ifft_op


def fft(inp, norm=None):
    scaling = 1.0
    cond_norm = _unitary(norm)
    if cond_norm == "ortho":
        scaling = tensor.sqrt(inp.shape[1])
    return fft_op(inp) / scaling


def ifft(inp, norm=None):
    scaling = 1.0
    cond_norm = _unitary(norm)
    if cond_norm == "ortho":
        scaling = tensor.sqrt(inp.shape[1])
    return ifft_op(inp) * scaling


def cufft(inp, norm=None):
    scaling = 1.0
    cond_norm = _unitary(norm)
    if cond_norm == "ortho":
        scaling = tensor.sqrt(inp.shape[1])
    return cufft_op(inp) / scaling


def cuifft(inp, norm=None):
    scaling = 1.0
    cond_norm = _unitary(norm)
    if cond_norm == "ortho":
        scaling = tensor.sqrt(inp.shape[1])
    return cuifft_op(inp) * scaling


def _unitary(norm):
    if norm not in (None, "ortho", "no_norm"):
        raise ValueError("Invalid value %s for norm, must be None, 'ortho' or "
                         "'no norm'" % norm)
    return norm