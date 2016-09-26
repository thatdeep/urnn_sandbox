
import scipy
import theano
import numpy as np

from theano import tensor, gof
from theano.gradient import DisconnectedType



__all__ = ["cufft", "cuifft"]


class FFTOp(gof.Op):
    __props__ = ()

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

        out = np.fft.fft(in_data, a.shape[1])
        output_storage[0][0] = np.stack([out.real, out.imag], axis=-1)

    def grad(self, inputs, output_gradients):
        gout, = output_gradients
        return [ifft(gout)]


class IFFTOp(gof.Op):
    __props__ = ()

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

        if a.dtype in [np.float32, np.int32]:
            complex_dtype = np.complex64
        elif a.dtype in [np.float64, np.int64]:
            complex_dtype = np.complex128

        in_data = np.zeros(a.shape[:-1], dtype=complex_dtype)
        in_data.real = a[:, :, 0]
        in_data.imag = a[:, :, 1]

        out = np.fft.ifft(in_data, a.shape[1])
        output_storage[0][0] = np.stack([out.real, out.imag], axis=-1)

    def grad(self, inputs, output_gradients):
        gout, = output_gradients
        return [fft(gout)]


fft_op, ifft_op = FFTOp(), IFFTOp()

cufft_op, cuifft_op = FFTOp(), IFFTOp()


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


"""
class IFFTOp(Op):
    __props__ = ("inverse",)

    def __init__(self, inverse=False):
        super(FFT, self).__init__()
        self.inverse = inverse

    def make_node(self, frames):
        _frames = tensor.as_tensor(frames, ndim=3)
        _n = tensor.as_tensor(_frames.shape[1], ndim=0)

        return theano.gof.Apply(self, [frames], [tensor.tensor3()])

    def perform(self, node, input, output_storage):
        frames, = input
        if self.inverse:
            fft_fn = scipy.fftpack.ifft
        else:
            fft_fn = scipy.fftpack.fft

        frames_dtype = frames.dtype
        if frames.dtype == np.float32 or frames_dtype == np.int32:
            complex_dtype = np.complex64
        elif frames.dtype == np.float64:
            complex_dtype = np.complex128

        fft = fft_fn(np.squeeze(frames.view(complex_dtype)), frames.shape[1])
        if self.inverse is True:
            fft *= np.sqrt(frames.shape[1])
        else:
            fft /= np.sqrt(frames.shape[1])

        out, = output_storage
        if frames_dtype == np.int32:
            frames_dtype = np.float32
        elif frames_dtype == np.int64:
            frames_dtype = np.float64
        out[0] = fft.view(frames_dtype).reshape(fft.shape + (2,)).copy()

    def grad(self, input, output_gradients):
        non_inverse = not self.inverse
        if self.inverse is True:
            return fft(output_gradients[0])
        else:
            return ifft(output_gradients[0])
        #return [FFT(inverse=non_inverse)(output_gradients[0])]


fft, ifft = FFT(inverse=False), FFT(inverse=True)

cufft, cuifft = FFT(inverse=False), FFT(inverse=True)
"""

"""
import theano

from theano import tensor as T

from theano.tensor.fft import rfft, irfft

def cufft(complex_input):
    out = T.zeros_like(complex_input)
    real, imag = complex_input[:, :, 0], complex_input[:, :, 1]
    freal, fimag = rfft(real), rfft(imag)


    fres = T.stack([freal[:, :, 0] - fimag[:, :, 1], freal[:, :, 1] + fimag[:, :, 0]], axis=1)

    fres_extended = T.zeros(out.shape)

    n, m = complex_input.shape[1], fres.shape[1]

    T.set_subtensor(out[:, :m], fres[:, :, 0])
    T.set_subtensor(out[:, n - m:], fres[:, ::-1, 0])

    T.inc_subtensor(out[:, :m], fres[:, :, 1])
    T.inc_subtensor(out[:, n - m:], -fres[:, ::-1, 1])

    out = T.stack([rfft(real), rfft(imag)], axis=-1)
    return out

def cuifft(complex_input):
    real, imag = complex_input[:, :, 0], complex_input[:, :, 1]
    out = T.stack([irfft(real), irfft(imag)], axis=-1)
    return out
"""