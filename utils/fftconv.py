
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

    def make_node(self, a, n=None):
        a = tensor.as_tensor_variable(a)
        if a.ndim < 3:
            raise TypeError('%s: input must have dimension >= 3,  with ' %
                            self.__class__.__name__ +
                            'first dimension batches and last real/imag parts')
        if n is None:
            n = a.shape[1]
            n = tensor.as_tensor_variable(n)
        else:
            n = tensor.as_tensor_variable(n)
            if (not n.dtype.startswith('int')) and \
               (not n.dtype.startswith('uint')):
                raise TypeError('%s: length of the transformed axis must be'
                                ' of type integer' % self.__class__.__name__)
        return gof.Apply(self, [a, n], [self.output_type(a)()])

    def perform(self, node, inputs, output_storage):
        a = inputs[0]
        n = inputs[1]


        frames_dtype = a.dtype
        if a.dtype == np.float32 or frames_dtype == np.int32:
            complex_dtype = np.complex64
        elif a.dtype == np.float64:
            complex_dtype = np.complex128

        out = np.fft.fft(np.squeeze(a.view(complex_dtype)), n=int(n))
        if frames_dtype == np.int32:
            frames_dtype = np.float32
        elif frames_dtype == np.int64:
            frames_dtype = np.float64
        output_storage[0][0] = out.view(frames_dtype).reshape(out.shape + (2,)).astype(frames_dtype)

    def grad(self, inputs, output_gradients):
        gout, = output_gradients
        n = inputs[1]
        return [ifft(gout, n), DisconnectedType()()]
        #return [FFT(inverse=non_inverse)(output_gradients[0])]

    def connection_pattern(self, node):
        # Specify that shape input parameter has no connection to graph and gradients.
        return [[True], [False]]


class IFFTOp(gof.Op):
    __props__ = ()

    def output_type(self, inp):
        # expect complex valued (last axis of size 2) input, returns tensor of same size
        return tensor.TensorType(inp.dtype, broadcastable=[False] * inp.type.ndim)

    def make_node(self, a, n=None):
        a = tensor.as_tensor_variable(a)
        if a.ndim < 3:
            raise TypeError('%s: input must have dimension >= 3,  with ' %
                            self.__class__.__name__ +
                            'first dimension batches and last real/imag parts')
        if n is None:
            n = a.shape[1]
            n = tensor.as_tensor_variable(n)
        else:
            n = tensor.as_tensor_variable(n)
            if (not n.dtype.startswith('int')) and \
               (not n.dtype.startswith('uint')):
                raise TypeError('%s: length of the transformed axis must be'
                                ' of type integer' % self.__class__.__name__)
        return gof.Apply(self, [a, n], [self.output_type(a)()])

    def perform(self, node, inputs, output_storage):
        a = inputs[0]
        n = inputs[1]


        frames_dtype = a.dtype
        if a.dtype == np.float32 or frames_dtype == np.int32:
            complex_dtype = np.complex64
        elif a.dtype == np.float64:
            complex_dtype = np.complex128

        out = np.fft.ifft(np.squeeze(a.view(complex_dtype)), n=int(n))
        if frames_dtype == np.int32:
            frames_dtype = np.float32
        elif frames_dtype == np.int64:
            frames_dtype = np.float64
        output_storage[0][0] = out.view(frames_dtype).reshape(out.shape + (2,)).astype(frames_dtype)

    def grad(self, inputs, output_gradients):
        gout, = output_gradients
        n = inputs[1]
        return [fft(gout, n), DisconnectedType()()]

    def connection_pattern(self, node):
        # Specify that shape input parameter has no connection to graph and gradients.
        return [[True], [False]]


fft, ifft = FFTOp(), IFFTOp()

cufft, cuifft = FFTOp(), IFFTOp()

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