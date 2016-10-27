import numpy as np

import theano
import lasagne
from theano import tensor as T
from utils.fftconv import cufft, cuifft


def initialize_matrix(n_in, n_out, name, rng):
    bin = np.sqrt(6. / (n_in + n_out))
    values = np.asarray(rng.uniform(low=-bin,
                                    high=bin,
                                    size=(n_in, n_out)),
                                    dtype=theano.config.floatX)
    return values


def do_fft(input, n_hidden):
    fft_input = T.reshape(input, (input.shape[0], 2, n_hidden))
    fft_input = fft_input.dimshuffle(0,2,1)
    fft_output = cufft(fft_input) * T.sqrt(n_hidden)
    fft_output = fft_output.dimshuffle(0,2,1)
    output = T.reshape(fft_output, (input.shape[0], 2*n_hidden))
    return output

def do_ifft(input, n_hidden):
    ifft_input = T.reshape(input, (input.shape[0], 2, n_hidden))
    ifft_input = ifft_input.dimshuffle(0,2,1)
    ifft_output = cuifft(ifft_input) / T.sqrt(n_hidden)
    ifft_output = ifft_output.dimshuffle(0,2,1)
    output = T.reshape(ifft_output, (input.shape[0], 2*n_hidden))
    return output


def times_diag(input, n_hidden, diag, swap_re_im):
    d = T.concatenate([diag, -diag])

    Re = T.cos(d).dimshuffle('x',0)
    Im = T.sin(d).dimshuffle('x',0)

    input_times_Re = input * Re
    input_times_Im = input * Im

    output = input_times_Re + input_times_Im[:, swap_re_im]

    return output


def vec_permutation(input, index_permute):
    return input[:, index_permute]


def times_reflection(input, n_hidden, reflection):
    input_re = input[:, :n_hidden]
    input_im = input[:, n_hidden:]
    reflect_re = reflection[:n_hidden]
    reflect_im = reflection[n_hidden:]

    vstarv = (reflection**2).sum()

    input_re_reflect_re = T.dot(input_re, reflect_re)
    input_re_reflect_im = T.dot(input_re, reflect_im)
    input_im_reflect_re = T.dot(input_im, reflect_re)
    input_im_reflect_im = T.dot(input_im, reflect_im)

    a = T.outer(input_re_reflect_re - input_im_reflect_im, reflect_re)
    b = T.outer(input_re_reflect_im + input_im_reflect_re, reflect_im)
    c = T.outer(input_re_reflect_re - input_im_reflect_im, reflect_im)
    d = T.outer(input_re_reflect_im + input_im_reflect_re, reflect_re)

    output = input
    output = T.inc_subtensor(output[:, :n_hidden], - 2. / vstarv * (a + b))
    output = T.inc_subtensor(output[:, n_hidden:], - 2. / vstarv * (d - c))

    return output



class ComplexLayer(lasagne.layers.Layer):
    def __init__(self, incoming, random_state=1234, **kwargs):
        super(ComplexLayer, self).__init__(incoming, **kwargs)
        np.random.seed(random_state)
        self.rng = np.random.RandomState(random_state)
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_inputs = num_inputs // 2
        self.n_hidden = num_inputs // 2
        self.shape = (self.n_hidden, self.n_hidden)

        reflection = initialize_matrix(2, 2*self.n_hidden, 'reflection', self.rng)
        self.reflection = self.add_param(reflection, reflection.shape, 'reflection')
        self.theta = self.add_param(np.asarray(self.rng.uniform(low=-np.pi,
                                                     high=np.pi,
                                                     size=(3, self.n_hidden))),
                                    shape=(3, self.n_hidden),
                                    name='theta')

        index_permute = np.random.permutation(self.n_hidden)

        self.index_permute_long = np.concatenate((index_permute, index_permute + self.n_hidden))
        self.swap_re_im = np.concatenate((np.arange(self.n_hidden, 2*self.n_hidden), np.arange(self.n_hidden)))

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        step1 = times_diag(input, self.n_hidden, self.theta[0,:], self.swap_re_im)
        step2 = do_fft(step1, self.n_hidden)
        step3 = times_reflection(step2, self.n_hidden, self.reflection[0,:])
        step4 = vec_permutation(step3, self.index_permute_long)
        step5 = times_diag(step4, self.n_hidden, self.theta[1,:], self.swap_re_im)
        step6 = do_ifft(step5, self.n_hidden)
        step7 = times_reflection(step6, self.n_hidden, self.reflection[1,:])
        step8 = times_diag(step7, self.n_hidden, self.theta[2,:], self.swap_re_im)
        output = step8
        return output

    def get_output_shape_for(self, input_shape):
        if len(input_shape) > 2:
            return (input_shape[0], int(np.prod(input_shape[1:])))
        return input_shape
