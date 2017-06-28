import theano
import numpy as np

from theano import tensor as T
from utils.fftconv import cufft, cuifft
from utils.theano_complex_extension import apply_complex_mat_to_kronecker


NP_FLOAT = np.float64
INT_STR = 'int64'
FLOAT_STR = 'float64'


def initialize_data_nodes(loss_function, input_type, out_every_t):
    x = T.tensor3() if input_type == 'real' else T.matrix(dtype=INT_STR)
    if loss_function == 'CE':
        y = T.matrix(dtype=INT_STR) if out_every_t else T.vector(dtype=INT_STR)
    else:
        y = T.tensor3() if out_every_t else T.matrix()
    return x, y


def initialize_matrix(n_in, n_out, name, rng):
    bin = np.sqrt(6. / (n_in + n_out))
    values = np.asarray(rng.uniform(low=-bin,
                                    high=bin,
                                    size=(n_in, n_out)),
                                    dtype=theano.config.floatX)
    return theano.shared(value=values, name=name)


def compute_cost_t(lin_output, loss_function, y_t):
    if loss_function == 'CE':
        RNN_output = T.nnet.softmax(lin_output)
        cost_t = T.nnet.categorical_crossentropy(RNN_output, y_t).mean()
        acc_t =(T.eq(T.argmax(RNN_output, axis=-1), y_t)).mean(dtype=theano.config.floatX)
    elif loss_function == 'MSE':
        cost_t = ((lin_output - y_t)**2).mean()
        acc_t = theano.shared(NP_FLOAT(0.0))

    return cost_t, acc_t


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


def unitary_transform(input, n_hidden, U):
    UR, UI = U[0, :, :], U[1, :, :]
    unitary_input = T.reshape(input, (input.shape[0], 2, n_hidden))
    IR, II = unitary_input[:, 0, :], unitary_input[:, 1, :]
    output = T.stack([IR.dot(UR) - II.dot(UI), IR.dot(UI) + II.dot(UR)], axis=1)
    output = T.reshape(output, (input.shape[0], 2*n_hidden))
    return output


def unitary_kron_transform(input, n_hidden, U):
    unitary_input = T.reshape(input, (input.shape[0], 2, n_hidden))
    unitary_input = T.transpose(unitary_input, (1, 0, 2))
    output = apply_complex_mat_to_kronecker(unitary_input, U)
    output = output.reshape((input.shape[0], -1))
    return output