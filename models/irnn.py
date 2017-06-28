import theano
import numpy as np

from theano import tensor as T
from .utils import initialize_matrix, initialize_data_nodes, compute_cost_t

NP_FLOAT = np.float64
INT_STR = 'int64'
FLOAT_STR = 'float64'


def IRNN(n_input, n_hidden, n_output, input_type='real', out_every_t=False, loss_function='CE'):
    np.random.seed(1234)
    rng = np.random.RandomState(1234)

    x, y = initialize_data_nodes(loss_function, input_type, out_every_t)
    inputs = [x, y]

    h_0 = theano.shared(np.zeros((1, n_hidden), dtype=theano.config.floatX))
    V = initialize_matrix(n_input, n_hidden, 'V', rng)
    W = theano.shared(np.identity(n_hidden, dtype=theano.config.floatX))
    out_mat = initialize_matrix(n_hidden, n_output, 'out_mat', rng)
    hidden_bias = theano.shared(np.zeros((n_hidden,), dtype=theano.config.floatX))
    out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX))

    parameters = [h_0, V, W, out_mat, hidden_bias, out_bias]

    def recurrence(x_t, y_t, h_prev, cost_prev, acc_prev, V, W, hidden_bias, out_mat, out_bias):
        if loss_function == 'CE':
            data_lin_output = V[x_t]
        else:
            data_lin_output = T.dot(x_t, V)

        h_t = T.nnet.relu(T.dot(h_prev, W) + data_lin_output + hidden_bias.dimshuffle('x', 0))
        if out_every_t:
            lin_output = T.dot(h_t, out_mat) + out_bias.dimshuffle('x', 0)
            cost_t, acc_t = compute_cost_t(lin_output, loss_function, y_t)
        else:
            cost_t = theano.shared(NP_FLOAT(0.0))
            acc_t = theano.shared(NP_FLOAT(0.0))

        return h_t, cost_t, acc_t

    non_sequences = [V, W, hidden_bias, out_mat, out_bias]

    h_0_batch = T.tile(h_0, [x.shape[1], 1])

    if out_every_t:
        sequences = [x, y]
    else:
        sequences = [x, T.tile(theano.shared(np.zeros((1,1), dtype=theano.config.floatX)), [x.shape[0], 1, 1])]

    outputs_info = [h_0_batch, theano.shared(NP_FLOAT(0.0)), theano.shared(NP_FLOAT(0.0))]

    [hidden_states, cost_steps, acc_steps], updates = theano.scan(fn=recurrence,
                                                                  sequences=sequences,
                                                                  non_sequences=non_sequences,
                                                                  outputs_info = outputs_info)

    if not out_every_t:
        lin_output = T.dot(hidden_states[-1,:,:], out_mat) + out_bias.dimshuffle('x', 0)
        costs = compute_cost_t(lin_output, loss_function, y)
    else:
        cost = cost_steps.mean()
        accuracy = acc_steps.mean()
        costs = [cost, accuracy]

    return inputs, parameters, costs