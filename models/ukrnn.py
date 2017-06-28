import theano
import numpy as np

from manifolds import UnitaryKron
from theano import tensor as T


from .utils import initialize_matrix, initialize_data_nodes, compute_cost_t, unitary_kron_transform

NP_FLOAT = np.float64
INT_STR = 'int64'
FLOAT_STR = 'float64'


def UKRNN(n_input, n_hidden, partition, n_output, input_type='real', out_every_t=False, loss_function='CE'):

    np.random.seed(1234)
    rng = np.random.RandomState(1234)

    # Initialize parameters: theta, V_re, V_im, hidden_bias, U, out_bias, h_0
    V = initialize_matrix(n_input, 2*n_hidden, 'V', rng)
    U = initialize_matrix(2 * n_hidden, n_output, 'U', rng)
    hidden_bias = theano.shared(np.asarray(rng.uniform(low=-0.01,
                                                       high=0.01,
                                                       size=(n_hidden,)),
                                           dtype=theano.config.floatX),
                                name='hidden_bias')
    kron_manifold = UnitaryKron(partition)

    MANIFOLD_NAMES = [manifold.str_id for manifold in kron_manifold._manifolds]
    UK = [theano.shared(value=manifold.rand_np(), name=manifold.str_id) for manifold in kron_manifold._manifolds]
    manifolds = {manifold.str_id: manifold for manifold in kron_manifold._manifolds}


    out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX), name='out_bias')

    bucket = np.sqrt(3. / 2 / n_hidden)
    h_0 = theano.shared(np.asarray(rng.uniform(low=-bucket,
                                               high=bucket,
                                               size=(1, 2 * n_hidden)),
                                   dtype=theano.config.floatX),
                        name='h_0')

    parameters = [V, U, hidden_bias] + UK + [out_bias, h_0]

    x, y = initialize_data_nodes(loss_function, input_type, out_every_t)


    swap_re_im = np.concatenate((np.arange(n_hidden, 2*n_hidden), np.arange(n_hidden)))

    # define the recurrence used by theano.scan
    def recurrence(x_t, y_t, h_prev, cost_prev, acc_prev, V, hidden_bias, out_bias, U, *UK):
        #unitary_step = unitary_transform(h_prev, n_hidden, unitary_matrix)
        unitary_step = unitary_kron_transform(h_prev, n_hidden, UK)

        hidden_lin_output = unitary_step

        # Compute data linear transform
        if loss_function == 'CE':
            data_lin_output = V[T.cast(x_t, INT_STR)]
        else:
            data_lin_output = T.dot(x_t, V)

        # Total linear output
        lin_output = hidden_lin_output + data_lin_output


        # Apply non-linearity ----------------------------

        # scale RELU nonlinearity
        modulus = T.sqrt(lin_output**2 + lin_output[:, swap_re_im]**2)
        rescale = T.maximum(modulus + T.tile(hidden_bias, [2]).dimshuffle('x', 0), 0.) / (modulus + 1e-5)
        h_t = lin_output * rescale

        if out_every_t:
            lin_output = T.dot(h_t, U) + out_bias.dimshuffle('x', 0)
            cost_t, acc_t = compute_cost_t(lin_output, loss_function, y_t)
        else:
            cost_t = theano.shared(NP_FLOAT(0.0))
            acc_t = theano.shared(NP_FLOAT(0.0))

        return h_t, cost_t, acc_t

    # compute hidden states
    h_0_batch = T.tile(h_0, [x.shape[1], 1])
    non_sequences = [V, hidden_bias, out_bias, U] + UK
    if out_every_t:
        sequences = [x, y]
    else:
        sequences = [x, T.tile(theano.shared(np.zeros((1,1), dtype=theano.config.floatX)), [x.shape[0], 1, 1])]

    outputs_info=[h_0_batch, theano.shared(NP_FLOAT(0.0)), theano.shared(NP_FLOAT(0.0))]

    [hidden_states, cost_steps, acc_steps], updates = theano.scan(fn=recurrence,
                                                                  sequences=sequences,
                                                                  non_sequences=non_sequences,
                                                                  outputs_info=outputs_info)

    if not out_every_t:
        lin_output = T.dot(hidden_states[-1,:,:], U) + out_bias.dimshuffle('x', 0)
        costs = compute_cost_t(lin_output, loss_function, y)
    else:
        cost = cost_steps.mean()
        accuracy = acc_steps.mean()
        costs = [cost, accuracy]

    return [x, y], parameters, costs, manifolds