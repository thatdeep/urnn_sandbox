import theano
import numpy as np

from theano import tensor as T
from .utils import initialize_matrix, initialize_data_nodes, compute_cost_t,\
    times_diag, times_reflection, vec_permutation, do_fft, do_ifft

NP_FLOAT = np.float64
INT_STR = 'int64'
FLOAT_STR = 'float64'


def complexRNN(n_input, n_hidden, n_output, input_type='real', out_every_t=False, loss_function='CE'):

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

    reflection = initialize_matrix(2, 2*n_hidden, 'reflection', rng)
    out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX), name='out_bias')
    theta = theano.shared(np.asarray(rng.uniform(low=-np.pi,
                                                 high=np.pi,
                                                 size=(3, n_hidden)),
                                     dtype=theano.config.floatX),
                                name='theta')

    bucket = np.sqrt(3. / 2 / n_hidden)
    h_0 = theano.shared(np.asarray(rng.uniform(low=-bucket,
                                               high=bucket,
                                               size=(1, 2 * n_hidden)),
                                   dtype=theano.config.floatX),
                        name='h_0')

    parameters = [V, U, hidden_bias, reflection, out_bias, theta, h_0]

    x, y = initialize_data_nodes(loss_function, input_type, out_every_t)

    index_permute = np.random.permutation(n_hidden)

    index_permute_long = np.concatenate((index_permute, index_permute + n_hidden))
    swap_re_im = np.concatenate((np.arange(n_hidden, 2*n_hidden), np.arange(n_hidden)))

    # define the recurrence used by theano.scan
    def recurrence(x_t, y_t, h_prev, cost_prev, acc_prev, theta, V, hidden_bias, out_bias, U):

        # Compute hidden linear transform
        step1 = times_diag(h_prev, n_hidden, theta[0,:], swap_re_im)
        step2 = do_fft(step1, n_hidden)
        step3 = times_reflection(step2, n_hidden, reflection[0,:])
        step4 = vec_permutation(step3, index_permute_long)
        step5 = times_diag(step4, n_hidden, theta[1,:], swap_re_im)
        step6 = do_ifft(step5, n_hidden)
        step7 = times_reflection(step6, n_hidden, reflection[1,:])
        step8 = times_diag(step7, n_hidden, theta[2,:], swap_re_im)     
        
        hidden_lin_output = step8
        
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
    non_sequences = [theta, V, hidden_bias, out_bias, U]
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

    return [x, y], parameters, costs