import theano
import lasagne
import numpy as np

import theano.tensor as T
from numpy import random as rnd, linalg as la

from layers import UnitaryLayer, UnitaryKronLayer, RecurrentUnitaryLayer, ComplexLayer, WTTLayer, Modrelu
from matplotlib import pyplot as plt
from utils.optimizations import nesterov_momentum, custom_sgd
from lasagne.nonlinearities import rectify

np.set_printoptions(linewidth=200, suppress=True)

#theano.config.exception_verbosity='high'
#theano.config.mode='DebugMode'
#theano.config.optimizer='None'


# Min/max sequence length
MIN_LENGTH = 50
MAX_LENGTH = 51
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 81
# Number of training sequences in each batch
N_BATCH = 100
# Optimization learning rate
LEARNING_RATE = 3 * 1e-3
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
EPOCH_SIZE = 100
# Number of epochs to train the net
NUM_EPOCHS = 600
# Exact sequence length
TIME_SEQUENCES=100



def gen_data(min_length=MIN_LENGTH, max_length=MAX_LENGTH, n_batch=N_BATCH):
    '''
    Generate a batch of sequences for the "add" task, e.g. the target for the
    following

    ``| 0.5 | 0.7 | 0.3 | 0.1 | 0.2 | ... | 0.5 | 0.9 | ... | 0.8 | 0.2 |
      |  0  |  0  |  1  |  0  |  0  |     |  0  |  1  |     |  0  |  0  |``

    would be 0.3 + .9 = 1.2.  This task was proposed in [1]_ and explored in
    e.g. [2]_.

    Parameters
    ----------
    min_length : int
        Minimum sequence length.
    max_length : int
        Maximum sequence length.
    n_batch : int
        Number of samples in the batch.

    Returns
    -------
    X : np.ndarray
        Input to the network, of shape (n_batch, max_length, 2), where the last
        dimension corresponds to the two sequences shown above.
    y : np.ndarray
        Correct output for each sample, shape (n_batch,).
    mask : np.ndarray
        A binary matrix of shape (n_batch, max_length) where ``mask[i, j] = 1``
        when ``j <= (length of sequence i)`` and ``mask[i, j] = 0`` when ``j >
        (length of sequence i)``.

    References
    ----------
    .. [1] Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory."
    Neural computation 9.8 (1997): 1735-1780.

    .. [2] Sutskever, Ilya, et al. "On the importance of initialization and
    momentum in deep learning." Proceedings of the 30th international
    conference on machine learning (ICML-13). 2013.
    '''
    # Generate X - we'll fill the last dimension later
    X = np.concatenate([np.random.uniform(size=(n_batch, max_length, 1)),
                        np.zeros((n_batch, max_length, 1))],
                       axis=-1)
    mask = np.zeros((n_batch, max_length), dtype='int32')
    y = np.zeros((n_batch,))
    # Compute masks and correct values
    for n in range(n_batch):
        # Randomly choose the sequence length
        length = np.random.randint(min_length, max_length)
        # Make the mask for this sample 1 within the range of length
        mask[n, :length] = 1
        # Zero out X after the end of the sequence
        X[n, length:, 0] = 0
        # Set the second dimension to 1 at the indices to add
        X[n, np.random.randint(length/10), 1] = 1
        X[n, np.random.randint(length/2, length), 1] = 1
        # Multiply and sum the dimensions of X to get the target value
        y[n] = np.sum(X[n, :, 0]*X[n, :, 1])
    # Center the inputs and outputs
    X -= X.reshape(-1, 2).mean(axis=0)
    y -= y.mean()
    return (X.astype(theano.config.floatX), y.astype(theano.config.floatX),
            mask.astype('int32'))


if __name__ == "__main__":
    print("Building network ...")
    N_INPUT=2
    learning_rate = theano.shared(np.array(LEARNING_RATE, dtype=theano.config.floatX))


    # input layer of shape (n_batch, n_timestems, n_input)
    l_in = lasagne.layers.InputLayer(shape=(None, MAX_LENGTH, N_INPUT))

    # mask of shape (n_batch, n_timesteps)
    l_mask = lasagne.layers.InputLayer(shape=(None, MAX_LENGTH),input_var=T.imatrix("mask"))

    # define input-to-hidden and hidden-to-hidden linear transformations
    l_in_hid = lasagne.layers.DenseLayer(lasagne.layers.InputLayer((None, N_INPUT)), N_HIDDEN * 2)
    #l_hid_hid = ComplexLayer(lasagne.layers.InputLayer((None, N_HIDDEN * 2)))
    #l_hid_hid = UnitaryLayer(lasagne.layers.InputLayer((None, N_HIDDEN * 2)))
    #manifolds = {}


    l_hid_hid = WTTLayer(lasagne.layers.InputLayer((None, N_HIDDEN * 2)), [3]*4, [2]*3)

    manifold = l_hid_hid.manifold
    if not isinstance(manifold, list):
        manifold = [manifold]
    manifolds = {man.str_id: man for man in manifold}


    #manifolds = {}

    # recurrent layer using linearities defined above
    l_rec = RecurrentUnitaryLayer(l_in, l_in_hid, l_hid_hid, nonlinearity=ModRelu(lasagne.layers.InputLayer((None, N_HIDDEN * 2))),
                                                mask_input=l_mask, only_return_final=True)
    print(lasagne.layers.get_output_shape(l_rec))

    # nonlinearity for recurrent layer output
    #l_nonlin = ModRelu(l_rec)
    #print(lasagne.layers.get_output_shape(l_nonlin))


    l_reshape = lasagne.layers.ReshapeLayer(l_rec, (-1, N_HIDDEN * 2))
    print(lasagne.layers.get_output_shape(l_reshape))
    # Our output layer is a simple dense connection, with 1 output unit
    l_dense = lasagne.layers.DenseLayer(l_reshape, num_units=1, nonlinearity=None)
    l_out = lasagne.layers.ReshapeLayer(l_dense, (N_BATCH, -1))
    print(lasagne.layers.get_output_shape(l_out))

    target_values = T.vector('target_output')

    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(l_out)

    predicted_values = network_output.flatten()
    # Our cost will be mean-squared error
    cost = T.mean((predicted_values - target_values)**2)
    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out,trainable=True)
    print(all_params)
    print(lasagne.layers.get_all_params(l_rec))
    # Compute SGD updates for training
    print("Computing updates ...")
    updates = custom_sgd(cost, all_params, LEARNING_RATE, manifolds)
    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values, l_mask.input_var],
                            cost, updates=updates, on_unused_input='warn')
    compute_cost = theano.function(
        [l_in.input_var, target_values, l_mask.input_var], cost, on_unused_input='warn')

    # We'll use this "validation set" to periodically check progress
    X_val, y_val, mask_val = gen_data(n_batch=100)

    #TEST
    #ll = lasagne.layers.InputLayer((None, N_HIDDEN * 2))
    #v = ModRelu(ll)
    #v_out =lasagne.layers.get_output(v)
    #print(T.grad(v_out.mean(),ll.input_var).eval({ll.input_var: np.zeros([5,N_HIDDEN*2])})) #with ones its okay
    #TEST

    try:
        for epoch in range(NUM_EPOCHS):
            if (epoch + 1) % 100 == 0:
                learning_rate.set_value(learning_rate.get_value() * 0.9)
            cost_val = compute_cost(X_val, y_val, mask_val)
            for _ in range(EPOCH_SIZE):
                X, y, m = gen_data()
                train(X, y, m.astype('int32'))
            print("Epoch {} validation cost = {}".format(epoch, cost_val))
    except KeyboardInterrupt:
        pass
