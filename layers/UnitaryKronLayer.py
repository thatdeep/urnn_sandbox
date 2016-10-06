import theano
import lasagne
import numpy as np

import theano.tensor as T
from manifolds import UnitaryKron


def apply_mat_to_kronecker(x, matrices):
    x = x.reshape((x.shape[0],) + tuple(mat.shape[0] for mat in matrices))
    result = x
    for mat in matrices:
        result = T.tensordot(result, mat, axes=([1], [0]))
    return result.reshape((x.shape[0], -1))


def complex_tensordot(a, b, axes=2):
 if type(axes) in _numberTypes: return dot(a.reshape_2d(a.ndim-axes), b.reshape_2d(axes)).reshape(a.shape[:a.ndim-axes] + b.shape[axes:])
 assert len(axes)==2 and len(axes[0])==len(axes[1]), 'the axes parameter to gnumpy.tensordot looks bad'
 aRemove, bRemove = (tuple(axes[0]), tuple(axes[1]))
 return complex_tensordot(a.transpose(filter(lambda x: x not in aRemove, tuple(range(a.ndim))) + aRemove),
                          b.transpose(bRemove + filter(lambda x: x not in bRemove, tuple(range(b.ndim)))),
                          len(aRemove))


class UnitaryLayer(lasagne.layers.Layer):
    def __init__(self, incoming, partition, **kwargs):
        super(UnitaryLayer, self).__init__(incoming, **kwargs)
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.n_inputs = num_inputs
        self.n_hidden = num_inputs // 2
        self.shape = (self.n_hidden, self.n_hidden)
        self.partition = partition
        assert(np.prod(partition) == self.n_hidden)
        self.manifold = UnitaryKron(partition)

        U = self.manifold.rand_np()
        basename = kwargs.get('name', '')
        self.U = self.add_param(U, (2, self.n_hidden, self.n_hidden), name=basename + "U", regularizable=False)

    def get_output_for(self, input, **kwargs):
        UR, UI = self.manifold.frac(self.U)
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        unitary_input = T.reshape(input, (input.shape[0], 2, self.num_inputs))
        IR, II = unitary_input[:, 0, :], unitary_input[:, 1, :]
        output = T.stack([IR.dot(UR) - II.dot(UI), IR.dot(UR) + II.dot(UR)], axis=1)
        output = output.reshape((input.shape[0], -1))
        return output

    def get_output_shape_for(self, input_shape):
        if len(input_shape) > 2:
            return (input_shape[0], int(np.prod(input_shape[1:])))
        return input_shape

