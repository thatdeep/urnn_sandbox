import theano
import lasagne
import numpy as np

import theano.tensor as T
from manifolds import Unitary


class UnitaryLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(UnitaryLayer, self).__init__(incoming, **kwargs)
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_inputs = num_inputs // 2
        self.n_hidden = num_inputs // 2
        self.shape = (self.n_hidden, self.n_hidden)
        self.manifold = Unitary(self.n_hidden)

        U = self.manifold.rand_np()
        basename = kwargs.get('name', '')
        self.U = self.add_param(U, (2, self.n_hidden, self.n_hidden), name=basename + "U" + self.manifold.str_id, regularizable=False)

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