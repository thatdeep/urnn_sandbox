import theano
import lasagne
import numpy as np

import theano.tensor as T
from manifolds import Unitary


class UnitaryLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(UnitaryLayer, self).__init__(incoming, **kwargs)
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_inputs = num_inputs
        self.num_units = num_inputs // 2
        self.shape = (self.num_inputs, self.num_inputs)
        self.manifold = Unitary(self.num_inputs)

        U = self.manifold.rand_np()
        basename = kwargs.get('name', '')
        self.U = self.add_param(U, (2, self.num_inputs, self.num_units), name=basename + "U", regularizable=False)

    def get_output_for(self, input, **kwargs):
        UR, UI = self.manifold.frac(self.U)
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        IR, II = input[:, :input.shape[1]//2], input[:, input.shape[1]//2:]
        return T.stack([IR.dot(UR) - II.dot(UI), IR.dot(UR) + II.dot(UI)]).reshape(input.shape)