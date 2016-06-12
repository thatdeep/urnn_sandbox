import theano
import lasagne
import numpy as np

import theano.tensor as T
from manifolds import Unitary


class UnitaryLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, rank, params=None, **kwargs):
        super(UnitaryLayer, self).__init__(incoming, **kwargs)
        num_inputs = int(np.prod(self.input_shape[1:]))
        if num_inputs != num_units:
            print("input dimension isn't equal units dimension. Cannot build square matrix!")
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.shape = (self.num_inputs, self.num_units)

        basename = kwargs.get('name', '')

        self.manifold = Unitary(self.num_inputs)
        if params:
            UR, UI = params
        else:
            UR, UI = self.manifold.rand_np()
        # give proper names
        self.WR = self.add_param(UR, (self.num_inputs, self.num_units), name=basename + "WR", regularizable=False)
        self.WR = self.add_param(UI, (self.num_inputs, self.num_units), name=basename + "WI", regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        return input