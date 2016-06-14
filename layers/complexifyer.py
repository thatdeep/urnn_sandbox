import theano
import lasagne
import numpy as np

import theano.tensor as T


class ComplexifyTransform(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(ComplexifyTransform, self).__init__(incoming, **kwargs)
        self.num_inputs = self.input_shape[1:]

    def get_output_for(self, input, **kwargs):
        output = T.stack([input, T.zeros_like(input)], 1)
        return output

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 2) + input_shape[1:]