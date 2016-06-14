import theano
import lasagne
import numpy as np

import theano.tensor as T

# works only for matrix input now
class ComplexifyTransform(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        shape = input.shape
        output = T.stack([input, T.zeros_like(input)], 1)#.reshape((shape[0], -1))
        return output