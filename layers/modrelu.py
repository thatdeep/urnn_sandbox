import theano
import lasagne
import numpy as np

import theano.tensor as T


class Modrelu(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(Modrelu, self).__init__(incoming, **kwargs)
        num_inputs = int(np.prod(self.input_shape[1:]))//2
        self.b = self.add_param(lasagne.init.Constant(0.), (num_inputs,), name="b",
                                regularizable=False)

    def modrelu(self, z, b):
        zr, zi = z[:, :z.shape[1]//2], z[:, z.shape[0]//2:]
        norms = T.sqrt(zr**2 + zi**2)
        vals = norms + b
        mask = T.gt(vals, 0)
        mask = T.stack([mask, mask], axis=2).reshape(z.shape)
        res = z * mask * vals / (norms + 1e-8)
        return res

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        return self.modrelu(input, self.b)


