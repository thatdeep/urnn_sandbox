import theano
import lasagne
import numpy as np

import theano.tensor as T


class ModRelu(lasagne.layers.Layer):
    def __init__(self, incoming, b=lasagne.init.Uniform(range=0.01), **kwargs):
        super(ModRelu, self).__init__(incoming, **kwargs)
        print(self.input_shape)
        self.n_hidden = self.input_shape[-1] // 2
        self.hb = self.add_param(b, (self.n_hidden,), name='hb', regularizable=False, trainable=True)

    def get_output_for(self, input, **kwargs):
        eps = 1e-5
        print("Inside a ModReLU")
        input_flattened = input.reshape((-1, self.n_hidden*2))

        swap_re_im = np.concatenate((np.arange(self.n_hidden, 2*self.n_hidden), np.arange(self.n_hidden)))
        modulus = T.sqrt(input_flattened**2 + input_flattened[:, swap_re_im]**2 + eps)
        rescale = T.maximum(modulus + T.tile(self.hb, [2]).dimshuffle('x', 0), 0.) / (modulus + 1e-5)
        out = (input_flattened * rescale).reshape(input.shape)
        return out