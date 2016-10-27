import theano
import lasagne
import numpy as np

import theano.tensor as T
from manifolds import UnitaryKron


from utils.theano_complex_extension import apply_complex_mat_to_kronecker


class UnitaryKronLayer(lasagne.layers.Layer):
    def __init__(self, incoming, partition, **kwargs):
        super(UnitaryKronLayer, self).__init__(incoming, **kwargs)
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_inputs = num_inputs // 2
        self.n_hidden = num_inputs // 2
        self.shape = (self.n_hidden, self.n_hidden)
        self.partition = partition
        assert(np.prod(partition) == self.n_hidden)
        self.manifold = UnitaryKron(partition)

        U = self.manifold.rand_np()
        basename = kwargs.get('name', '')

        attr_names = ["U" + str(i) for i in range(len(partition))]
        unique_ids = [man.str_id for man in self.manifold._manifolds]
        for attr_name, unique_id, Ui, dimsize in zip(attr_names, unique_ids, U, partition):
            added_param = self.add_param(Ui, (2, dimsize, dimsize), name=basename + "U" + unique_id, regularizable=False)
            setattr(self, attr_name, added_param)
        #self.U = self.add_param(U, (2, self.n_hidden, self.n_hidden), name=basename + "U", regularizable=False)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        unitary_input = T.reshape(input, (input.shape[0], 2, self.num_inputs))
        unitary_input = T.transpose(unitary_input, (1, 0, 2))
        U = tuple(getattr(self, "U" + str(i)) for i in range(len(self.partition)))
        output = apply_complex_mat_to_kronecker(unitary_input, U)
        output = output.reshape((input.shape[0], -1))
        return output

    def get_output_shape_for(self, input_shape):
        if len(input_shape) > 2:
            return (input_shape[0], int(np.prod(input_shape[1:])))
        return input_shape

