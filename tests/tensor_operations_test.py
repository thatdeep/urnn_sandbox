import theano
import unittest
import numpy as np

from theano import tensor
from numpy import random as rnd


from utils.theano_complex_extension import complex_reshape, complex_tensordot, apply_mat_to_kronecker
from utils.theano_complex_extension import apply_complex_mat_to_kronecker, np_complex_tensordot
from utils.theano_complex_extension import np_apply_complex_mat_to_kronecker


class TensorDotFunction(unittest.TestCase):
    def test_complex_reshape(self):
        def get_all_prod_pairs(n):
            return [(i, n // i) for i in range(1, n+1) if n % i == 0]

        size = 2**3 * 3 * 4**2 * 5**2 * 6
        all_prod_pairs = get_all_prod_pairs(size)
        initial_shape = all_prod_pairs[len(all_prod_pairs) // 2]
        re = rnd.normal(size=initial_shape)
        im = rnd.normal(size=initial_shape)

        X = tensor.tensor3('X')
        x = np.stack([re, im])

        ethalon = re + 1j * im

        abstract_shape = tensor.ivector('shape')

        func = theano.function([X, abstract_shape], complex_reshape(X, abstract_shape, ndim=2))

        self.assertEqual(x.shape[1:], ethalon.shape)
        self.assertTrue(np.allclose(x[0, ...] + 1j * x[1, ...], ethalon))

        for next_shape in all_prod_pairs:
            x = func(x, next_shape)
            ethalon = np.reshape(ethalon, next_shape)

            self.assertEqual(x.shape[1:], ethalon.shape)
            self.assertTrue(np.allclose(x[0, ...] + 1j * x[1, ...], ethalon))


    def test_complex_tensordot(self):
        ms = (3, 5, 7, 9)
        ns = (4, 6, 8, 10)
        m = int(np.prod(ms))
        n = int(np.prod(ns))
        l = 100

        X = tensor.tensor3('X')
        Factors = [tensor.tensor3('factor_{}_{}'.format(mm, nn)) for (mm, nn) in zip(ms, ns)]

        x = rnd.normal(size=(2, l, m))
        factors = [rnd.normal(size=(2, mm, nn)) for (mm, nn) in zip(ms, ns)]

        x = np.reshape(x, (2, l,) + ms)
        X_reshaped = tensor.reshape(X, (2, l) + ms)
        def gen_index_dot(X, Factor, i):
            func = theano.function([X, Factor], complex_tensordot(X, Factor, axes=([i + 1], [0])))
            return func

        x_etha = x[0, ...] + 1j * x[1, ...]

        for i, factor in enumerate(factors):
            args = [x, factor]
            func = gen_index_dot(X_reshaped, Factors[i], i)
            computed = func(*args)
            computed_with_np = np_complex_tensordot(x, factor, axes=[[i+1], 0])
            ethalon = np.tensordot(x_etha, factor[0, ...] + 1j * factor[1, ...], axes=[[i + 1], [0]])
            self.assertEqual(computed.shape[1:], ethalon.shape)
            self.assertTrue(np.allclose(computed[0, ...] + 1j * computed[1, ...], ethalon))
            self.assertTrue(np.allclose(computed_with_np[0, ...] + 1j * computed_with_np[1, ...], ethalon))

    def test_apply_mat_to_kronecker(self):
        ms = (3, 5, 7, 9)
        ns = (4, 6, 8, 10)
        m = int(np.prod(ms))
        n = int(np.prod(ns))
        l = 100

        X = tensor.tensor3('X')
        Factors = [tensor.tensor3('factor_{}_{}'.format(mm, nn)) for (mm, nn) in zip(ms, ns)]

        x = rnd.normal(size=(2, l, m))
        factors = [rnd.normal(size=(2, mm, nn)) for (mm, nn) in zip(ms, ns)]

        func = theano.function([X,] + Factors, apply_complex_mat_to_kronecker(X, Factors))

        x_etha = x[0, ...] + 1j * x[1, ...]
        etha_factors = [factor[0, ...] + 1j * factor[1, ...] for factor in factors]

        print(x.shape)
        print([fac.shape for fac in factors])
        computed_with_np = np_apply_complex_mat_to_kronecker(x, factors)
        computed = func(x, *factors)
        ethalon = apply_mat_to_kronecker(x_etha, etha_factors)
        self.assertEqual(computed.shape[1:], ethalon.shape)
        self.assertTrue(np.allclose(computed[0, ...] + 1j * computed[1, ...], ethalon))
        self.assertTrue(np.allclose(computed_with_np[0, ...] + 1j * computed_with_np[1, ...], ethalon))


if __name__ == '__main__':
    unittest.main()