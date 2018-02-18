import theano
import numpy as np

from theano import Op, Apply

from theano.tensor import as_tensor_variable


try:
    import scipy.linalg
    imported_scipy = True
except ImportError:
    # some ops (e.g. Cholesky, Solve, A_Xinv_b) won't work
    imported_scipy = False


class ComplexExpm(Op):
    """
    Compute the matrix exponential of a square array.
    """

    __props__ = ()

    def make_node(self, A):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the Expm op")

        A = as_tensor_variable(A)
        assert A.ndim == 3
        expm = theano.tensor.tensor3(dtype=A.dtype)
        return Apply(self, [A, ], [expm, ])

    def perform(self, node, inputs, outputs):
        (A,) = inputs
        (expm,) = outputs
        temp = scipy.linalg.expm(A[0, :, :] + 1j * A[1, :, :])
        expm[0] = np.stack([temp.real, temp.imag])

    def grad(self, inputs, outputs):
        (A,) = inputs
        (g_out,) = outputs
        return [ComplexExpmGrad()(A, g_out)]

    def infer_shape(self, node, shapes):
        return [shapes[0]]


def _hconj_internal(x):
    x_hconj = np.transpose(x, axes=(0, 2, 1)).copy()
    x_hconj[1, :, :] = -x_hconj[1, :, :]
    return x_hconj


class ComplexExpmGrad(Op):
    """
    Gradient of the matrix exponential of a square array.
    """

    __props__ = ()

    def make_node(self, A, gw):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the Expm op")
        A = as_tensor_variable(A)
        assert A.ndim == 3
        out = theano.tensor.tensor3(dtype=A.dtype)
        return Apply(self, [A, gw], [out, ])

    def infer_shape(self, node, shapes):
        return [shapes[0]]

    def perform(self, node, inputs, outputs):
        # Kalbfleisch and Lawless, J. Am. Stat. Assoc. 80 (1985) Equation 3.4
        # Kind of... You need to do some algebra from there to arrive at
        # this expression.

        (A, gA) = inputs
        (out,) = outputs

        w, V = scipy.linalg.eig(A[0, :, :] + 1j * A[1, :, :], right=True)
        U = scipy.linalg.inv(V)

        exp_w = np.exp(w)
        X = np.subtract.outer(exp_w, exp_w) / np.subtract.outer(w, w)
        np.fill_diagonal(X, exp_w)
        Y = np.conj(V.dot(U.dot(gA[0, :, :].T - 1j * gA[1, :, :].T).dot(V) * X).dot(U)).T

        out[0] = np.stack([Y.real, Y.imag]).astype(A.dtype)


complex_expm = ComplexExpm()