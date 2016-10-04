import numpy as np
import scipy as sp
import numpy.linalg as la
import numpy.random as rnd

import theano
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams


try:
    import scipy.linalg
    imported_scipy = True
except ImportError:
    # some ops (e.g. Cholesky, Solve, A_Xinv_b) won't work
    imported_scipy = False


srnd = RandomStreams(rnd.randint(0, 1000))

import warnings

from manifolds.manifold import Manifold

import copy

import theano
from theano import tensor as T, Op, Apply

from theano.tensor import slinalg, as_tensor_variable

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

        """
        w, V = scipy.linalg.eig(A[0, :, :] + 1j * A[1, :, :], right=True)
        V = V.T
        #print(np.sum(w.imag), np.sum(V.imag))
        U = scipy.linalg.inv(V)

        exp_w = np.exp(w)
        X = np.subtract.outer(exp_w, exp_w) / np.subtract.outer(w, w)
        np.fill_diagonal(X, exp_w)
        #print(X.real, X.imag)
        real_middle = V.dot(gA[0, ...]).dot(U)
        imag_middle = V.dot(gA[1, ...]).dot(U)
        middle = V.dot(gA[0, :, :] - 1j * gA[1, :, :]).dot(U)
        #middle = middle.real * X.real + 1j * middle.imag * X.imag
        middle = middle * X
        Y = U.dot(middle).dot(np.conj(V).T)

        out[0] = np.stack([Y.real, -Y.imag]).astype(A.dtype)
        """
        """
        n = A.shape[1]
        fun = hconj
        Z = np.zeros_like(A)
        scale = la.norm(gA)
        gA /= scale
        res_mat = np.concatenate([np.concatenate([A, fun(gA)], axis=2),
                                  np.concatenate([Z, A], axis=2)], axis=1)
        res = scipy.linalg.expm(res_mat[0, :, :] + 1j * res_mat[1, :, :])
        out_mat = np.stack([res[:n, n:].real, res[:n, n:].imag], axis=0)
        out[0] = fun(out_mat * scale)
        """
complex_expm = ComplexExpm()


class UnitaryKron(Manifold):
    def __init__(nd, self):
        super(UnitaryKron, self).__init__()
        self._manifolds = tuple(Unitary(n=n) for n in nd)
        self._name = "Product of Stiefel unitary manifolds of dims {}".format(nd)

    @property
    def name(self):
        return self._name

    @property
    def dim(self):
        #hasn't computed
        #self._k * (2 * self._n * self._p - self._p**2)
        raise NotImplementedError


    @property
    def typicaldist(self):
        raise NotImplementedError

    def frac(self, A):
        return tuple(a[0, :, :] for a in A), tuple(a[1, :, :] for a in A)

    def complex_dot(self, A, B):
        prod = []
        for (a, b, manifold) in zip(A, B, self._manifolds):
            prod.append(manifold.complex_dot(a, b))
        return tuple(prod)

    def transpose(self, X):
        return tuple(manifold.transpose(x) for (manifold, x) in zip(self._manifolds, X))

    def conj(self, X):
        return tuple(manifold.conj(x) for (manifold, x) in zip(self._manifolds, X))

    def hconj(self, X):
        return tuple(manifold.hconj(x) for (manifold, x) in zip(self._manifolds, X))

    def inner(self, X, G, H):
        raise NotImplementedError

    def norm(self, X, G):
        raise NotImplementedError

    def dist(self, X, Y):
        raise NotImplementedError

    def herm(self, X):
        return tuple(manifold.herm(x) for (manifold, x) in zip(self._manifolds, X))

    def proj(self, X, U):
        return tuple(manifold.proj(x, u) for (manifold, x, u) in zip(self._manifolds, X, U))


    def tangent(self, X, U):
        return self.proj(X, U)

    def egrad2rgrad(self, X, U):
        return self.proj(X, U)

    def ehess2rhess(self, X, egrad, ehess, H):
        return tuple(manifold.ehess2rhess(x, eg, eh, h) for (manifold, x, eg, eh, h) in \
                     zip(self._manifolds, X, egrad, ehess, H))

    def retr(self, X, U):
        return tuple(manifold.retr(x, u) for (manifold, x, u) in zip(self._manifolds, X, U))


    def exp(self, X, U):
        # The exponential (in the sense of Lie group theory) of a tangent
        # vector U at X.
        raise NotImplementedError

    def log(self, X, Y):
        # The logarithm (in the sense of Lie group theory) of Y. This is the
        # inverse of exp.
        raise NotImplementedError

    def rand_np(self):
        return tuple(manifold.rand_np() for manifold in self._manifolds)


    def identity_np(self):
        return tuple(manifold.identity_np() for manifold in self._manifolds)

    def rand(self):
        return tuple(manifold.rand() for manifold in self._manifolds)

    def randvec(self, X):
        return tuple(manifold.randvec(x) for (manifold, x) in zip(self._manifolds, X))

    def zerovec(self, X):
        return tuple(manifold.zerovec(x) for (manifold, x) in zip(self._manifolds, X))

    def transp(self, x1, x2, d):
        return self.proj(x2, d)

    def lincomb(self, X, a1, u1, a2=None, u2=None):
        if u2 is None and a2 is None:
            return a1 * u1
        elif None not in [a1, u1, a2, u2]:
            return a1 * u1 + a2 * u2
        else:
            raise ValueError('FixedRankEmbeeded.lincomb takes 3 or 5 arguments')





class Unitary(Manifold):
    """
    Returns a manifold struct to optimize over the set of subspaces in C^n.

    function M = grassmanncomplexfactory(n, p)
    function M = grassmanncomplexfactory(n, p, k)

    Complex Grassmann manifold: each point on this manifold is a collection
    of k vector subspaces of dimension p embedded in C^n.

    The metric is obtained by making the Grassmannian a Riemannian quotient
    manifold of the complex Stiefel manifold, i.e., the manifold of
    orthonormal matrices, itself endowed with a metric by making it a
    Riemannian submanifold of the Euclidean space, endowed with the usual
    real-trace inner product, that is, it is the usual metric for the complex
    plane identified with R^2.

    This structure deals with complex matrices X of size n x p x k
    (or n x p if k = 1, which is the default) such that each n x p matrix is
    orthonormal, i.e., X'*X = eye(p) if k = 1, or X(:, :, i)' * X(:, :, i) =
    eye(p) for i = 1 : k if k > 1. Each n x p matrix is a numerical
    representation of the vector subspace its columns span.

    By default, k = 1.

    See also: grassmannfactory, stiefelcomplexfactory, grassmanngeneralizedfactory

    This file is part of Manopt: www.manopt.org.
    Original author: Hiroyuki Sato, May 21, 2015.
    Contributors:
    Change log:
    """
    def __init__(self, n, p=None, k=None):
        if p is None:
            p = n
        if k is None:
            k = 1
        if n <= 0:
            raise ValueError('n must be at least 1')
        if k <= 0:
            raise ValueError('k must be 1 or greater')
        if p > n:
            raise ValueError('p must be less or equal than n')
        self._n = n
        self._p = p
        # I didn't implement it for k > 1
        self._k = 1

        if k == 1:
            self._name = 'Complex Stiefel manifold St({}, {})'.format(n, p)
        else:
            self._name = 'Product complex Stiefel manifold St({}, {})^{}'.format(n, p, k)
        self._exponential = False


    @property
    def name(self):
        return self._name

    @property
    def dim(self):
        return self._k * (2 * self._n * self._p - self._p**2)

    @property
    def typicaldist(self):
        return np.sqrt(self._p, self._k)

    def frac(self, A):
        return A[0, :, :], A[1, :, :]

    def complex_dot_(self, A, B):
        A_real, A_imag = self.frac(A)
        B_real, B_imag = self.frac(B)
        re = A_real.dot(B_real) - A_imag.dot(B_imag)
        im = A_real.dot(B_imag) + A_imag.dot(B_real)

        return T.stack([re, im], axis=0)

    def complex_dot(self, A, B):
        A_real, A_imag = self.frac(A)
        B_real, B_imag = self.frac(B)
        prod = T.zeros((2, A_real.shape[0], B_real.shape[1]))
        prod = T.set_subtensor(prod[0, :, :], A_real.dot(B_real) - A_imag.dot(B_imag))
        prod = T.set_subtensor(prod[1, :, :], A_real.dot(B_imag) + A_imag.dot(B_real))
        return prod

    def transpose(self, X):
        return T.transpose(X, axes=(0, 2, 1))

    def conj(self, X):
        X_conj = T.copy(X)
        T.set_subtensor(X[1, :, :], -1 * X[1, :, :])
        return X_conj

    def hconj(self, X):
        XR, XI = self.frac(X)
        X_hconj = T.transpose(X, axes=(0, 2, 1))
        #X_hconj = T.zeros((2,) + XR.T.shape)
        #T.set_subtensor(X_hconj[0, :, :], XR.T)
        T.set_subtensor(X_hconj[1, :, :], -1 * X_hconj[1, :, :])
        return X_hconj

    def inner(self, X, G, H):
        GR, GI = self.frac(G)
        HR, HI = self.frac(H)
        # (AR + iAI)(BR + iBI) = ARBR - AIBI + i(ARBI + AIBR)
        # we return only real part of sum(hadamard(G, H))
        # old # return T.real(T.sum((GR + 1j * GI) *(HR + 1j * HI)))
        return T.sum(GR * HR - GI * HI)

    def norm(self, X, G):
        GR, GI = self.frac(G)
        return (GR + 1j * GI).norm()

    def dist(self, X, Y):
        raise NotImplementedError

    def herm(self, X):
        XH = self.hconj(X)
        return 0.5 * (X + XH)

    def proj(self, X, U):
        XHU = self.complex_dot(self.hconj(X), U)
        #XHU = X.conj().dot(U)
        herXHU = self.herm(XHU)
        #Up = U - X.dot(herXHU)
        Up = U - self.complex_dot(X, herXHU)
        return Up

    def tangent(self, X, U):
        return self.proj(X, U)

    def egrad2rgrad(self, X, U):
        return self.proj(X, U)

    def ehess2rhess(self, X, egrad, ehess, H):
        XHG = self.complex_dot(self.hconj(X), egrad)
        #XHG = X.conj().dot(egrad)
        herXHG = self.herm(XHG)
        HherXHG = self.complex_dot(H, herXHG)
        rhess = self.proj(X, ehess - HherXHG)
        return rhess

    def retr(self, X, U):
        YR, YI = self.frac(X + U)

        Q, R = T.nlinalg.qr(YR + 1j * YI)
        #Y = Q.dot(T.diag(T.sgn(T.sgn(T.diag(R))+.5)))
        Y = T.stack([Q.real, Q.imag])
        return Y

    def concat(self, arrays, axis):
        return T.concatenate(arrays, axis=axis+1)

    def exp(self, X, U):
        # The exponential (in the sense of Lie group theory) of a tangent
        # vector U at X.
        first = self.concat([X, U], axis=1)
        XhU = self.complex_dot(self.hconj(X), U)
        second = complex_expm(self.concat([self.concat([XhU, -self.complex_dot(self.hconj(U), U)], 1),
                                  self.concat([self.identity(), XhU], 1)], 0))
        third = self.concat([complex_expm(-XhU), self.zeros()], 0)
        exponential = self.complex_dot(self.complex_dot(first, second), third)
        return exponential

    def log(self, X, Y):
        # The logarithm (in the sense of Lie group theory) of Y. This is the
        # inverse of exp.
        raise NotImplementedError

    def rand_np(self):
        Q, unused = la.qr(rnd.normal(size=(self._n, self._p)) +
                          1j * rnd.normal(size=(self._n, self._p)))
        return np.stack([Q.real, Q.imag])

    def zeros(self):
        return tensor.zeros((2, self._n, self._n))

    def identity(self):
        return tensor.stack([tensor.eye(self._n), tensor.zeros((self._n, self._n))], axis=0)

    def identity_np(self):
        return np.stack([np.identity(self._n), np.zeros((self._n, self._n))])

    def rand(self):
        Q, unused = T.nlinalg.qr(srnd.normal(size=(self._n, self._p)) +
                          1j * srnd.normal(size=(self._n, self._p)))
        return T.stack[Q.real, Q.imag]

    def randvec(self, X):
        U = self.proj(X, T.stack([rnd.normal(size=(self._n, self._p)),
                          1j * rnd.normal(size=(self._n, self._p))]))
        U = U / self.norm(X, U)
        return T.stack([U.real, U.imag])

    def zerovec(self, X):
        return T.zeros_like(X)

    def transp(self, x1, x2, d):
        return self.proj(x2, d)

    def lincomb(self, X, a1, u1, a2=None, u2=None):
        if u2 is None and a2 is None:
            return a1 * u1
        elif None not in [a1, u1, a2, u2]:
            return a1 * u1 + a2 * u2
        else:
            raise ValueError('FixedRankEmbeeded.lincomb takes 3 or 5 arguments')


class DoubleComplex(Op):
    """
    Compute the matrix exponential of a square array.
    """

    __props__ = ()

    def make_node(self, A):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the Expm op")

        A = as_tensor_variable(A)
        assert A.ndim == 3
        sumc = theano.tensor.tensor3(dtype=A.dtype)
        return Apply(self, [A, ], [sumc, ])

    def perform(self, node, inputs, outputs):
        (A,) = inputs
        (sumc,) = outputs
        sumc[0] = 2 * A

    def grad(self, inputs, outputs):
        (A,) = inputs
        (g_out,) = outputs
        return [g_out * 2]

dcom = DoubleComplex()