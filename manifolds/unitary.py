import numpy as np
import numpy.linalg as la
import numpy.random as rnd

import theano
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams

srnd = RandomStreams(rnd.randint(0, 1000))

import warnings
import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from manifolds.manifold import Manifold

import copy

import theano
from theano import tensor as T


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

    def complex_dot(self, A, B):
        A_real, A_imag = self.frac(A)
        B_real, B_imag = self.frac(B)
        dotted = T.zeros((2, A_real.shape[0], B_real.shape[1]))
        T.set_subtensor(dotted[0, :, :], A_real.dot(B_real) - A_imag.dot(B_imag))
        T.set_subtensor(dotted[1, :, :], A_real.dot(B_imag) + A_imag.dot(B_real))
        return dotted

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

    def exp(self, X, U):
        # The exponential (in the sense of Lie group theory) of a tangent
        # vector U at X.
        raise NotImplementedError

    def log(self, X, Y):
        # The logarithm (in the sense of Lie group theory) of Y. This is the
        # inverse of exp.
        raise NotImplementedError

    def rand_np(self):
        Q, unused = la.qr(rnd.normal(size=(self._n, self._p)) +
                          1j * rnd.normal(size=(self._n, self._p)))
        return np.stack([Q.real, Q.imag])

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


