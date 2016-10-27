import theano
import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from theano import tensor

from manifolds.manifold import Manifold
from utils.complex_expm import complex_expm
from theano.tensor.shared_randomstreams import RandomStreams


srnd = RandomStreams(rnd.randint(0, 1000))


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
    def __init__(self, n, retr_type='svd'):
        if n <= 0:
            raise ValueError('n must be at least 1')
        if retr_type not in ['svd', 'qr']:
            raise ValueError('retr_type mist be either "svd" or "qr"')
        self.retr_type = retr_type
        self._n = n
        # I didn't implement it for k > 1
        self._name = 'Unitary manifold U({}) = St({}, {})'.format(n, n, n)
        self._exponential = False

    @property
    def name(self):
        return self._name

    @property
    def short_name(self):
        return "Unitary({})".format(self._n)

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

        return tensor.stack([re, im], axis=0)

    def complex_dot(self, A, B):
        A_real, A_imag = self.frac(A)
        B_real, B_imag = self.frac(B)
        prod = tensor.zeros((2, A_real.shape[0], B_real.shape[1]))
        prod = tensor.set_subtensor(prod[0, :, :], A_real.dot(B_real) - A_imag.dot(B_imag))
        prod = tensor.set_subtensor(prod[1, :, :], A_real.dot(B_imag) + A_imag.dot(B_real))
        return prod

    def transpose(self, X):
        return tensor.transpose(X, axes=(0, 2, 1))

    def conj(self, X):
        X_conj = tensor.copy(X)
        tensor.set_subtensor(X[1, :, :], -1 * X[1, :, :])
        return X_conj

    def hconj(self, X):
        XR, XI = self.frac(X)
        X_hconj = tensor.transpose(X, axes=(0, 2, 1))
        #X_hconj = T.zeros((2,) + XR.T.shape)
        #T.set_subtensor(X_hconj[0, :, :], XR.T)
        tensor.set_subtensor(X_hconj[1, :, :], -1 * X_hconj[1, :, :])
        return X_hconj

    def inner(self, X, G, H):
        GR, GI = self.frac(G)
        HR, HI = self.frac(H)
        # (AR + iAI)(BR + iBI) = ARBR - AIBI + i(ARBI + AIBR)
        # we return only real part of sum(hadamard(G, H))
        # old # return T.real(T.sum((GR + 1j * GI) *(HR + 1j * HI)))
        return tensor.sum(GR * HR - GI * HI)

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

        #Q, R = tensor.nlinalg.qr(YR + 1j * YI)
        #Y = Q.dot(T.diag(T.sgn(T.sgn(T.diag(R))+.5)))
        #Y = tensor.stack([Q.real, Q.imag])
        U, S, V = tensor.nlinalg.svd(YR + 1j * YI, full_matrices=False)
        Y = U.dot(tensor.eye(S.size)).dot(V)
        Y = tensor.stack([Y.real, Y.imag])
        return Y

    def get_back(self, X):
        XR, XI = self.frac(X)
        Q, R = tensor.nlinalg.qr(XR + 1j * XI)
        Y = tensor.stack([Q.real, Q.imag])
        return Y

    def concat(self, arrays, axis):
        return tensor.concatenate(arrays, axis=axis+1)

    def exp(self, X, U, t):
        # The exponential (in the sense of Lie group theory) of a tangent
        # vector U at X.
        first = self.concat([X, U], axis=1)
        XhU = self.complex_dot(self.hconj(X), U)
        second = complex_expm(t * self.concat([self.concat([XhU, -self.complex_dot(self.hconj(U), U)], 1),
                                  self.concat([self.identity(), XhU], 1)], 0))
        third = self.concat([t * complex_expm(-XhU), self.zeros()], 0)
        exponential = self.complex_dot(self.complex_dot(first, second), third)
        return exponential

    def log(self, X, Y):
        # The logarithm (in the sense of Lie group theory) of Y. This is the
        # inverse of exp.
        raise NotImplementedError

    def rand_np(self):
        Q, unused = la.qr(rnd.normal(size=(self._n, self._n)) +
                          1j * rnd.normal(size=(self._n, self._n)))
        return np.stack([Q.real, Q.imag])

    def zeros(self):
        return tensor.zeros((2, self._n, self._n))

    def identity(self):
        return tensor.stack([tensor.eye(self._n), tensor.zeros((self._n, self._n))], axis=0)

    def identity_np(self):
        return np.stack([np.identity(self._n), np.zeros((self._n, self._n))])

    def rand(self):
        Q, unused = tensor.nlinalg.qr(srnd.normal(size=(self._n, self._n)) +
                          1j * srnd.normal(size=(self._n, self._n)))
        return tensor.stack[Q.real, Q.imag]

    def randvec(self, X):
        U = self.proj(X, tensor.stack([rnd.normal(size=(self._n, self._n)),
                          1j * rnd.normal(size=(self._n, self._n))]))
        U = U / self.norm(X, U)
        return tensor.stack([U.real, U.imag])

    def zerovec(self, X):
        return tensor.zeros_like(X)

    def transp(self, x1, x2, d):
        return self.proj(x2, d)

    def lincomb(self, X, a1, u1, a2=None, u2=None):
        if u2 is None and a2 is None:
            return a1 * u1
        elif None not in [a1, u1, a2, u2]:
            return a1 * u1 + a2 * u2
        else:
            raise ValueError('FixedRankEmbeeded.lincomb takes 3 or 5 arguments')