import theano
import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from theano import tensor

from manifolds.manifold import Manifold
from utils.complex_expm import complex_expm
from theano.tensor.shared_randomstreams import RandomStreams

from utils.theano_complex_extension import frac, identity, zeros, complex_dot, hconj


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
    def __init__(self, n, retr_mode="svd"):
        if n <= 0:
            raise ValueError('n must be at least 1')
        if retr_mode not in ["svd", "qr", "exp"]:
            raise ValueError('retr_type mist be "svd", "qr" or "exp", but is "{}"'.format(retr_mode))
        self.retr_mode = retr_mode
        self._n = n
        # I didn't implement it for k > 1
        self._name = 'Unitary manifold U({}) = St({}, {})'.format(n, n, n)

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

    def inner(self, X, G, H):
        GR, GI = frac(G)
        HR, HI = frac(H)
        # (AR + iAI)(BR + iBI) = ARBR - AIBI + i(ARBI + AIBR)
        # we return only real part of sum(hadamard(G, H))
        # old # return T.real(T.sum((GR + 1j * GI) *(HR + 1j * HI)))
        return tensor.sum(GR * HR - GI * HI)

    def norm(self, X, G):
        GR, GI = frac(G)
        return (GR + 1j * GI).norm()

    def dist(self, X, Y):
        raise NotImplementedError

    def herm(self, X):
        XH = hconj(X)
        return 0.5 * (X + XH)

    def proj(self, X, U):
        XHU = complex_dot(hconj(X), U)
        herXHU = self.herm(XHU)
        Up = U - complex_dot(X, herXHU)
        return Up

    def tangent(self, X, U):
        return self.proj(X, U)

    def egrad2rgrad(self, X, U):
        return self.proj(X, U)

    def ehess2rhess(self, X, egrad, ehess, H):
        XHG = complex_dot(hconj(X), egrad)
        #XHG = X.conj().dot(egrad)
        herXHG = self.herm(XHG)
        HherXHG = complex_dot(H, herXHG)
        rhess = self.proj(X, ehess - HherXHG)
        return rhess

    def retr(self, X, U, mode="default"):
        if mode == "exp":
            return self.exp(X, U)
        elif mode == "qr":
            YR, YI = frac(X + U)
            Q, R = tensor.nlinalg.qr(YR + 1j * YI)
            Y = tensor.stack([Q.real, Q.imag])
            return Y
        elif mode == "svd":
            YR, YI = frac(X + U)
            U, S, V = tensor.nlinalg.svd(YR + 1j * YI, full_matrices=False)
            Y = U.dot(tensor.eye(S.size)).dot(V)
            Y = tensor.stack([Y.real, Y.imag])
            return Y
        elif mode == "default":
            return self.retr(X, U, mode=self.retr_mode)
        else:
            raise ValueError('mode must equal to "svd", "qr", "exp" or "default", but "{}" is given'.format(mode))

    def concat(self, arrays, axis):
        return tensor.concatenate(arrays, axis=axis+1)

    def exp(self, X, U):
        # The exponential (in the sense of Lie group theory) of a tangent
        # vector U at X.
        first = self.concat([X, U], axis=1)
        XhU = complex_dot(hconj(X), U)
        second = complex_expm(self.concat([self.concat([XhU, -complex_dot(hconj(U), U)], 1),
                                  self.concat([identity(self._n), XhU], 1)], 0))
        third = self.concat([complex_expm(-XhU), zeros((self._n, self._n))], 0)
        exponential = complex_dot(complex_dot(first, second), third)
        return exponential

    def log(self, X, Y):
        # The logarithm (in the sense of Lie group theory) of Y. This is the
        # inverse of exp.
        raise NotImplementedError

    def rand_np(self):
        Q, unused = la.qr(rnd.normal(size=(self._n, self._n)) + 1j * rnd.normal(size=(self._n, self._n)))
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
        randvec_embedding = tensor.stack([rnd.normal(size=(self._n, self._n)),
                                          1j * rnd.normal(size=(self._n, self._n))])
        U = self.proj(X, randvec_embedding)
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