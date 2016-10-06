import numpy as np

from .unitary import Unitary
from .manifold import Manifold


class UnitaryKron(Manifold):
    def __init__(self, nd):
        super(UnitaryKron, self).__init__()
        self._manifolds = tuple(Unitary(n=n) for n in nd)
        self.n = int(np.prod(nd))
        self.n_factors = len(nd)
        self._name = "Product of Unitary manifolds of dims {}".format(nd)

    @property
    def name(self):
        return self._name

    @property
    def short_name(self):
        return "UnitaryKron({})".format(tuple(manifold._n for manifold in self._manifolds))

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
        return tuple(manifold.exp(x, u) for (manifold, x, u) in zip(self._manifolds, X, U))

    def get_back(self, X):
        return tuple(manifold.get_back(x) for (manifold, x) in zip(self._manifolds, X))


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
        a1 = [a1] * self.n_factors if not hasattr(a1, '__len__') else a1
        a2 = [a2] * self.n_factors if not hasattr(a2, '__len__') else a2
        if u2 is None:
            u2 = [u2] * self.n_factors
        return tuple(manifold.lincomb(x, a1_, u1_, a2_, u2_) for \
                     (manifold, x, a1_, u1_, a2_, u2_) in zip(self._manifolds, X, a1, u1, a2, u2))