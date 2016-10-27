import theano
import lasagne
import numpy as np

from theano import tensor
from manifolds import Unitary
from numpy import linalg as la, random as rnd


def wtt_construct(a, n, ranks):
    d = int(np.prod(n))
    assert(a.size == d)
    cores = []
    tail = a
    r0 = 1
    tail = np.reshape(tail, (r0 * n[0], -1))
    for k in range(len(n)):
        nk = n[k]
        r = ranks[k] if k < len(n) - 1 else 1
        if k == len(n) - 1:
            u, s, v = la.svd(tail, full_matrices=True)
            cores.append(u)
            return cores
        else:
            u, s, v = la.svd(tail, full_matrices=True)
        s = s[:r]
        v = v[:r]
        tail = np.dot(np.diag(s), v)
        tail = np.reshape(tail, (r * n[k+1], -1))
        cores.append(u)
    return cores


def _wtt_construct(a, n, ranks):
    cores = []
    tail = a
    r0 = 1
    tail = tensor.reshape(tail, (r0 * n[0], -1))
    for k in range(len(n)):
        print(k)
        nk = n[k]
        r = ranks[k]
        if k == len(n) - 1:
            u, s, v = tensor.nlinalg.svd(tail, full_matrices=True)
            cores.append(u)
            return cores
        else:
            u, s, v = tensor.nlinalg.svd(tail, full_matrices=False)
        s = s[:r]
        v = v[:r]
        tail = tensor.dot(tensor.diag(s), v)
        tail = tensor.reshape(tail, (r * n[k+1], -1))
        cores.append(tensor.stack([u.real, u.image], axis=0))
    return cores


def _wtt_compute_shapes(n, ranks):
    narr = np.array(n, dtype='int')
    nranks= np.array([1] + ranks, dtype='int')
    prods = narr * nranks
    shapes = [(prod, prod) for prod in prods]
    return shapes


def _frac(X):
    return X[0, ...], X[1, ...]


def complex_dot(A, B):
    print(A.ndim, B.ndim)
    A_real, A_imag = _frac(A)
    B_real, B_imag = _frac(B)
    re = A_real.dot(B_real) - A_imag.dot(B_imag)
    im = A_real.dot(B_imag) + A_imag.dot(B_real)

    return tensor.stack([re, im], axis=0)


def _transpose(X):
    return tensor.transpose(X, axes=(0, 2, 1))


def conj(X):
    #X_conj = tensor.set_subtensor(X[1, :, :], -1 * X[1, :, :])
    X_conj = tensor.stack([X[0, :, :], -1 * X[1, :, :]], axis=0)
    return X_conj


def hconj(X):
    return conj(_transpose(X))


def _wtt_image(x, cores, n, ranks, rec_dep=0):
    xk = x
    r0 = 1
    k = rec_dep
    if k == len(n) - 1:
        return complex_dot(hconj(cores[-1]), x)
    rk = ranks[k]
    rkm1 = r0 if k == 0 else ranks[k-1]
    nk = n[k]
    xk = tensor.reshape(xk, (2, rkm1 * nk, -1))
    xk = complex_dot(hconj(cores[k]), xk)
    xk1 = xk[:, :rk, :]
    zk1 = xk[:, rk:rkm1 * nk, :]
    xk1 = xk1.reshape((2, -1))
    yk1 = _wtt_image(xk1, cores, n, ranks, rec_dep=rec_dep + 1)
    yk1 = tensor.reshape(yk1, (2, rk, -1))
    yk = tensor.concatenate([yk1, zk1], axis=1)
    yk = yk.reshape((2, -1))
    return yk



def comp_wtt_image(x, cores, n, ranks, rec_dep=0, verbose=False):
    rd = '\t'*rec_dep
    if verbose:
        print('{}Insided rec step {}:'.format(rd, rec_dep))
        print('{}x shape is {}'.format(rd, x.shape))
    xk = x
    # SOOOOOOOQA
    #r0 = int(x.size / np.prod(n))
    r0 = 1

    k = rec_dep

    if k == len(n) - 1:
        if verbose:
            print('{}output of last iteration. Its shape = {}'.format(rd, complex_dot(hconj(cores[-1]), x).shape))
        print('core {}: {}'.format(k, cores[k].get_value().shape))
        return complex_dot(hconj(cores[-1]), x)
    rk = ranks[k]
    rkm1 = r0 if k == 0 else ranks[k-1]
    nk = n[k]
    xk = tensor.reshape(x, (2, rkm1 * nk, -1))
    if verbose:
        print('{}after first reshape with r[k-1]={} and nd[k]={} xk.shape = {}'.format(rd, rkm1, nk, xk.shape))
    print('core {}: {}'.format(k, cores[k].get_value().shape))
    xk = complex_dot(hconj(cores[k]), xk)
    if verbose:
        print('{}after dot with conj transposed cores[k] xk.shape = {}'.format(rd, xk.shape))
    xk1 = xk[:, :rk, :]
    if verbose:
        print('{}after retrieve first r[k]={} rows from xk: xk1.shape = {}'.format(rd, rk, xk1.shape))
    zk1 = xk[:, rk:rkm1 * nk, :]
    if verbose:
        print('{}after retrieve next rows from k: xk1.shape = {}'.format(rd, rk, xk1.shape))
    xk1 = tensor.reshape(xk1, (2, -1))
    if verbose:
        print('{}after reshape xk1 to ravel: xk1.shape = {}'.format(rd, xk1.shape))
        print('{}'.format((rd + '-'*40 + '\n') * 1))
        print('{}Leap into next recursive step'.format(rd))
    yk1 = comp_wtt_image(xk1, cores, n, ranks, rec_dep=rec_dep + 1, verbose=verbose)
    if verbose:
        print('{}Leap from next recursive step'.format(rd))
        print('{}'.format((rd + '-'*40 + '\n') * 1))
        print('{} result of next rec step shape: yk1.shape = {}'.format(rd, yk1.shape))
    yk1 = tensor.reshape(yk1, (2, rk, -1))
    if verbose:
        print('{}after reshape yk1 to ravel: yk1.shape = {}'.format(rd, yk1.shape))
    yk = tensor.concatenate([yk1, zk1], axis=1)
    if verbose:
        print('{}after concat yk1 and zk1: yk1 = {}, zk1 = {}, [yk1, zk1] = {}'.format(rd, yk1.shape, zk1.shape, yk.shape))
    yk = tensor.reshape(yk, (2, -1))
    if verbose:
        print('{}finally reshape yk to ravel: yk.shape = {}'.format(rd, yk.shape))
    return yk


class WTTLayer(lasagne.layers.Layer):
    def __init__(self, incoming, nd, ranks, **kwargs):
        super(WTTLayer, self).__init__(incoming, **kwargs)
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_inputs = num_inputs // 2
        self.n_hidden = num_inputs // 2
        self.shape = (self.n_hidden, self.n_hidden)

        self.nd = nd
        self.ranks = ranks

        self.shapes = _wtt_compute_shapes(nd, ranks)
        self.manifold = [Unitary(shape[0]) for shape in self.shapes]

        cores_np = [man.rand_np() for man in self.manifold]

        basename = kwargs.get('name', '')
        self.attr_names = ["core[{}]".format(i) for i in range(len(cores_np))]
        unique_ids = [man.str_id for man in self.manifold]
        for attr_name, unique_id, core, man in zip(self.attr_names, unique_ids, cores_np, self.manifold):
            added_param = self.add_param(core,
                                         core.shape,
                                         name="{}core[{}]".format(basename, unique_id),
                                         regularizable=False)
            setattr(self, attr_name, added_param)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        cores = tuple(getattr(self, attr_name) for attr_name in self.attr_names)
        unitary_input = tensor.reshape(input, (input.shape[0], 2, self.num_inputs))
        IR, II = unitary_input[:, 0, :], unitary_input[:, 1, :]
        I = tensor.stack([IR, II], axis=0)
        output = _wtt_image(tensor.stack([IR, II], axis=0), cores, self.nd, self.ranks)
        output = tensor.stack(_frac(output), axis=1)
        output = output.reshape((input.shape[0], -1))
        return output

    def get_output_shape_for(self, input_shape):
        if len(input_shape) > 2:
            return (input_shape[0], int(np.prod(input_shape[1:])))
        return input_shape