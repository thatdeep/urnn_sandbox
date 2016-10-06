import numpy as np

from theano import tensor


def complex_reshape(x, shape, ndim=None):
    if ndim is not None:
        return x.reshape(tensor.concatenate([(2,), shape]), ndim + 1)
    return x.reshape((2,) + shape, ndim)


def apply_mat_to_kronecker(x, matrices):
    x = x.reshape((x.shape[0],) + tuple(mat.shape[0] for mat in matrices))
    result = x
    for mat in matrices:
        result = np.tensordot(result, mat, axes=([1], [0]))
    return result.reshape((x.shape[0], -1))

def apply_complex_mat_to_kronecker(x, matrices):
    x = tensor.reshape(x, (2, x.shape[1]) + tuple(mat.shape[1] for mat in matrices))
    result = x
    for mat in matrices:
        result = complex_tensordot(result, mat, axes=([1], [0]))
    return tensor.reshape(result, (2, x.shape[1], -1))


def complex_tensordot(a, b, axes=2):
    AR, AI = a[0, ...], a[1, ...]
    BR, BI = b[0, ...], b[1, ...]

    output = tensor.stack([
        tensor.tensordot(AR, BR, axes=axes) - tensor.tensordot(AI, BI, axes=axes),
        tensor.tensordot(AR, BI, axes=axes) + tensor.tensordot(AI, BR, axes=axes),
    ], axis=0)
    return output