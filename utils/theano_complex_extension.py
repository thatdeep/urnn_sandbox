import numpy as np

from theano import tensor


#------------------------------------------------------------------------------
# Complex theano funcs


def complex_reshape(x, shape, ndim=None):
    if ndim is not None:
        return x.reshape(tensor.concatenate([(2,), shape]), ndim + 1)
    return x.reshape((2,) + shape, ndim)


def complex_tensordot(a, b, axes=2):
    AR, AI = a[0, ...], a[1, ...]
    BR, BI = b[0, ...], b[1, ...]

    output = tensor.stack([
        tensor.tensordot(AR, BR, axes=axes) - tensor.tensordot(AI, BI, axes=axes),
        tensor.tensordot(AR, BI, axes=axes) + tensor.tensordot(AI, BR, axes=axes),
    ], axis=0)
    return output


def apply_complex_mat_to_kronecker(x, matrices):
    x = x.reshape((2, x.shape[1]) + tuple(mat.shape[1] for mat in matrices))
    result = x
    for mat in matrices:
        print(x.ndim)
        print(mat)
        result = complex_tensordot(result, mat, axes=([1], [0]))
    return result
    return tensor.reshape(result, (2, x.shape[1], -1))


#------------------------------------------------------------------------------
# Ordinary theano funcs


def apply_mat_to_kronecker(x, matrices):
    x = x.reshape((x.shape[0],) + tuple(mat.shape[0] for mat in matrices))
    result = x
    for mat in matrices:
        result = np.tensordot(result, mat, axes=([1], [0]))
    return result
    return result.reshape((x.shape[0], -1))


#------------------------------------------------------------------------------
# Numpy funcs for unit tests

def np_apply_complex_mat_to_kronecker(x, matrices):
    x = x.reshape((2, x.shape[1]) + tuple(mat.shape[1] for mat in matrices))
    result = x
    for mat in matrices:
        result = np_complex_tensordot(result, mat, axes=([1], [0]))
    return result
    return np.reshape(result, (2, x.shape[1], -1))



def np_complex_tensordot(a, b, axes=2):
    AR, AI = a[0, ...], a[1, ...]
    BR, BI = b[0, ...], b[1, ...]

    output = np.stack([
        np.tensordot(AR, BR, axes=axes) - np.tensordot(AI, BI, axes=axes),
        np.tensordot(AR, BI, axes=axes) + np.tensordot(AI, BR, axes=axes),
    ], axis=0)
    return output