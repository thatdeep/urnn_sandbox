import theano
import numpy as np

from theano import Op, Apply

from theano.tensor import as_tensor_variable

from .theano_complex_extension import hconj, complex_matrix_dot



class ComplexMatrixInverse(Op):
    """Computes the inverse of a matrix :math:`A`.
    Given a square matrix :math:`A`, ``matrix_inverse`` returns a square
    matrix :math:`A_{inv}` such that the dot product :math:`A \cdot A_{inv}`
    and :math:`A_{inv} \cdot A` equals the identity matrix :math:`I`.
    Notes
    -----
    When possible, the call to this op will be optimized to the call
    of ``solve``.
    """

    __props__ = ()

    def __init__(self):
        pass

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 3
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        inv_res = np.linalg.inv(x[0, ...] + 1j * x[1, ...])

        z[0] = np.stack((inv_res.real, inv_res.imag), axis=0).astype(x.dtype)

    def grad(self, inputs, g_outputs):
        r"""The gradient function should return
            .. math:: V\frac{\partial X^{-1}}{\partial X},
        where :math:`V` corresponds to ``g_outputs`` and :math:`X` to
        ``inputs``. Using the `matrix cookbook
        <http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3274>`_,
        one can deduce that the relation corresponds to
            .. math:: (X^{-1} \cdot V^{T} \cdot X^{-1})^T.
        """
        x, = inputs
        xi = self(x)
        gz, = g_outputs
        # TT.dot(gz.T,xi)
        return [-hconj(complex_matrix_dot(xi, hconj(gz), xi))]

    def R_op(self, inputs, eval_points):
        r"""The gradient function should return
            .. math:: \frac{\partial X^{-1}}{\partial X}V,
        where :math:`V` corresponds to ``g_outputs`` and :math:`X` to
        ``inputs``. Using the `matrix cookbook
        <http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3274>`_,
        one can deduce that the relation corresponds to
            .. math:: X^{-1} \cdot V \cdot X^{-1}.
        """
        x, = inputs
        xi = self(x)
        ev, = eval_points
        if ev is None:
            return [None]
        return [-complex_matrix_dot(xi, ev, xi)]

    def infer_shape(self, node, shapes):
        return shapes

complex_matrix_inverse = ComplexMatrixInverse()