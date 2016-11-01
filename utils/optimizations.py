import theano
import lasagne
import theano.tensor as T
import numpy as np

from collections import OrderedDict


def clipped_gradients(gradients, gradient_clipping):
    clipped_grads = [T.clip(g, -gradient_clipping, gradient_clipping)
                     for g in gradients]
    return clipped_grads

def gradient_descent(learning_rate, parameters, gradients):        
    updates = [(p, p - learning_rate * g) for p, g in zip(parameters, gradients)]
    return updates

def gradient_descent_momentum(learning_rate, momentum, parameters, gradients):
    velocities = [theano.shared(np.zeros_like(p.get_value(), 
                                              dtype=theano.config.floatX)) for p in parameters]

    updates1 = [(vel, momentum * vel - learning_rate * g) 
                for vel, g in zip(velocities, gradients)]
    updates2 = [(p, p + vel) for p, vel in zip(parameters, velocities)]
    updates = updates1 + updates2
    return updates 


def rms_prop(learning_rate, parameters, gradients):        
    rmsprop = [theano.shared(1e-3*np.ones_like(p.get_value())) for p in parameters]
    new_rmsprop = [0.9 * vel + 0.1 * (g**2) for vel, g in zip(rmsprop, gradients)]

    updates1 = list(zip(rmsprop, new_rmsprop))
    updates2 = [(p, p - learning_rate * g / T.sqrt(rms)) for 
                p, g, rms in zip(parameters, gradients, new_rmsprop)]
    updates = updates1 + updates2
    return updates, rmsprop


def custom_sgd(loss_or_grads, params, learning_rate, manifolds=None):
    """Stochastic Gradient Descent (SGD) updates

    Generates update expressions of the form:

    * ``param := param - learning_rate * gradient``

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    manifolds : dict
        Dictionary that contains manifolds for manifold parameters

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    """
    def filter_func(manifold_name, inverse=False):
        def inner_filter_func(param_grad_tuple):
            filter_result = (hasattr(param_grad_tuple[0], 'name') and manifold_name in param_grad_tuple[0].name)
            return not filter_result if inverse else filter_result
        return inner_filter_func

    manifolds = manifolds if manifolds else {}
    grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    manifolds_params_stack = []
    manifolds_grads_stack = []


    if isinstance(manifolds, dict) and manifolds:
        for manifold_name in manifolds:
            # filter parameters and gradients for specific manifold
            man_params_tuple, man_grads_tuple = zip(*filter(filter_func(manifold_name), zip(params, grads)))

            man_params_tuple = {manifold_name: tuple(man_params_tuple)}
            man_grads_tuple = {manifold_name: tuple(man_grads_tuple)}

            if len(man_params_tuple[manifold_name]) == len(params):
                params, grads = [], []
            else:
                params, grads = zip(*filter(filter_func(manifold_name, inverse=True), zip(params, grads)))
            manifolds_params_stack.append(man_params_tuple)
            manifolds_grads_stack.append(man_grads_tuple)
            params = list(params)
            grads = list(grads)

    params = manifolds_params_stack + params
    grads = manifolds_grads_stack + grads

    for param, grad in zip(params, grads):
        if isinstance(param, dict):
            manifold_name = list(param.keys())[0]
            manifold = manifolds[manifold_name]
            if hasattr(manifold, "from_partial"):
                grad_from_partial = manifold.from_partial(param[manifold_name], grad[manifold_name])
                grad_step = manifold.lincomb(param[manifold_name], grad_from_partial, -learning_rate)
                param_updates = manifold.retr(param[manifold_name], grad_step)
                for p, upd in zip(param[manifold_name], param_updates):
                    updates[p] = upd
            else:
                param_tuple = param[manifold_name]
                grad_tuple = grad[manifold_name]
                if len(param_tuple) == 1:
                    param_tuple, grad_tuple = param_tuple[0], grad_tuple[0]
                grad_step = manifold.lincomb(param_tuple, manifold.proj(param_tuple, grad_tuple), -learning_rate)
                param_updates = manifold.retr(param_tuple, grad_step)
                updates[param_tuple] = param_updates
        else:
            updates[param] = param - learning_rate * grad
    return updates


def sgd(loss_or_grads, params, learning_rate):
    """Stochastic Gradient Descent (SGD) updates
    Generates update expressions of the form:
    * ``param := param - learning_rate * gradient``
    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    """
    grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad

    return updates


def apply_nesterov_momentum(updates, params=None, momentum=0.9, manifolds=None):
    """Returns a modified update dictionary including Nesterov momentum
    Generates update expressions of the form:
    * ``velocity := momentum * velocity + updates[param] - param``
    * ``param := param + momentum * velocity + updates[param] - param``
    Parameters
    ----------
    updates : OrderedDict
        A dictionary mapping parameters to update expressions
    params : iterable of shared variables, optional
        The variables to apply momentum to. If omitted, will apply
        momentum to all `updates.keys()`.
    momentum : float or symbolic scalar, optional
        The amount of momentum to apply. Higher momentum results in
        smoothing over more update steps. Defaults to 0.9.
    Returns
    -------
    OrderedDict
        A copy of `updates` with momentum updates for all `params`.
    Notes
    -----
    Higher momentum also results in larger update steps. To counter that,
    you can optionally scale your learning rate by `1 - momentum`.
    The classic formulation of Nesterov momentum (or Nesterov accelerated
    gradient) requires the gradient to be evaluated at the predicted next
    position in parameter space. Here, we use the formulation described at
    https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617,
    which allows the gradient to be evaluated at the current parameters.
    See Also
    --------
    nesterov_momentum : Shortcut applying Nesterov momentum to SGD updates
    """
    manifolds = {} if manifolds is None else manifolds
    if params is None:
        params = updates.keys()
    updates = OrderedDict(updates)
    updates_backup = updates

    if isinstance(manifolds, dict) and manifolds:
        for manifold_name in manifolds:
            manifold_tuple = tuple(param for param in params if hasattr(param, "name") and manifold_name in param.name)
            manifold_tuple = {manifold_name: manifold_tuple}
            params = [param for param in params if not hasattr(param, "name") or manifold_name not in param.name]
            params = [manifold_tuple] + list(params)

    for param in params:
        if param and isinstance(param, dict) and len(param) == 1 and isinstance(list(param.values())[0], tuple):# and "fixed_rank" in param[0].name:
            manifold_name = list(param.keys())[0]
            manifold = manifolds[manifold_name]

            values = [p.get_value(borrow=True) for p in param[manifold_name]]
            if len(values) == 1:
                value = values[0]
                lone_param = param[manifold_name][0]
                velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=lone_param.broadcastable)
                multiplied = manifold.lincomb(lone_param, momentum, velocity)
                update_for_transport = updates[lone_param]
                x = manifold.transp(lone_param, update_for_transport, multiplied)
                updates[velocity] = x
                if hasattr(manifold, '_exponential') and manifold._exponential:
                    result = manifold.exp(updates[lone_param], x, momentum)
                    #result = manifold.exp(updates[lone_param], manifold.lincomb(updates[lone_param], momentum, x))
                else:
                    result = manifold.retr(updates[lone_param], manifold.lincomb(updates[lone_param], momentum, x))
                updates[lone_param] = result
            else:
                velocities = tuple(theano.shared(np.zeros(v.shape, dtype=v.dtype), broadcastable=p.broadcastable)\
                                 for (v, p) in zip(values, param[manifold_name]))
                multiplied = manifold.lincomb(param[manifold_name], momentum, velocities)
                updates_for_transport = [updates[p] for p in param[manifold_name]]
                x = manifold.transp(param[manifold_name], updates_for_transport, multiplied)
                for v, x_part in zip(velocities, x):
                    updates[v] = x_part
                upp = [updates[p] for p in param[manifold_name]]
                if hasattr(manifold, '_exponential') and manifold._exponential:
                    result = manifold.exp(upp, manifold.lincomb(upp, momentum, x))
                else:
                    result = manifold.retr(upp, manifold.lincomb(upp, momentum, x))
                for p, r in zip(param[manifold_name], result):
                    updates[p] = r
        else:
            value = param.get_value(borrow=True)
            velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                     broadcastable=param.broadcastable)
            x = momentum * velocity + updates[param] - param
            updates[velocity] = x
            updates[param] = momentum * x + updates[param]
    """
        for param in params:
            value = param.get_value(borrow=True)
            velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                     broadcastable=param.broadcastable)
            x = momentum * velocity + updates[param] - param
            updates[velocity] = x
            updates[param] = momentum * x + updates[param]
    """
    return updates



def nesterov_momentum(loss_or_grads, params, learning_rate, momentum=0.9, manifolds=None):
    """Stochastic Gradient Descent (SGD) updates with Nesterov momentum
    Generates update expressions of the form:
    * ``velocity := momentum * velocity - learning_rate * gradient``
    * ``param := param + momentum * velocity - learning_rate * gradient``
    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    momentum : float or symbolic scalar, optional
        The amount of momentum to apply. Higher momentum results in
        smoothing over more update steps. Defaults to 0.9.
    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    Notes
    -----
    Higher momentum also results in larger update steps. To counter that,
    you can optionally scale your learning rate by `1 - momentum`.
    The classic formulation of Nesterov momentum (or Nesterov accelerated
    gradient) requires the gradient to be evaluated at the predicted next
    position in parameter space. Here, we use the formulation described at
    https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617,
    which allows the gradient to be evaluated at the current parameters.
    See Also
    --------
    apply_nesterov_momentum : Function applying momentum to updates
    """
    updates = custom_sgd(loss_or_grads, params, learning_rate, manifolds=manifolds)
    return apply_nesterov_momentum(updates, momentum=momentum, manifolds=manifolds)
 