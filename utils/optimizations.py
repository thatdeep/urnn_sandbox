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


def modified_sgd(loss_or_grads, params, learning_rate, manifolds=None):
    manifolds = manifolds if manifolds else {}
    grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    if isinstance(manifolds, dict) and manifolds:
        for manifold_name in manifolds:
            # group all paramteters that are belongs to manifold
            man_param_tuple, man_grad_tuple = list(zip(*tuple((param, grad) for (param, grad) in zip(params, grads)
                                                              if (hasattr(param, 'name') and manifold_name in param.name))))
            man_param_pair = {manifold_name: man_param_tuple}
            man_grad_pair = {manifold_name: man_grad_tuple}

            # remove this parameters from params list and add as one tuple
            if len(man_param_tuple) == len(grads):
                params, grads = [], []
            else:
                params, grads = list(zip(*tuple((param, grad) for (param, grad) in zip(params, grads)
                                            if (hasattr(param, 'name') and manifold_name not in param.name))))
            params = [man_param_pair] + list(params)
            grads = [man_grad_pair] + list(grads)

    for param, grad in zip(params, grads):
        if param and isinstance(param, dict):
            # we have manifold parameters
            manifold_name = list(param.keys())[0]
            manifold = manifolds[manifold_name]

            pp = param[manifold_name]
            gg = grad[manifold_name]
            if len(pp) == 1:
                pp, gg = pp[0], gg[0]
                param_updates = manifold.retr(pp, manifold.lincomb(pp, -learning_rate, manifold.proj(pp, gg)))
            updates[pp] = param_updates
        else:
            updates[param] = param - learning_rate * grad
    return updates


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

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    """
    manifolds = manifolds if manifolds else {}
    grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    manifold_param_stack = []
    manifold_grad_stack = []

    if isinstance(manifolds, dict) and manifolds:

        for manifold_name in manifolds:
            manifold_tuple, manifold_grads_tuple = list(zip(*tuple((param, grad) for (param, grad) in zip(params, grads)\
                                                                   if (hasattr(param, 'name') and manifold_name in param.name))))
            manifold_tuple = {manifold_name: manifold_tuple}
            manifold_grads_tuple = {manifold_name: manifold_grads_tuple}

            if len(manifold_tuple[manifold_name]) == len(grads):
                params, grads = [], []
            else:
                params, grads = list(zip(*tuple((param, grad) for (param, grad) in zip(params, grads)
                                            if (hasattr(param, 'name') and manifold_name not in param.name))))
            manifold_param_stack.append(manifold_tuple)
            manifold_grad_stack.append(manifold_grads_tuple)
            params = list(params)
            grads = list(grads)

    params = manifold_param_stack + params
    grads = manifold_grad_stack + grads

    for param, grad in zip(params, grads):
        if param and isinstance(param, dict) and len(param) == 1 and isinstance(list(param.values())[0], tuple):# and "fixed_rank" in param[0].name:
            manifold_name = list(param.keys())[0]
            manifold = manifolds[manifold_name]
            if hasattr(manifold, "from_partial"):
                if hasattr(manifold, '_exponential') and manifold._exponential:
                    param_updates = manifold.exp(param[manifold_name],
                                              manifold.from_partial(param[manifold_name], grad[manifold_name]),
                                              -learning_rate)
                else:
                    param_updates = manifold.retr(param[manifold_name],
                                              manifold.from_partial(param[manifold_name], grad[manifold_name]),
                                              -learning_rate)
                for p, upd in zip(param[manifold_name], param_updates):
                    updates[p] = upd
            else:
                pp = param[manifold_name]
                gg = grad[manifold_name]
                if len(pp) == 1:
                    pp, gg = pp[0], gg[0]
                if hasattr(manifold, '_exponential') and manifold._exponential:
                    param_updates = manifold.exp(pp, manifold.proj(pp, gg), -learning_rate)
                    #param_updates = manifold.exp(pp, manifold.lincomb(pp, -learning_rate, manifold.proj(pp, gg)))
                else:
                    param_updates = manifold.retr(pp, manifold.lincomb(pp, -learning_rate, manifold.proj(pp, gg)))
                updates[pp] = param_updates
        else:
            updates[param] = param - learning_rate * grad

    return updates

'''
def adam(loss_or_grads, params, learning_rate=0.001, beta1=0.9,
         beta2=0.999, epsilon=1e-8, manifolds=None):
    """Adam updates
    Adam updates implemented as in [1]_.
    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float
        Learning rate
    beta1 : float
        Exponential decay rate for the first moment estimates.
    beta2 : float
        Exponential decay rate for the second moment estimates.
    epsilon : float
        Constant for numerical stability.
    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    Notes
    -----
    The paper [1]_ includes an additional hyperparameter lambda. This is only
    needed to prove convergence of the algorithm and has no practical use
    (personal communication with the authors), it is therefore omitted here.
    References
    ----------
    .. [1] Kingma, Diederik, and Jimmy Ba (2014):
           Adam: A Method for Stochastic Optimization.
           arXiv preprint arXiv:1412.6980.
    """
    all_grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    t_prev = theano.shared(lasagne.utils.floatX(0.))
    manifolds = manifolds if manifolds else {}
    updates = OrderedDict()

    if isinstance(manifolds, dict) and manifolds:

        for manifold_name in manifolds:
            manifold_tuple, manifold_grads_tuple = list(zip(*tuple((param, grad) for (param, grad) in zip(params, all_grads)\
                                                                   if (hasattr(param, 'name') and manifold_name in param.name))))
            manifold_tuple = {manifold_name: manifold_tuple}
            manifold_grads_tuple = {manifold_name: manifold_grads_tuple}

            params, grads = list(zip(*tuple((param, grad) for (param, grad) in zip(params, all_grads)
                                            if (hasattr(param, 'name') and manifold_name not in param.name))))
            params = [manifold_tuple] + list(params)
            grads = [manifold_grads_tuple] + list(grads)

    t = t_prev + 1
    a_t = learning_rate*T.sqrt(1-beta2**t)/(1-beta1**t)

    for param, g_t in zip(params, all_grads):
        if param and isinstance(param, dict) and len(param) == 1 and isinstance(list(param.values())[0], tuple):# and "fixed_rank" in param[0].name:
            manifold_name = list(param.keys())[0]
            manifold = manifolds[manifold_name]
            #param_updates = manifold.retr(param[manifold_name], grad[manifold_name], -learning_rate)
            #for p, upd in zip(param[manifold_name], param_updates):
            #    updates[p] = upd

            values = (p.get_value(borrow=True) for p in param)
            m_prev = (theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=p.broadcastable)
                      for (value, p) in zip(values, param))
            v_prev = (theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=p.broadcastable)
                      for (value, p) in zip(values, param))

            m_t = manifold.lincomb(param, beta1, m_prev, (1 - beta1), g_t)
            #v_t = manifold.lincomb(param, beta2, m_prev, (1 - beta2), manifold.proj((g_ta[0].dot(g_ta[1]).dot(g_ta[2]))**2))
            v_t = manifold.lincomb(param, beta2, m_prev, (1 - beta2), manifold.proj((g_t[0]**2, g_t[1]**2, g_t[2]**2), type='tan_vec'))
            step =

        else:
            value = param.get_value(borrow=True)
            m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)
            v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)

            m_t = beta1*m_prev + (1-beta1)*g_t
            v_t = beta2*v_prev + (1-beta2)*g_t**2
            step = a_t*m_t/(T.sqrt(v_t) + epsilon)

            updates[m_prev] = m_t
            updates[v_prev] = v_t
            updates[param] = param - step

    updates[t_prev] = t
    return updates
'''


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
 