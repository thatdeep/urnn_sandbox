import pickle
import gzip
import theano
import numpy as np
import theano.tensor as T
from theano.ifelse import ifelse
from layers.models import *
from utils.optimizations import *
from numpy import linalg as la
import argparse

theano.config.exception_verbosity = "high"
from theano.compile.debugmode import DebugMode


def generate_data(time_steps, n_data):
    x = np.asarray(np.zeros((time_steps, int(n_data), 2)),
                   dtype=theano.config.floatX)

    x[:,:,0] = np.asarray(np.random.uniform(low=0.,
                                            high=1.,
                                            size=(time_steps, n_data)),
                          dtype=theano.config.floatX)
    
    inds = np.asarray(np.random.randint(time_steps//2, size=(n_data, 2)))
    inds[:, 1] += time_steps//2
    
    for i in range(int(n_data)):
        x[inds[i, 0], i, 1] = 1.0
        x[inds[i, 1], i, 1] = 1.0
 
    y = (x[:,:,0] * x[:,:,1]).sum(axis=0)
    y = np.reshape(y, (n_data, 1))

    return x, y

    
    
def main(n_iter, n_batch, n_hidden, time_steps, learning_rate, savefile, model, input_type, out_every_t, loss_function):
    
    # --- Set data params ----------------
    n_input = 2
    n_output = 1
    n_train = 100000
    n_test = 10000
    num_batches = n_train // n_batch
  

    # --- Create data --------------------
    train_x, train_y = generate_data(time_steps, n_train)
    test_x, test_y = generate_data(time_steps, n_test)
 

    s_train_x = theano.shared(train_x)
    s_train_y = theano.shared(train_y)

    s_test_x = theano.shared(test_x)
    s_test_y = theano.shared(test_y)

    # --- Create theano graph and compute gradients ----------------------

    gradient_clipping = np.float32(1)

    if (model == 'LSTM'):           
        inputs, parameters, costs = LSTM(n_input, n_hidden, n_output, input_type=input_type,
                                         out_every_t=out_every_t, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)
        gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]

    elif (model == 'complex_RNN_momentum'):
        inputs, parameters, costs = complex_RNN(n_input, n_hidden, n_output, input_type=input_type,
                                                out_every_t=out_every_t, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)
    elif (model == 'complex_RNN'):
        inputs, parameters, costs = complex_RNN(n_input, n_hidden, n_output, input_type=input_type,
                                                out_every_t=out_every_t, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)

    elif (model == 'URNN'):
        inputs, parameters, costs, manifolds = URNN(n_input, n_hidden, n_output, input_type=input_type,
                                                    out_every_t=out_every_t, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)

    elif (model == 'IRNN'):
        inputs, parameters, costs = IRNN(n_input, n_hidden, n_output, input_type=input_type,
                                         out_every_t=out_every_t, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)
        gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]

    elif (model == 'RNN'):
        inputs, parameters, costs = tanhRNN(n_input, n_hidden, n_output, input_type=input_type,
                                            out_every_t=out_every_t, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)
        gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]
    
    else:
        print("Unsuported model:", model)
        return


    # --- Compile theano functions --------------------------------------------------

    index = T.iscalar('i')

    learning_rate = theano.shared(np.array(learning_rate, dtype=theano.config.floatX))


    if model == 'complex_RNN_momentum':
        manifolds = {}
        updates = nesterov_momentum(gradients, parameters, learning_rate, momentum=0.9, manifolds=manifolds)
        rmsprop = []
    if model == 'URNN':
        updates = nesterov_momentum(gradients, parameters, learning_rate, manifolds=manifolds)
        rmsprop = []
    else:
        updates, rmsprop = rms_prop(learning_rate, parameters, gradients)

    givens = {inputs[0] : s_train_x[:, n_batch * index : n_batch * (index + 1), :],
              inputs[1] : s_train_y[n_batch * index : n_batch * (index + 1), :]}

    givens_test = {inputs[0] : s_test_x,
                   inputs[1] : s_test_y}
    
    
    train = theano.function([index], costs[0], givens=givens, updates=updates)
    
    test = theano.function([], costs[0], givens=givens_test)


    # --- Training Loop ---------------------------------------------------------------

    train_mse_res = []
    test_mse_res = []

    train_loss = []
    test_loss = []
    best_params = [p.get_value() for p in parameters]
    best_rms = [r.get_value() for r in rmsprop]
    best_test_loss = 1e6
    print('Learning rate is {}'.format(learning_rate.get_value()))
    for i in range(n_iter):
        if False and model == "URNN":
            unitary_matrix = [p for p in parameters if p.name == "UNITARY"][0]
            print('How much of unitarity U holds?')
            uval = unitary_matrix.get_value()
            uval = uval[0, :, :] + 1j * uval[1, :, :]
            print("U norm {}".format(la.norm(uval)))
            print('Delta between U^*U and I: {}'.format(la.norm(np.conj(uval).T.dot(uval) - np.eye(uval.shape[0]))))
        if (n_iter % int(num_batches) == 0):
            inds = np.random.permutation(int(n_train))
            data_x = s_train_x.get_value()
            s_train_x.set_value(data_x[:,inds,:])
            data_y = s_train_y.get_value()
            s_train_y.set_value(data_y[inds,:])


        mse = train(i % int(num_batches))
        train_loss.append(mse)
        np.array(train_loss).tofile(savefile + '_train_loss.npfile')
        print("Iteration:", i)
        print("mse:", mse)
        print()

        # learn rate annealing
        if ((i + 1) % 100==0):
            learning_rate.set_value(learning_rate.get_value() * 0.9)
            print('Learning rate is decayed. It now becomes {}'.format(learning_rate.get_value()))



        if ((i + 1) % 25==0):
            mse = test()
            print()
            print("TEST")
            print("mse:", mse)
            print()
            test_loss.append(mse)
            np.array(test_loss).tofile(savefile + '_test_loss.npfile')


            if mse < best_test_loss:
                best_params = [p.get_value() for p in parameters]
                best_rms = [r.get_value() for r in rmsprop]
                best_test_loss = mse

            
            save_vals = {'parameters': [p.get_value() for p in parameters],
                         'rmsprop': [r.get_value() for r in rmsprop],
                         'train_loss': train_loss,
                         'test_loss': test_loss,
                         'best_params': best_params,
                         'best_rms': best_rms,
                         'best_test_loss': best_test_loss,
                         'model': model,
                         'time_steps': time_steps}

            pickle.dump(save_vals,
                         open(savefile, 'wb'),
                         pickle.HIGHEST_PROTOCOL)



        

    
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="training a model")
    parser.add_argument("n_iter", type=int, default=2000)
    parser.add_argument("n_batch", type=int, default=20)
    parser.add_argument("n_hidden", type=int, default=128)
    parser.add_argument("time_steps", type=int, default=200)
    parser.add_argument("learning_rate", type=float, default=0.001)
    parser.add_argument("savefile")
    parser.add_argument("model", default='complex_RNN')
    parser.add_argument("input_type", default='categorical')
    parser.add_argument("out_every_t", default='False')
    parser.add_argument("loss_function", default='MSE')
    
    args = parser.parse_args()
    dict = vars(args)

    kwargs = {'n_iter': dict['n_iter'],
              'n_batch': dict['n_batch'],
              'n_hidden': dict['n_hidden'],
              'time_steps': dict['time_steps'],
              'learning_rate': np.float32(dict['learning_rate']),
              'savefile': dict['savefile'],
              'model': dict['model'],
              'input_type': dict['input_type'],              
              'out_every_t': 'True'==dict['out_every_t'],
              'loss_function': dict['loss_function']}

    main(**kwargs)