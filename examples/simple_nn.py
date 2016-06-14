import time
import theano
import lasagne
import numpy as np

import theano.tensor as T

from utils import custom_sgd, iterate_minibatches
from layers import UnitaryLayer, ComplexifyTransform, Modrelu


def build_custom_unn(input_var=None,
                     width=None,
                     num_hidden=2,
                     num_outputs=10,
                     final_nonlin=lasagne.nonlinearities.softmax,
                     drop_input=.2,
                     drop_hidden=.5):
    width = 10 if width is None else width
    manifolds = {}

    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)

    network = lasagne.layers.DenseLayer(network, width)
    network = ComplexifyTransform(network)
    #input_mapper = lasagne.layers.DenseLayer(network, widths[0], nonlinearity=lasagne.nonlinearities.rectify, b=None)
    for i in range(num_hidden):
        man_name = "unitary" + str(i)
        network = UnitaryLayer(network, name=man_name)
        manifolds[man_name] = network.manifold
        #network = lasagne.layers.NonlinearityLayer(network, nonlinearity=lasagne.nonlinearities.rectify)
        network = Modrelu(network)
    network = lasagne.layers.DenseLayer(network, num_units=num_outputs, nonlinearity=final_nonlin)
    return network, manifolds

def generate_train_acc(input_X=None, target_y=None, width=10, num_hidden=2):
    input_X = T.tensor4("X") if input_X is None else input_X
    target_y = T.vector("target Y integer", dtype='int32') if target_y is None else target_y
    dense_output, manifolds = build_custom_unn(input_X, width=width, num_hidden=num_hidden)

    y_predicted = lasagne.layers.get_output(dense_output)
    all_weights = lasagne.layers.get_all_params(dense_output)


    loss = lasagne.objectives.categorical_crossentropy(y_predicted,target_y).mean()
    accuracy = lasagne.objectives.categorical_accuracy(y_predicted,target_y).mean()

    updates_sgd = custom_sgd(loss, all_weights, learning_rate=1e-2, manifolds=manifolds)


    train_fun = theano.function([input_X,target_y],[loss, accuracy],updates=updates_sgd)
    accuracy_fun = theano.function([input_X,target_y],accuracy)
    return train_fun, accuracy_fun


def run(X_train, y_train, X_val, y_val, X_text, y_test):
    num_epochs = 20
    batch_size = 100

    train, acc = generate_train_acc(width=50, num_hidden=10)
    res = {}
    res["train_fun"] = train
    res["accuracy_fun"] = acc
    res["train_err"] = []
    res["train_acc"] = []
    res["epoch_times"] = []
    res["val_acc"] = []

    for epoch in range(num_epochs):
        train_err = 0
        train_acc = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train,batch_size):
            inputs, targets = batch
            train_err_batch, train_acc_batch= res["train_fun"](inputs, targets)
            train_err += train_err_batch
            train_acc += train_acc_batch
            train_batches += 1

        # And a full pass over the validation data:
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batch_size):
            inputs, targets = batch
            val_acc += res["accuracy_fun"](inputs, targets)
            val_batches += 1

        # Then we print the results for this epoch:
        print("for {}".format("unitary"))
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))

        print("  training loss (in-iteration):\t\t{:.6f}".format(train_err / train_batches))
        print("  train accuracy:\t\t{:.2f} %".format(
            train_acc / train_batches * 100))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        res["train_err"].append(train_err / train_batches)
        res["train_acc"].append(train_acc / train_batches * 100)
        res["val_acc"].append(val_acc / val_batches * 100)


if __name__ == "__main__":
    from mnist import load_dataset
    X_train,y_train,X_val,y_val,X_test,y_test = load_dataset()
    print(X_train.shape,y_train.shape)

    run(X_train,y_train,X_val,y_val,X_test,y_test)