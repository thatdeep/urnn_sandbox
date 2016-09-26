import theano
import numpy as np
import scipy as sp

from theano import tensor as T
from numpy import random as rnd, linalg as la
from utils.fftconv import cufft, cuifft

if __name__ == "__main__":
    X = T.tensor3()

    m, n = 10, 5
    data = rnd.normal(size=(m, n, 2))

    poker = X.dimshuffle(1, 0, 2)
    stage1 = theano.function([X], poker)(data)
    print(poker.ndim)
    poker = cufft(poker)
    stage2 = theano.function([X], poker)(data)
    print(poker.ndim)
    poker = cufft(poker)
    stage3 = theano.function([X], poker)(data)
    print(poker.ndim)
    poker = cufft(poker)
    stage4 = theano.function([X], poker)(data)
    print(poker.ndim)
    poker = poker.dimshuffle(0, 2, 1)
    stage5 = theano.function([X], poker)(data)
    print(poker.ndim)

    print("after X.dimshuffle(1, 0, 2):")
    print(stage1.shape)
    print("after {}:".format("cufft(poker)"))
    print(stage2.shape)
    print("after {}:".format("cufft(poker)"))
    print(stage3.shape)
    print("after {}:".format("cufft(poker)"))
    print(stage4.shape)
    print("after {}:".format("poker.dimshuffle(0, 2, 1)"))
    print(stage5.shape)

    fft_res = cufft(X)
    fft_sum = T.sum(fft_res**2)
    simple_sum = T.sum(X**2)

    simple_grad = theano.grad(simple_sum, [X])[0]


    fft_res_num = theano.function([X], fft_res)(data)
    simple_grad_num = theano.function([X], simple_grad)(fft_res_num)
    print(simple_grad_num.shape)
    simple_data_back = theano.function([X], cuifft(X))(simple_grad_num)
    print(simple_data_back.shape)

    target_grad = theano.grad(fft_sum, [X])[0]
    target_grad_func = theano.function([X], target_grad)

    target_grad_num = target_grad_func(data)

    #print(simple_data_back)
    print('-'*80)
    #print(target_grad_num)
    print('-'*80)
    print(np.allclose(simple_data_back, target_grad_num))

"""
    fftx = fft(X, n)
    fftx3 = ifft(X, n)

    gf = theano.grad(T.sum(fftx), [X])
    gff = theano.function([X], gf)

    f3 = theano.function([X], fftx3)

    data = rnd.normal(size=(m, n, 2))

    print(data)
    print('-'*80)
    gff(data)
    print(gff(data))
    print('-'*80)
    f3(data)
    print(f3(data))
"""