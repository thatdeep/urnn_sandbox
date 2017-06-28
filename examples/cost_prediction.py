import pickle
import gzip
import theano
import pdb
import numpy as np
import theano.tensor as T
from theano.ifelse import ifelse
from models import *
from utils.optimizations import *
import argparse, timeit


NP_FLOAT = np.float64
INT_STR = 'int64'
FLOAT_STR = 'float64'

def generate_data(time_steps, n_data, n_sequence):
    seq = np.random.randint(1, high=9, size=(n_data, n_sequence))
    zeros1 = np.zeros((n_data, time_steps-1))
    zeros2 = np.zeros((n_data, time_steps))
    marker = 9 * np.ones((n_data, 1))
    zeros3 = np.zeros((n_data, n_sequence))

    x = np.concatenate((seq, zeros1, marker, zeros3), axis=1).astype(FLOAT_STR)
    y = np.concatenate((zeros3, zeros2, seq), axis=1).astype(FLOAT_STR)

    return x.T, y.T