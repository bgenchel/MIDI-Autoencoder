from __future__ import print_function
import cPickle as pickle
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from reverse_pianoroll import piano_roll_to_pretty_midi as pr2pm

N = 1000
# load a NxMxC dataset
    # N: Number of clips
    # M: Piano roll size, the number of midi notes that could possibly be 'on'
    # C: Clip length, in 100ths of a second
dataset = pickle.load(open('mh-midi-data.pickle', 'rb'))
######## take a subset of the data for training ######
# based on the mean and standard deviation of non zero entries in the data, I've
# found that the most populous, and thus best range of notes to take is from
# 48 to 84 (C2 - C5); this is 3 octaves, which is much less than the original
# 10 and a half. Additionally, we're going to take a subsample of 1000 because
# i'm training on my macbook and the network is pretty simple
######################################################
dataset = dataset[:, :, 48:84, :]
dataset = dataset[:N]
######################################################

midi_dim, clip_len = dataset.shape[2:]

class VAE(nn.Module):
    def __init__(self, **kwargs):
        super(VAE, self).__init__(**kwargs)
        pass

    def forward(self, x):
        pass


