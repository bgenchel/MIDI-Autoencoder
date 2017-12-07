import argparse
import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--index', default=0, type=int,
                    help="datapoint to display, indexed from 0")
args = parser.parse_args()

dataset = pickle.load(open('../data/mh-midi-data.pickle', 'rb'))
dataset = dataset[:, :, 48:84, :]

# import pdb
# pdb.set_trace()
plt.imshow(dataset[args.index, 0, :, :])
plt.show()
