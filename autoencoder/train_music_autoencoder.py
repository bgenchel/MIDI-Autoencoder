"""
Most absolutely simple assumptions:
    - not changing the key of any of the files
    - not changing the tempo of any of the files

- take blocks of 36 by ___
- divide up all songs by this amount, cutting off any excess from the end, train
"""
from __future__ import print_function
import argparse
import cPickle as pickle
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as op
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from datetime import datetime
from pprint import pprint
from src.custom_loss import myMSELoss, WeightedBCELoss
from src.dataloaders import MHDataLoader
from src.models import SimpleEncoder, SimpleDecoder, ConvEncoder, ConvDecoder
from src.utils.reverse_pianoroll import piano_roll_to_pretty_midi as pr2pm

########## helper functions ##########
def plot_series(series, show=False, save=True, title="", fpath="series_plot"):
    plt.plot(series)
    plt.title(title)
    if show:
        plt.show()
        # plt.waitforbuttonpress()
    if save:
        plt.savefig(fpath)
    plt.close()
    return


def compute_avg_loss(batched_data):
    total_loss = 0.0
    for batch in batched_data:
        inpt = Variable(torch.FloatTensor(batch))
        # inpt = inpt.view(inpt.size()[0], np.prod(inpt.size()[1:]))
        output = decoder(encoder(inpt))
        output = output.view(batch.shape)
        loss = loss_fn(output, inpt)
        total_loss += loss.data[0]
    avg_loss = total_loss/len(batched_data)
    return avg_loss
##############################

np.random.seed(1)

MIDI_LOW = 48
MIDI_HIGH = 84

run_datetime_str = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
info_dict = {'midi_low': MIDI_LOW,
             'midi_high': MIDI_HIGH,
             'run_datetime': run_datetime_str}

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--architecture', default='simple',
                    choices=('simple', 'conv', 'midinet'),
                    help="either simple or midi_net")
parser.add_argument('-e', '--epochs', default=2, type=int,
                    help="number of training epochs")
parser.add_argument('-b', '--batch_size', default=10, type=int,
                    help="number of training epochs")
parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float,
                    help="learning rate for sgd")
parser.add_argument('-m', '--momentum', default=0.9, type=float,
                    help="learning rate for sgd")
parser.add_argument('-n', '--dataset_size', default=1000, type=int,
                    help="number of training samples to include in training")
parser.add_argument('-k', '--keep', action='store_true',
                    help="save information about this run")
parser.add_argument('-s', '--show', action='store_true',
                    help="show figures as they are generated")
parser.add_argument('-t', '--title', default=run_datetime_str, type=str,
                    help="custom title for run data directory")
args = parser.parse_args()
info_dict.update(vars(args))

########## load full NxCxMxL dataset #########
    # N: Number of clips
    # C: Number of 'channels' for interpretation as an image (1)
    # M: Piano roll size, the number of midi notes that could possibly be 'on'
    # L: Clip length, in 100ths of a second
dataset = pickle.load(open('../data/mh-midi-data.pickle', 'rb'))
##############################

######## divide into train, validation, and test ########
# based on the mean and standard deviation of non zero entries in the data, I've
# found that the most populous, and thus best range of notes to take is from
# 48 to 84 (C2 - C5); this is 3 octaves
np.random.shuffle(dataset)
dataset = dataset[:, :, MIDI_LOW:MIDI_HIGH, :]

train = dataset[:int(len(dataset)*0.8)]
train_dataloader = MHDataLoader(train, args.dataset_size)
batched_train = train_dataloader.get_batched_data(args.batch_size)

valid = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
valid_dataloader = MHDataLoader(valid, int(0.1*args.dataset_size))
batched_valid = valid_dataloader.get_batched_data(args.batch_size)

test = dataset[int(len(dataset)*0.9):]
test_dataloader = MHDataLoader(test, int(0.1*args.dataset_size))
batched_test = test_dataloader.get_batched_data(args.batch_size)
##############################

midi_dim, clip_len = train.shape[2:]
flattened_size = midi_dim*clip_len
encoded_size = flattened_size/8  # replace this hard coding with something more flexible
print("Training Set:")
print("\tnumber of clips: %d"%train.shape[0])
print("\tpiano roll size: %d"%midi_dim)
print("\tclip length: %d"%clip_len)

if args.architecture == "simple":
    encoder = SimpleEncoder(flattened_size)
    decoder = SimpleDecoder(encoded_size)
elif args.architecture == "conv":
    enc_channels = [8, 16]
    dec_channels = enc_channels[::-1]
    encoder = ConvEncoder(train[0].shape, enc_channels)
    decoder = ConvDecoder(encoder.output_dim, dec_channels, train[0].shape)
else:
    raise ValueError("Only SimpleAutoencoder is implemented.")
    # net = MIDINetAutoencoder()

params = itertools.chain(encoder.parameters(), decoder.parameters())
optimizer = optim.Adam(params, lr=args.learning_rate)
# optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum)
# optimizer = optim.Adagrad(net.parameters(), lr=args.learning_rate)

# loss_fn = nn.MSELoss(size_average=False)
# loss_fn = myMSELoss(size_average=False)
# loss_fn = SpiralLoss()
# loss_fn = nn.L1Loss()
loss_fn = WeightedBCELoss(size_average=True, one_weight=10)
# loss_fn = nn.BCELoss()

try:
    interrupted = False
    train_losses = []
    valid_losses = []
    print_every = 1
    print("Beginning Training")
    for epoch in xrange(args.epochs):
        batch_count = 0
        avg_loss = 0.0
        epoch_loss = 0.0
        for batch in batched_train:
            # get the data, wrap it in a Variable
            inpt = Variable(torch.FloatTensor(batch))
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass
            output = decoder(encoder(inpt))
            output = output.view(batch.shape)
            # backward + optimize
            loss = loss_fn(output, inpt)
            loss.backward()
            optimizer.step()
            # print stats out
            avg_loss += loss.data[0]
            epoch_loss += loss.data[0]
            if batch_count % print_every == print_every - 1:
                print('epoch: %d, batch_count: %d, loss: %.5f'%(
                    epoch + 1, batch_count + 1, avg_loss / print_every))
                avg_loss = 0.0
            batch_count += 1
        print('Average Epoch Loss: %f'%(epoch_loss/batch_count))
        train_losses.append(epoch_loss/batch_count)
        valid_loss = compute_avg_loss(batched_valid)
        valid_losses.append(valid_loss)
        print('Validation Loss: %.5f'%valid_loss)
    print('Finished Training')
except KeyboardInterrupt:
    print('Training Interrupted')
    interrupted = True

test_loss = compute_avg_loss(batched_test)
print('Test Loss: %.5f'%test_loss)

info_dict.update({'interrupted': interrupted,
                  'epochs_completed': len(train_losses),
                  'final_training_loss': train_losses[-1],
                  'final_validation_loss': valid_losses[-1],
                  'test_loss': test_loss})

########## make a new directory to store data from this run ##########
dirpath = op.join('runs', args.title)
if args.keep:
    if op.isdir(dirpath):
        raise IOError("path to run data directory already exists.")
    os.makedirs(dirpath)
##############################

plot_series(train_losses, show=args.show, save=args.keep,
            title="Training Losses", fpath=op.join(dirpath, 'train_losses'))
plot_series(valid_losses, show=args.show, save=args.keep,
            title="Validation Losses", fpath=op.join(dirpath, 'valid_losses'))

# output some autencoded input. Guess that 8 clips is about one song, since 1 clip
# should be about 1 second on audio.
num_clips = (100/clip_len)*8 # 8 sec ~ 1 song
# num_clips = args.dataset_size
song = train[:num_clips]
song_array = np.concatenate(song[:, 0, :, :], axis=1)
song_var = Variable(torch.FloatTensor(song))
decoded_song_var = decoder(encoder(song_var))
decoded_song = decoded_song_var.view(song.shape).data.numpy()
decoded_song_array = np.concatenate(decoded_song[:, 0, :, :], axis=1)

fig = plt.figure()
fig.suptitle("Autoencoding of Training Data")
top = fig.add_subplot(2, 1, 1)
top.set_title("Input")
top.imshow(song_array, aspect='auto')
bottom = fig.add_subplot(2, 1, 2)
bottom.set_title("Output")
bottom.imshow(decoded_song_array, aspect='auto')
fig.subplots_adjust(top=0.95)
fig.tight_layout()
if args.show:
    plt.show()
if args.keep:
    plt.savefig(op.join(dirpath, 'output_comparison'))
plt.close()

if args.keep:
    print('Creating output midi file ...')
    midi_array = np.zeros((128, clip_len*num_clips))
    midi_array[MIDI_LOW:MIDI_HIGH] = decoded_song_array
    outmidi = pr2pm(midi_array.round())

    print('Writing output midi file ...')
    outmidi.write(op.join(dirpath, 'autoencoded_midi.mid'))

    print('Writing run info file ...')
    with open(op.join(dirpath, 'info.txt'), 'wb') as fp:
        for k, v in info_dict.iteritems():
            fp.write('%s:\t %s\n'%(str(k), str(v)))
        fp.close()

    print('Saving models ...')
    torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()},
               op.join(dirpath, 'model_state.pt'))
