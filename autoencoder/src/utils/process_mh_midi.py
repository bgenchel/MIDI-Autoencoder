import argparse
import pretty_midi
import glob
import numpy as np
import os
import os.path as op

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--clip_len', default='40', type=int,
                    help="either simple or midi_net")
args = parser.parse_args()
clip_len = int(args.clip_len)

data_dir = op.expanduser('~/Dropbox/Data/MH_Dataset')
midi_files = glob.glob(op.join(data_dir, "*.mid"))

dataset = []
for i, f in enumerate(midi_files):
    midi_data = pretty_midi.PrettyMIDI(f)
    melody = midi_data.instruments[0]
    pr = melody.get_piano_roll()
    if pr.shape[1]%clip_len != 0:
        pr = pr[:, :-(pr.shape[1]%clip_len)]
    clips = np.split(pr, pr.shape[1]/clip_len, axis=1)
    for clip in clips:
        dataset.append([clip])
    if i > 0 and i%100 == 0:
        print('%d files processed'%i)

dataset = np.asarray(dataset)
dataset[dataset > 0] = 1
print('dataset shape: {}'.format(dataset.shape))
dataset.dump('../../../data/mh-midi-data.pickle')
