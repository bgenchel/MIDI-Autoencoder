import pretty_midi
import numpy as np
import os
import os.path as op
import re

clip_len = 20

data_dir = op.expanduser('~/Dropbox/Data/piano-midi.de')
midi_files = os.listdir(data_dir)
chpn_pattern = re.compile(r'chpn.*\.mid')
chpn_files = [op.join(op.expanduser(data_dir), f)
              for f in midi_files if chpn_pattern.match(f) is not None]

dataset = []
for f in chpn_files:
    midi_data = pretty_midi.PrettyMIDI(f)
    n2i = {inst.name: inst for inst in midi_data.instruments}
    if 'Piano right' not in n2i.keys():
        continue

    rh = n2i['Piano right']
    pr = rh.get_piano_roll()
    pr = pr[:, :-(pr.shape[1]%20)]
    clips = np.split(pr, pr.shape[1]/20, axis=1)
    dataset.append(np.asarray(clips))

dataset = np.asarray(dataset)
dataset.dump('chpn-data.pickle')
