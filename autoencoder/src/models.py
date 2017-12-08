import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SimpleEncoder(nn.Module):

    def __init__(self, init_dim, **kwargs):
        super(SimpleEncoder, self).__init__(**kwargs)
        self.init_dim = init_dim
        self.fc1 = nn.Linear(init_dim, init_dim/2)
        self.fc2 = nn.Linear(init_dim/2, init_dim/4)
        self.fc3 = nn.Linear(init_dim/4, init_dim/8)
        return

    def forward(self, inpt):
        encoded = F.relu(self.fc1(inpt))
        encoded = F.relu(self.fc2(encoded))
        encoded = F.relu(self.fc3(encoded))
        return encoded

class SimpleDecoder(nn.Module):

    def __init__(self, init_dim, **kwargs):
        super(SimpleDecoder, self).__init__(**kwargs)
        self.fc4 = nn.Linear(init_dim, init_dim*2)
        self.fc5 = nn.Linear(init_dim*2, init_dim*4)
        self.fc6 = nn.Linear(init_dim*4, init_dim*8)
        self.sigmoid = nn.Sigmoid()
        return

    def forward(self, encoded):
        encoded = F.relu(self.fc4(encoded))
        encoded = F.relu(self.fc5(encoded))
        decoded = self.fc6(encoded)
        return decoded


class SimpleAutoencoder(nn.Module):
    def __init__(self, init_dim, **kwargs):
        super(SimpleAutoencoder, self).__init__(**kwargs)
        self.init_dim = init_dim
        self.fc1 = nn.Linear(init_dim, init_dim/2)
        self.fc2 = nn.Linear(init_dim/2, init_dim/4)
        self.fc3 = nn.Linear(init_dim/4, init_dim/8)
        self.fc4 = nn.Linear(init_dim/8, init_dim/4)
        self.fc5 = nn.Linear(init_dim/4, init_dim/2)
        self.fc6 = nn.Linear(init_dim/2, init_dim)
        self.sigmoid = nn.Sigmoid()
        return

    def forward(self, x):
        input_shape = x.size()
        x = x.view(x.size()[0], np.prod(x.size()[1:]))
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        h = F.relu(self.fc5(h))
        d = self.fc6(h)
        return d


class MIDINetAutoencoder(nn.Module):
    def __init__(self):
        super(MIDINetAutoencoder, self).__init__()
        return

    def forward(self, x):
        pass
