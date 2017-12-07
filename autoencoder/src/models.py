import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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

    # def forward(self, x):
    #     input_shape = x.size()
    #     h = self.encoder(x)
    #     d = self.decoder(h)
    #     x = x.view(input_shape)
    #     return x

    def encoder(self, x):
        x = x.view(x.size()[0], np.prod(x.size()[1:]))
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return h

    def decoder(self, h):
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
