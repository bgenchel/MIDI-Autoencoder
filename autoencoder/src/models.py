import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SimpleEncoder(nn.Module):

    def __init__(self, input_dim, **kwargs):
        super(SimpleEncoder, self).__init__(**kwargs)
        self.fc1 = nn.Linear(input_dim, input_dim/2)
        self.fc2 = nn.Linear(input_dim/2, input_dim/4)
        self.fc3 = nn.Linear(input_dim/4, input_dim/8)
        return

    def forward(self, inpt):
        inpt = inpt.view(inpt.size()[0], np.prod(inpt.size()[1:]))
        enc = F.relu(self.fc1(inpt))
        enc = F.relu(self.fc2(enc))
        enc = F.relu(self.fc3(enc))
        return enc

class SimpleDecoder(nn.Module):

    def __init__(self, input_dim, **kwargs):
        super(SimpleDecoder, self).__init__(**kwargs)
        self.fc4 = nn.Linear(input_dim, input_dim*2)
        self.fc5 = nn.Linear(input_dim*2, input_dim*4)
        self.fc6 = nn.Linear(input_dim*4, input_dim*8)
        self.sigmoid = nn.Sigmoid()
        return

    def forward(self, enc):
        enc = F.relu(self.fc4(enc))
        enc = F.relu(self.fc5(enc))
        dec = self.sigmoid(self.fc6(enc))
        return dec


class ConvEncoder(nn.Module):

    def __init__(self, input_dim, channels, **kwargs):
        super(ConvEncoder, self).__init__(**kwargs)
        c, h, w = input_dim
        self.channels = channels
        self.l1_kernel_size = (h, 2)
        self.l1_cout = self.channels[0]
        self.l2_kernel_size = (1, 4)
        self.l2_stride = 2
        self.l2_cout = self.channels[1]

        # n x c x h x w
        self.conv1 = nn.Conv2d(c, self.l1_cout, self.l1_kernel_size)
        # n x l1_cout x 1 x w - 1
        self.conv2 = nn.Conv2d(self.l1_cout, self.l2_cout, self.l2_kernel_size,
                               stride=self.l2_stride)
        # n x l2_cout x 1 x floor((w - 1)/2) - 1

        self.fc_input_height = 1
        self.fc_input_width = int(np.floor((w - 1)/2) - 1)
        # self.fc_input_width = w - 2
        self.fc1_input_dim = self.l2_cout*self.fc_input_height*self.fc_input_width
        self.fc2_input_dim = self.fc1_input_dim/2
        self.output_dim = self.fc2_input_dim/2

        self.fc1 = nn.Linear(self.fc1_input_dim, self.fc2_input_dim)
        # self.fc1 = nn.Linear(self.fc1_input_dim, self.output_dim)
        self.fc2 = nn.Linear(self.fc2_input_dim, self.output_dim)
        return

    def forward(self, inpt):
        # import pdb
        # pdb.set_trace()
        # dim is n x c x h x w
        enc = F.relu(self.conv1(inpt))
        # dim is n x l1_cout x 1 x w/2
        enc = F.relu(self.conv2(enc))
        # dim is n x l2_cout x 1 x w/4
        enc = enc.view(enc.size()[0], np.prod(enc.size()[1:]))
        enc = F.relu(self.fc1(enc))
        enc = F.relu(self.fc2(enc))
        return enc


class ConvDecoder(nn.Module):

    def __init__(self, input_dim, channels, output_dim, **kwargs):
        super(ConvDecoder, self).__init__(**kwargs)
        self.fc1 = nn.Linear(input_dim, input_dim*2)
        self.fc2 = nn.Linear(input_dim*2, input_dim*4)

        self.c, self.h, self.w = output_dim
        self.l1_cin = channels[0]
        self.l1_kernel_size = (1, 4)
        self.l1_stride = 2
        self.l1_cout = channels[1]
        self.l2_kernel_size = (self.h, 2)
        self.l2_cout = 1

        self.tconv1 = nn.ConvTranspose2d(self.l1_cin, self.l1_cout, self.l1_kernel_size,
                                         stride=self.l1_stride, output_padding=(0, 1))
        self.tconv2 = nn.ConvTranspose2d(self.l1_cout, self.l2_cout, self.l2_kernel_size)
        return

    def forward(self, enc):
        # import pdb
        # pdb.set_trace()
        enc = F.relu(self.fc1(enc))
        enc = F.relu(self.fc2(enc))
        enc = enc.view(enc.size()[0], self.l1_cin, 1, -1)
        enc = F.relu(self.tconv1(enc))
        dec = self.tconv2(enc)
        return dec


# class SimpleAutoencoder(nn.Module):
#     def __init__(self, init_dim, **kwargs):
#         super(SimpleAutoencoder, self).__init__(**kwargs)
#         self.init_dim = init_dim
#         self.fc1 = nn.Linear(init_dim, init_dim/2)
#         self.fc2 = nn.Linear(init_dim/2, init_dim/4)
#         self.fc3 = nn.Linear(init_dim/4, init_dim/8)
#         self.fc4 = nn.Linear(init_dim/8, init_dim/4)
#         self.fc5 = nn.Linear(init_dim/4, init_dim/2)
#         self.fc6 = nn.Linear(init_dim/2, init_dim)
#         self.sigmoid = nn.Sigmoid()
#         return

#     def forward(self, x):
#         input_shape = x.size()
#         x = x.view(x.size()[0], np.prod(x.size()[1:]))
#         h = F.relu(self.fc1(x))
#         h = F.relu(self.fc2(h))
#         h = F.relu(self.fc3(h))
#         h = F.relu(self.fc4(h))
#         h = F.relu(self.fc5(h))
#         d = self.fc6(h)
#         return d


class MIDINetAutoencoder(nn.Module):
    def __init__(self):
        super(MIDINetAutoencoder, self).__init__()
        return

    def forward(self, x):
        pass
