import torch
from torch.autograd import Variable

def myMSELoss(size_average=True):
    def mse(inpt, outpt):
        assert inpt.size() == outpt.size()
        flat_in = inpt.view(-1, torch.prod(inpt.size()))
        flat_out = outpt.view(-1, torch.prod(outpt.size()))
        error = torch.sum(torch.pow(flat_in - flat_out, 2))
        if size_average:
            error /= len(flat_in)
        return error
    return mse

def SpiralLoss():
    def spiral_loss(input, output):
        loss = Variable(torch.FloatTensor([0]))
        d = 5
        r = 10
        for i in xrange(input.size()[0]):
            for j in xrange(input.size()[3]):
                # take along the 1 axis because it's a column vector
                inval, inind = torch.max(input[i, :, :, j], 1)
                outval, outind = torch.max(output[i, :, :, j], 1)
                note_loss = (r*30*(inind%12 - outind%12)).float()
                octave_loss = (d*(inind/12 - outind/12)).float()
                loss += torch.sqrt(torch.pow(note_loss, 2) + torch.pow(octave_loss, 2))
        return loss
    return spiral_loss
