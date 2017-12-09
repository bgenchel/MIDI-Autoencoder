import numpy as np
import torch
from torch.autograd import Variable

def myMSELoss(size_average=True):
    def mse(output, target):
        assert type(output) == Variable
        assert type(target) == Variable
        assert output.size() == target.size()
        flat_output = output.view(-1, torch.prod(output.size()))
        flat_target = target.view(-1, torch.prod(target.size()))
        error = torch.sum(torch.pow(flat_output - flat_target, 2))
        if size_average:
            error /= len(flat_target)
        return error
    return mse


def WeightedBCELoss(size_average=True, one_weight=1, zero_weight=1):
    """
    parameters:
        size_average:  take the average of the loss over all elements
        one_weight:  the weight of the loss in the case that a 1 is misclassified
        zero_weight:  the weight of the loss in the case that a 0 is misclassified
    """
    def wbce(output, target):
        assert type(output) == Variable
        assert type(target) == Variable
        assert output.size() == target.size()
        epsilon = 1e-6
        ones = Variable(torch.ones(output.size()))
        one_error = one_weight*(target*torch.log(output + ones*epsilon))
        zero_error = zero_weight*((ones - target)*torch.log(ones - output + ones*epsilon))
        error = -torch.sum(one_error + zero_error)
        if size_average:
            n = np.prod(output.size())
            error /= n
        return error
    return wbce


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
