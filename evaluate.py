import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from utils.utils import *

def evaluate(model, data_loader):
    model.eval()
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            context, question, option, _, answer = data
            context, question, option, answer = put_to_cuda([context, question, option, answer])

            output = model(context, question, option)
            _, predict = torch.max(output, 1)
            total_correct += (predict == answer).sum().detach().cpu().numpy()
            total_count += context.size()[0]

    return total_correct/total_count
