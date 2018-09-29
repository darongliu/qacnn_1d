import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from utils import *

def train(model, data_loader, saver, total_epoch, lr, log_path=None, start_epoch=0):
    f_log = open(log_path, 'r')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    min_loss = np.inf

    for epoch in range(start_epoch, start_epoch+total_epoch):
        total_loss = 0.
        total_correct = 0
        total_count = 0

        for i, data in enumerate(data_loader):
            context, question, option, _, answer = data
            context, question, option, answer = put_to_cuda([context, question, option, answer])

            optimizer.zero_grad()
            output = model(context, question, option)
            loss = criterion(output, answer)
            loss.backward()
            optimizer.step()


            total_loss = total_loss + loss.cpu().numpy()[0]
            _, predict = torch.max(output)
            total_correct += (predict == answer).sum().cpu().numpy()[0]
            total_count += context.size()[0]

            average_loss = total_loss/total_count
            average_accuracy = total_correct/total_count

            log = f'batch: {i}/{len(data_loader)}, average loss: {average_loss}, average accuracy: {average_accuracy}'
            print_and_logging(f_log, log)

        #save model
        if min_loss > total_loss/total_count:
            log = f'average loss: {average_loss}, save model'
            print_and_logging(f_log, log)
            #save model
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'average_loss': average_loss,
                'average_accuracy': average_accuracy
            }
            name = f'epoch_{epoch}_average_accuracy_{average_accuracy}'
            saver.save(state, name)
        else:
            log = f'average loss: {average_loss}, higher loss!!!!!!'
            print_and_logging(f_log, log)

















