import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from evaluate import evaluate
from utils.utils import *

def train(model, train_data_loader, dev_data_loader, saver, total_epoch, lr, log_path, start_epoch=0):
    f_log = open(log_path, 'w')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    max_dev_acc = 0

    for epoch in range(start_epoch, start_epoch+total_epoch):
        model.train()
        total_loss = 0.
        total_correct = 0
        total_count = 0

        for i, data in enumerate(train_data_loader):
            context, question, option, _, answer, useful_feat = data
            context, question, option, answer, useful_feat = put_to_cuda([context, question, option, answer, useful_feat])

            optimizer.zero_grad()
            output = model(context, question, option, useful_feat)
            loss = criterion(output, answer)
            loss.backward()
            optimizer.step()


            total_loss = total_loss + loss.detach().cpu().numpy()
            _, predict = torch.max(output, 1)
            total_correct += (predict == answer).sum().detach().cpu().numpy()
            total_count += context.size()[0]

            average_loss = total_loss/total_count
            average_accuracy = total_correct/total_count

            log = f'batch: {i}/{len(train_data_loader)}, train average loss: {average_loss}, train average accuracy: {average_accuracy}'
            print_and_logging(f_log, log)

        #evaluate on dev data
        print('start evalating')
        dev_acc = evaluate(model, dev_data_loader)
        log = f'dev accuracy: {dev_acc}'
        print_and_logging(f_log, log)

        if max_dev_acc <= dev_acc:
            max_dev_acc = dev_acc
            log = f'save model'
            print_and_logging(f_log, log)
            #save model
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_average_loss': average_loss,
                'train_average_accuracy': average_accuracy,
                'dev_acc': dev_acc
            }
            name = f'epoch_{epoch}_dev_accuracy_{dev_acc}'
            saver.save(state, name)
        else:
            log = f'higher loss!!!!!!'
            print_and_logging(f_log, log)
    log = 'training end, max dev acc: ' + str(max_dev_acc)
    print_and_logging(f_log, log)
    '''
        #save model
        if min_loss > average_loss:
            min_loss = average_loss
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
    '''

















