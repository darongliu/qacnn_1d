import argparse
import torch
import sys
import os

from model import qacnn_1d
from data_loader import myDataset
from saver import pytorch_saver

from train import train
from inference import inference

parser = argparse.ArgumentParser(description='PyTorch 1D QACNN')
parser.add_argument('pos1', default='train', type=str,
                    help='train or test (default: train)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run (default: 50)')
parser.add_argument('-b', '--batch_size', default=20, type=int,
                    metavar='N', help='mini-batch size (default: 20)')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')

parser.add_argument('-filter_num', '--filter_num', default=256, type=int,
                    help='the number of the filter (default: 256)')
parser.add_argument('-filter_size', '--filter_size', default=3, type=int,
                    help='the size of the filter (default: 3)')
parser.add_argument('-cnn_layers', '--cnn_layers', default=3, type=int,
                    help='the number of the cnn layers(default: 3)')
parser.add_argument('-dnn_size', '--dnn_size', default=256, type=int,
                    help='the size of the dnn layer (default: 256)')
parser.add_argument('-dr', '--dropout', default=0.1, type=float,
                    metavar='DR', help='dropout rate (default: 0.1)')

parser.add_argument('--question_length', default=30, type=int,
                    help='the length of question (default: 20)')
parser.add_argument('--option_length', default=35, type=int,
                    help='the length of option (default: 3)')

parser.add_argument('--train_data', default='', type=str, metavar='PATH',
                    help='The path of the training (default: none)')
parser.add_argument('--dev_data', default='', type=str, metavar='PATH',
                    help='The path of the dev (default: none)')
parser.add_argument('--test_data', default='', type=str, metavar='PATH',
                    help='The path of the testing (default: none)')
parser.add_argument('--resume_dir', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--log',help='training log file',
                    default='./log', type=str)
parser.add_argument('--test_result', dest='test_result',
                    help='The output path of the inference result',
                    default='', type=str)

def main(args):
    if args.pos1 == 'train':
        train_dataset = myDataset(args.train_data)
        dev_dataset = myDataset(args.dev_data)
        #prepare dataloader
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=myDataset.get_collate_fn(args.question_length, args.option_length))
        dev_data_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size, num_workers=4, collate_fn=myDataset.get_collate_fn(args.question_length, args.option_length))
        saver = pytorch_saver(10, args.save_dir)
        #build model
        model = qacnn_1d(args.question_length, args.option_length, args.filter_num, args.filter_size, args.cnn_layers, args.dnn_size, train_dataset.word_dim, dropout=args.dropout)
        if args.resume_dir != '':
            model.load_state_dict(pytorch_saver.load_dir(args.resume_dir)['state_dict'])

        model.train()
        model.cuda()
        args.log = os.path.join(args.save_dir, args.log)
        train(model, train_data_loader, dev_data_loader, saver, args.epochs, args.learning_rate, args.log)


    else:
        test_dataset = myDataset(args.test_data)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                  shuffle=False, num_workers=1)
        if args.resume_dir == '':
            print("resume should exist in inference mode", file=sys.stderr)
            sys.exit(-1)
        else:
            model = qacnn_1d(args.question_length, args.option_length, args.filter_num, args.filter_size, args.cnn_layers, args.dnn_size, test_dataset.word_dim, dropout=args.dropout)
            model.load_state_dict(pytorch_saver.load_dir(args.resume_dir)['state_dict'])
            model.eval()
            model.cuda()

            inference(model, test_data_loader, args.test_result)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


#filter need deeper
#add dropout
