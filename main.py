import argparse
import torch
import sys

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
parser.add_argument('-b', '--batch-size', default=20, type=int,
                    metavar='N', help='mini-batch size (default: 20)')
parser.add_argument('-lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')

parser.add_argument('-filter_num', '--filter_num', default=128, type=int,
                    help='the number of the filter (default: 128)')
parser.add_argument('-filter_size', '--filter_size', default=3, type=int,
                    help='the size of the filter (default: 3)')
parser.add_argument('-dnn_size', '--dnn_size', default=128, type=int,
                    help='the size of the dnn layer (default: 128)')
parser.add_argument('-dr', '--dropout', default=0.2, type=float,
                    metavar='DR', help='dropout rate (default: 0.2)')

parser.add_argument('--data', default='', type=str, metavar='PATH',
                    help='The path of the training or testing data (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--test-result', dest='test_result',
                    help='The output path of the inference result',
                    default='', type=str)

def main(args):
    dataset = myDataset(args.data)
    if args.pos1 == 'train':
        #prepare dataloader
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch-size,
                                                  shuffle=True, num_workers=4, collate_fn=myDataset.collate_fn)
        saver = pytorch_saver(10, args.save_dir)
        #build model
        model = qacnn_1d(dataset.word_dim, args.filter_num, args.filter_size,
                         args.dnn_size, args.dropout, dataset.choice_num)
        if args.resume:
            model.load_state_dict(pytorch_saver.load_dir(args.resume)['state_dict'])

        model.train()
        model.cuda()
        train(model, data_loader, saver, args.epochs, args.lr)


    else:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                  shuffle=False, num_workers=1)
        if args.resume == '':
            print("resume should exist in inference mode", file=sys.stderr)
            sys.exit(-1)
        else:
            model = qacnn_1d(dataset.word_dim, args.filter_num, args.filter_size,
                             args.dnn_size, args.dropout, dataset.choice_num)
            model.load_state_dict(pytorch_saver.load_dir(args.resume)['state_dict'])
            model.eval()
            model.cuda()

            inference(model, data_loader, output_path)



