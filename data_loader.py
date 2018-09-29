import torch
import torch.utils.data
import json
import jieba
import numpy as np
import fastText
from utils.utils import text_norm_before_cut, text_norm_after_cut

class myDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, fasttext_path='./word2vec/model.bin'):
        super(myDataset).__init__()
        print('start loading word2vec model')
        self.model = fastText.FastText.load_model(fasttext_path)
        self.word_dim = self.model.get_dimension()
        print('start loading data')
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.processed_data = [self.process_data(data) for data in self.data]
        self.option_num = len(self.data[0]['options'])
        print('finish loading data')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.processed_data[idx]

    def process_data(self, one_data):
        return self._sent2np(one_data['context']), self._sent2np(one_data['question']), [self._sent2np(option) for option in one_data['options']], one_data['id'], one_data['answer']-1 # assume answer is 1 based

    @staticmethod
    def collate_fn(batch):
        def _pad_sequence(np_list):
            tensor_list = [torch.from_numpy(np_array) for np_array in np_list]
            return torch.nn.utils.rnn.pad_sequence(tensor_list, batch_first=True)
        # for dataloader
        all_context, all_question, all_option, all_id, all_answer = zip(*batch)
        option_num = len(all_option[0])
        flatten_option = [item for sublist in all_option for item in sublist]
        pad_option = _pad_sequence(flatten_option)
        pad_option = pad_option.view([int(pad_option.size()[0]/option_num),option_num,pad_option.size()[1], pad_option.size()[2]])

        return _pad_sequence(all_context), _pad_sequence(all_question), pad_option, all_id, torch.tensor(all_answer)

    def _sent2np(self, text):
        text = text_norm_before_cut(text)
        word_list = list(jieba.cut(text))
        #word_list = list(text)
        word_list = text_norm_after_cut(word_list)

        return np.array([self.model.get_word_vector(word) for word in word_list])

if __name__ == '__main__':
    path = 'data/result_kaldi2.json'
    dataset = myDataset(path)
    merge_option = myDataset.collate_fn([dataset.processed_data[0]])[0].size()
    print('merge option size', merge_option)
    print('option1', dataset.data[0]['options'])
    print('option2', dataset.data[1]['options'])
