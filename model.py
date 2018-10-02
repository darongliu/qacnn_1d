import torch
import torch.nn as nn
import torch.nn.functional as F

class multi_Conv1d(nn.Module):
    def __init__(self, input_channel, filter_num_list, filter_size_list, dropout):
        super(multi_Conv1d, self).__init__()
        assert(len(filter_num_list) == len(filter_size_list))
        self.conv_layers_list = []
        self.dropout_layers_list = []
        for idx in range(len(filter_num_list)):
            if idx == 0:
                self.conv_layers_list.append(nn.Conv1d(input_channel, filter_num_list[0], filter_size_list[0]))
            else:
                self.conv_layers_list.append(nn.Conv1d(filter_num_list[idx-1], filter_num_list[idx], filter_size_list[idx]))
            self.dropout_layers_list.append(nn.Dropout(dropout))
        self.layers = nn.ModuleList(self.conv_layers_list+self.dropout_layers_list)
    def forward(self, input_tensor):
        output = input_tensor
        for i in range(len(self.conv_layers_list)-1):
            output = self.conv_layers_list[i](output)
            output = F.relu(output)
            output = self.dropout_layers_list[i](output)
        return self.conv_layers_list[-1](output)

class qacnn_1d(nn.Module):
    def __init__(self, question_length, option_length, filter_num, filter_size, cnn_layers, dnn_size, dropout=0.1):
        super(qacnn_1d, self).__init__()
        self.question_length = question_length
        self.option_length = option_length
        self.dropout = dropout

        filter_num_list = [filter_num]*cnn_layers
        filter_size_list = [filter_size]*cnn_layers
        self.conv_first_att =  multi_Conv1d(question_length, filter_num_list, filter_size_list, dropout)
        self.conv_first_pq  =  multi_Conv1d(question_length, filter_num_list, filter_size_list, dropout)
        self.conv_first_pc  =  multi_Conv1d(option_length, filter_num_list, filter_size_list, dropout)

        self.linear_second_pq = nn.Linear(filter_num, filter_num)
        self.linear_second_pc = nn.Linear(filter_num, filter_num)

        self.linear_output_dropout = nn.Dropout(dropout)
        self.linear_output_1 = nn.Linear(filter_num, dnn_size)
        self.linear_output_2 = nn.Linear(dnn_size, 1)

    def forward(self, p, q, c):
        #get option num
        option_num = c.size()[1]
        p = p.unsqueeze(1).repeat(1,option_num,1,1)
        q = q.unsqueeze(1).repeat(1,option_num,1,1)
        #generate similarity map
        p = p.view([p.size()[0]*option_num, p.size()[2], p.size()[3]])
        q = q.view([q.size()[0]*option_num, q.size()[2], q.size()[3]])
        c = c.view([c.size()[0]*option_num, c.size()[2], c.size()[3]])

        pq_map = self.compute_similarity_map(p, q) # [batch x p_length x q_length]
        pc_map = self.compute_similarity_map(p, c) # [batch x p_length x c_length]
        pq_map = pq_map.permute([0,2,1]) # [batch x q_length x p_length]
        pc_map = pc_map.permute([0,2,1]) # [batch x c_length x p_length]

        #first stage
        first_att = torch.sigmoid(torch.max(self.conv_first_att(pq_map), dim=1)[0])
        first_representation_pq =  F.relu(torch.max(self.conv_first_pq(pq_map), dim=-1)[0])
        first_representation_pc =  F.relu(self.conv_first_pc(pc_map))*first_att.unsqueeze(1)
        first_representation_pc =  torch.max(first_representation_pc, dim=-1)[0]

        #second stage
        second_att = torch.sigmoid(self.linear_second_pq(first_representation_pq))
        second_pc  = F.relu(self.linear_second_pc(first_representation_pc))
        second_final_representation = second_att * second_pc

        #output
        output = self.linear_output_dropout(second_final_representation)
        output = self.linear_output_2(torch.tanh(self.linear_output_1(output)))
        output = output.view([-1, option_num])

        return F.softmax(output, dim=-1)

    def compute_similarity_map(self, a, b):
        '''
        a, b: [batch_size x length x dim]
        '''
        a_norm = F.normalize(a, dim=-1)
        b_norm = F.normalize(b, dim=-1)
        return torch.matmul(a_norm, b_norm.permute(0,2,1))

if __name__ == '__main__':
    question_length = 10
    option_length = 5
    test_passage_length = 20
    test_batch_size = 32
    test_word_dim = 300
    test_option_num = 5

    qacnn = qacnn_1d(question_length, option_length, 256, 3, 256)
    p = torch.rand([test_batch_size, test_passage_length, test_word_dim])
    print(p)
    print(p.type())
    q = torch.rand([test_batch_size, question_length, test_word_dim]).float()
    c = torch.rand([test_batch_size, test_option_num, option_length, test_word_dim]).float()
    result = qacnn(p, q, c)
    print(result)
