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
                self.conv_layers_list.append(nn.Conv1d(filter_num_list[idx-1], filter_num_list[idx], filter_size_list[idx], padding=int((filter_size_list[idx]-1)/2)))
            self.dropout_layers_list.append(nn.Dropout(dropout))
        self.layers = nn.ModuleList(self.conv_layers_list+self.dropout_layers_list)
    def forward(self, input_tensor):
        output = input_tensor
        for i in range(len(self.conv_layers_list)-1):
            previous_output = output
            output = self.conv_layers_list[i](output)
            output = F.relu(output)
            output = self.dropout_layers_list[i](output)
            if i>0:
                output += previous_output
        return self.conv_layers_list[-1](output)
"""
class similarity_map(nn.Module):
    def __init__(self, word_dim, hidden_units=1, dropout=0.1):
        super(similarity_map, self).__init__()
        self.dropout = dropout
        self.linear_1_a = nn.Linear(word_dim, hidden_units)
        self.linear_1_b = nn.Linear(word_dim, hidden_units)
        #self.dropout_1 = nn.Dropout(dropout)
        #self.linear_2 = nn.Linear(hidden_units, 1)

    def forward(self, a, b):
        size_a = a.size()[1]
        size_b = b.size()[1]

        a_proj = self.linear_1_a(a)
        b_proj = self.linear_1_b(b)
        a_expand = a_proj.unsqueeze(2).repeat(1,1,size_b,1)
        b_expand = b_proj.unsqueeze(1).repeat(1,size_a,1,1)
        merge_tensor = a_expand + b_expand

        output = merge_tensor
        #output = F.tanh(output)
        #output = self.dropout_1(output)
        #output = self.linear_2(output)
        output = F.sigmoid(output.squeeze(-1))
        return output
"""

class qacnn_1d(nn.Module):
    def __init__(self, question_length, option_length, filter_num, filter_size, cnn_layers, dnn_size, word_dim, useful_feat_dim=27, dropout=0.1):
        super(qacnn_1d, self).__init__()
        self.question_length = question_length
        self.option_length = option_length
        self.dropout = dropout
        self.useful_feat_dim = useful_feat_dim

        #self.similarity_layer_pq = self.com(word_dim, dropout=dropout)
        #self.similarity_layer_pc = self.com(word_dim, dropout=dropout)
        #self.p_transform_1 = nn.Linear(word_dim, word_dim)
        #self.p_transform_2 = nn.Linear(word_dim, word_dim)
        #self.q_transform_1 = nn.Linear(word_dim, word_dim)
        #self.q_transform_2 = nn.Linear(word_dim, word_dim)
        #self.c_transform_1 = nn.Linear(word_dim, word_dim)
        #self.c_transform_2 = nn.Linear(word_dim, word_dim)

        filter_num_list = [filter_num]*cnn_layers
        filter_size_list = [filter_size]*cnn_layers
        self.conv_first_att =  multi_Conv1d(question_length, filter_num_list, filter_size_list, dropout)
        self.conv_first_pq  =  multi_Conv1d(question_length, filter_num_list, filter_size_list, dropout)
        self.conv_first_pc  =  multi_Conv1d(option_length, filter_num_list, filter_size_list, dropout)

        self.linear_second_pq = nn.Linear(filter_num, filter_num)
        self.linear_second_pc = nn.Linear(filter_num, filter_num)

        self.linear_output_dropout = nn.Dropout(dropout)
        self.linear_output_1 = nn.Linear(filter_num, dnn_size)
        self.linear_output_2 = nn.Linear(dnn_size+useful_feat_dim, 1)

    def forward(self, p, q, c, useful_feat):
        #get option num
        option_num = c.size()[1]

        #expand p and c for option_num times
        p_expand = p.unsqueeze(1).repeat(1,option_num,1,1)
        p_expand = p_expand.view([p_expand.size()[0]*option_num, p_expand.size()[2], p_expand.size()[3]])
        c_expand = c.view([c.size()[0]*option_num, c.size()[2], c.size()[3]])

        pq_map = self.compute_similarity_map(p, q) # [batch x p_length x q_length]
        pc_map_expand = self.compute_similarity_map(p_expand, c_expand) # [batch*option_num x p_length x c_length]
        pq_map = pq_map.permute([0,2,1]) # [batch x q_length x p_length]
        pc_map_expand = pc_map_expand.permute([0,2,1]) # [batch*option_num x c_length x p_length]

        #first stage
        first_att = torch.sigmoid(torch.max(self.conv_first_att(pq_map), dim=1)[0]) # [batch x p_length]
        first_att = first_att.unsqueeze(1).repeat(1,option_num,1)
        first_att = first_att.view([first_att.size()[0]*option_num, first_att.size()[2]])
        first_representation_pq =  F.relu(torch.max(self.conv_first_pq(pq_map), dim=-1)[0]) # [batch x channel]
        first_representation_pq = first_representation_pq.unsqueeze(1).repeat(1,option_num,1)
        first_representation_pq = first_representation_pq.view([first_representation_pq.size()[0]*option_num, first_representation_pq.size()[2]])

        first_representation_pc =  F.relu(self.conv_first_pc(pc_map_expand))*first_att.unsqueeze(1)
        first_representation_pc =  torch.max(first_representation_pc, dim=-1)[0]

        #second stage
        second_att = torch.sigmoid(self.linear_second_pq(first_representation_pq))
        second_pc  = F.relu(self.linear_second_pc(first_representation_pc))
        second_final_representation = second_att * second_pc

        #output
        useful_feat = useful_feat.view([useful_feat.size()[0]*useful_feat.size()[1], useful_feat.size()[2]])
        output = self.linear_output_dropout(second_final_representation)
        output = torch.tanh(self.linear_output_1(output))
        output = torch.cat([output, useful_feat], -1)
        output = self.linear_output_2(output)
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
    cnn_layers=2
    test_batch_size = 32
    test_word_dim = 300
    test_option_num = 5

    qacnn = qacnn_1d(question_length, option_length, 256, 3, cnn_layers, 256, test_word_dim)
    p = torch.rand([test_batch_size, test_passage_length, test_word_dim])
    print(p)
    print(p.type())
    q = torch.rand([test_batch_size, question_length, test_word_dim]).float()
    c = torch.rand([test_batch_size, test_option_num, option_length, test_word_dim]).float()
    result = qacnn(p, q, c)
    print(result.size())
