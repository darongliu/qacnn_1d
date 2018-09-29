import torch
import torch.nn as nn
import torch.nn.functional as F

class qacnn_1d(nn.module):
    def __init__(self, word_dimension, filter_num_1, filter_size_1, dnn_size, dropout, choice_num):
        super(qacnn_1d, self).__init__()
        self.word_dimension = word_dimension
        self.choice_num = choice_num
        self.dropout = dropout

        self.conv_first_att = nn.conv1d(word_dimension, filter_num_1, filter_size_1)
        self.conv_first_pq  = nn.conv1d(word_dimension, filter_num_1, filter_size_1)
        self.conv_first_pc  = nn.conv1d(word_dimension, filter_num_1, filter_size_1)

        self.linear_second_pq = nn.Linear(filter_size_1, filter_size_1)
        self.linear_second_pc = nn.Linear(filter_size_1, filter_size_1)

        self.linear_output_1 = nn.Linear(dnn_size, dnn_size)
        self.linear_output_2 = nn.Linear(dnn_size, 1)

    def forward(self, p, q, c):
        #generate similarity map
        c = c.view([c.size()[0]*self.choice_num, c.size()[2], c.size()[3]])

        pq_map = compute_similarity_map(p, q)
        pc_map = compute_similarity_map(p, c)

        #first stage
        first_att = F.sigmoid(torch.argmax(self.conv_first_att(pq_map), dim=-1))
        first_representation_pq =  F.relu(torch.argmax(self.conv_first_pq(pq_map), dim=1))
        first_representation_pc =  self.conv_first_pc(pc_map)*first_att.unsqueeze(-1)
        first_representation_pc =  F.relu(torch.argmax(first_representation_pc, dim=1))

        #second stage
        second_att = F.sigmoid(self.linear_second_pq(first_representation_pq))
        second_pc  = F.relu(self.linear_second_pc(first_representation_pc))
        second_final_representation = second_att * second_pc

        #output
        output = self.linear_output_2(F.tanh(self.linear_output_2(second_final_representation)))
        output = output.view([-1, self.choice_num])

        return F.softmax(output)

    def compute_similarity_map(a, b):
        a_norm = F.normalize(a, dim=-1)
        b_norm = F.normalize(b, dim=-1)
        return torch.matmul(a_norm, b_norm.permute(0,2,1))
