import torch
import torch.nn as nn
from Dataset_Dataloader import *
import torch.nn.functional as F


class TangShiRNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 初始化词嵌入层
        self.ebd = nn.Embedding(vocab_size, 128)
        # 循环网络层
        self.rnn = nn.RNN(128, 128, 1)
        # 输出层
        self.out = nn.Linear(128, vocab_size)

    def forward(self, inputs, hidden):

        embed = self.ebd(inputs)

        # 正则化层
        embed = F.dropout(embed, p=0.2)

        output, hidden = self.rnn(embed.transpose(0, 1), hidden)

        # 正则化层
        embed = F.dropout(output, p=0.2)

        output = self.out(output.squeeze())

        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 64, 128)