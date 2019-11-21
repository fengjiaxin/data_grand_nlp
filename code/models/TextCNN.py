#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-11-18 19:45
# @Author  : 冯佳欣
# @File    : TextCNN.py
# @Desc    : TextCNN model 文件

import torch
import torch.nn as nn
from .BasicModule import BasicModule

kernel_sizes = [1,2,3,4,5]

class TextCNN(BasicModule):
    def __init__(self,hp,vectors=None):
        super(TextCNN,self).__init__()

        self.embedding = nn.Embedding(hp.vocab_size,hp.embedding_dim)
        if vectors is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(vectors))

        convs = [
            nn.Sequential(
                nn.Conv1d(in_channels=hp.embedding_dim,
                          out_channels=hp.kernel_num,
                          kernel_size=kernel_size),
                nn.BatchNorm1d(hp.kernel_num),
                nn.ReLU(inplace=True),

                # 到这里 [batch_size,kernel_num,(seq_len - kernel_size + 1)]
                nn.MaxPool1d(kernel_size=(hp.max_seq_len - kernel_size + 1))
                # [batch_size,kernel_num,1]
            )
            for kernel_size in kernel_sizes
        ]
        self.convs = nn.ModuleList(convs)

        self.fc = nn.Sequential(
            nn.Linear(len(kernel_sizes) * hp.kernel_num,hp.linear_hidden_size),
            nn.BatchNorm1d(hp.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hp.linear_hidden_size,hp.label_size)
        )

    def forward(self,inputs):
        '''
        :param inputs: [batch_size,max_seq_len]
        :return:
        '''
        # [batch_size,max_seq_len,embedding_dim]
        embeds = self.embedding(inputs)
        # 交换维度，换成 [batch_size,embedding_dim,max_seq_len]
        embeds = embeds.permute(0,2,1)

        # a list of [batch_size,kernel_num,1] length = len(kernel_sizes)
        conv_out = [cov(embeds) for cov in self.convs]

        # 沿着横着的方向连接
        # [batch_size,kernel_num,len(kernel_sizes)]
        conv_out = torch.cat(conv_out,dim=1)
        # [batch_size,kernel_num * len(kernel_sizes)]
        flatten = conv_out.view(conv_out.size(0),-1)
        logits = self.fc(flatten)
        return logits



