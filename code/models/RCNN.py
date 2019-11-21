#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-11-21 15:50
# @Author  : 冯佳欣
# @File    : RCNN.py
# @Desc    : RNN 和 CNN 模型

import torch
import torch.nn as nn
from .BasicModule import BasicModule
import torch.nn.functional as F

def kmax_pooling(x,k):
    '''
    沿着seq_len的维度 取 topk
    :param x: [batch_size,hidden_size,seq_len]
    :param k:kmax
    :return: [batch_size,hidden_size,k]
    '''
    kmax_output,_ = x.topk(dim=-1,k=k)
    return kmax_output

class RCNN(BasicModule):
    def __init__(self,hp,vectors):
        super(RCNN,self).__init__()
        self.k_max = hp.k_max

        self.embedding = nn.Embedding(hp.vocab_size,hp.embedding_dim)
        if vectors is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(vectors))

        self.bigru = nn.GRU(
            input_size = hp.embedding_dim,
            hidden_size = hp.layer_hidden_size//2,
            num_layers = hp.gru_layers,
            batch_first=True,
            dropout=hp.gru_dropout,
            bidirectional=True
        )

        self.conv = nn.Conv1d(
            in_channels=hp.embedding_dim + hp.layer_hidden_size,
            out_channels = hp.rcnn_kernel_channels,
            kernel_size = hp.rcnn_kernel_size
        )

        self.fc = nn.Sequential(
            nn.Linear(hp.k_max * hp.rcnn_kernel_channels,hp.linear_hidden_size),
            nn.BatchNorm1d(hp.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hp.linear_hidden_size,hp.label_size)
        )

    def forward(self,text):
        embeds = self.embedding(text) # [batch_size,seq_len,embedding_dim]
        gru_outputs,_ = self.bigru(embeds) # [batch_size,seq_len,layer_hidden_size]
        # 每个词的表示为词向量加上上下文[context_l,w,context_r]
        word_representation = torch.cat((embeds,gru_outputs),dim=-1) # [batch_size,seq_len,layer_hidden_size + embedding_size]

        conv_out = self.conv(word_representation.permute(0,2,1)) # [batch_size,rcnn_kernel_channels,seq_len - rcnn_kernel_size + 1]

        kmax_output = kmax_pooling(conv_out,self.k_max) # [batch_size,rcnn_kernel_channels,k_max]

        flatten = kmax_output.view(kmax_output.size(0),-1) # [batch_size,k_max*(rcnn_kernel_channels)]
        logits = self.fc(flatten)
        return logits



