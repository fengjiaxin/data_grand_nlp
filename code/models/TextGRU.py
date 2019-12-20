#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-11-19 20:02
# @Author  : 冯佳欣
# @File    : TextGRU.py
# @Desc    : 采用每个单词的hidden_size作为特征，然后去k_max pooling

from .BasicModule import BasicModule
import torch
import numpy as np
from torch import nn

class TextGRU(BasicModule):
    def __init__(self,hp,vectors):
        super(TextGRU, self).__init__(hp.vocab_size,hp.embedding_dim,vectors)

        self.vocab_size = hp.vocab_size
        self.embedding_dim = hp.embedding_dim
        self.linear_hidden_size = hp.linear_hidden_size
        self.label_size = hp.label_size
        self.layer_hidden_size = hp.layer_hidden_size
        self.gru_layers = hp.gru_layers
        self.gru_dropout = hp.gru_dropout
        self.k_max = hp.k_max

        # input of shape (batch,seq_len,input_size)
        self.bigru = nn.GRU(
            input_size = self.embedding_dim,
            hidden_size=self.layer_hidden_size//2,
            num_layers=self.gru_layers,
            batch_first=True,
            dropout=self.gru_dropout,
            bidirectional=True
        )

        # 两层全连接层，中间添加批标准化层
        self.fc = nn.Sequential(
            nn.Linear(self.k_max * self.layer_hidden_size,self.linear_hidden_size),
            nn.BatchNorm1d(self.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.linear_hidden_size,self.label_size)
        )

    # 对GRU的所有hidden state vector 做kmax_pooling
    def forward(self,text):
        '''
        :param text:[batch_size,seq_len]
        :return:
        '''
        # [batch_size,seq_len,embedding_dim]
        embed = self.embedding(text)
        # output [batch_size,seq_len,layer_hidden_size]
        output,_ = self.bigru(embed)
        # [batch_size,layer_hidden_size,seq_len]
        output = output.permute(0,2,1)
        # 采用topk,期待变成[batch_size,layer_hidden_size,kmax],其中是有顺序的
        kmax_output,_ = output.topk(dim=-1,k=2)
        batch_size = output.size(0)
        # [batch_size, k * layer_hidden_size]
        flatten = kmax_output.view(batch_size,-1)
        logits = self.fc(flatten)
        return logits

