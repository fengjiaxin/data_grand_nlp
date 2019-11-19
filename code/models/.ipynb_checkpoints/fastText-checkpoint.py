#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-11-13 22:01
# @Author  : 冯佳欣
# @File    : FastText.py
# @Desc    : fastText 模型文件

from .BasicModule import BasicModule
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class FastText(BasicModule):
    def __init__(self,hp_dict,vectors=None):
        '''
        :param hp_dict:
        :param vectors: 默认是numpy array
        '''
        super(FastText,self).__init__()
        self.vocab_size = hp_dict['vocab_size']
        self.embedding_dim = hp_dict['embedding_dim']
        self.hidden_size = hp_dict['hidden_size']
        self.label_size = hp_dict['label_size']

        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        if vectors is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(vectors))

        self.pre = nn.Sequential(
            nn.Linear(self.embedding_dim,self.embedding_dim * 2),
            nn.BatchNorm1d(self.embedding_dim * 2),
            # 修改输入数据
            nn.ReLU(True)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.embedding_dim * 2,self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size,self.label_size)

        )

    def forward(self,x):
        '''
        前向传播
        :param x:[batch_size,seq_len] 其中的值都是word_index
        :return: [batch_size,label_size]
        '''

        # [batch_size,seq_len,emb_dim]
        embed = self.embedding(x)
        embed_size = embed.size()
        # 在送入到pre 之前，需要将embed reshape [batch_size * seq_len ,emb_dim]
        embed = embed.contiguous().view(-1,self.embedding_dim)
        # [batch_size,emb_dim * 2]
        out = self.pre(embed)
        # reshape [batch_size,seq_len,emb_dim * 2]
        out = out.view(embed_size[0],embed_size[1],-1)

        # [batch_size,seq_len,2 * emb_dim]
        print('out size:' + str(out.size()))
        # [batch_size,2 * emb_dim]
        mean_pre_embed = torch.mean(out,dim=1)
        print('mean_pre_embed size:' + str(mean_pre_embed.size()))
        logit = self.fc(mean_pre_embed)
        print(logit.size())
        return logit

