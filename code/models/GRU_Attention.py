#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-11-21 13:51
# @Author  : 冯佳欣
# @File    : GRU_Attention.py
# @Desc    : 双向GRU Attention机制

import torch
import torch.nn as nn
from .BasicModule import BasicModule
import torch.nn.functional as F

class GRU_Attention(BasicModule):
    def __init__(self,hp,vectors):
        super(GRU_Attention,self).__init__()
        self.layer_hidden_dim = hp.layer_hidden_size
        self.gru_layers = hp.gru_layers
        self.vocab_size = hp.vocab_size
        self.embedding_dim = hp.embedding_dim
        self.label_size = hp.label_size

        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        if vectors is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(vectors))

        self.bigru = nn.GRU(
            hp.embedding_dim,
            self.layer_hidden_dim//2,
            num_layers=self.gru_layers,
            batch_first=True,
            bidirectional=True)

        # 计算u的得分向量 linear
        self.u_linear = nn.Linear(self.layer_hidden_dim,self.layer_hidden_dim)

        # 将u得分映射到一维向量 linear
        self.e_linear = nn.Linear(self.layer_hidden_dim,1)

        # 最后的全连接层
        self.fc = nn.Linear(self.layer_hidden_dim,self.label_size)

    def forward(self,text):
        '''
        注意，需要注意mask 和 padding
        :param text: [batch_size,seq_len]
        :return:
        '''

        attend_mask = text.clone()
        attend_mask[attend_mask!=0]=1
        attend_mask[attend_mask==0]=0
        # 经过上述操作，[batch_size,seq_len],其中是pad的单词元素为0，不是padding的单词元素为1

        softmax_mask = text.clone()
        softmax_mask[softmax_mask!=0] = 0
        softmax_mask[softmax_mask==0] = -9e8 # make the softmax very small




        embeds = self.embedding(text) # [batch_size,seq_len,embedding_dim]
        gru_out,_ = self.bigru(embeds) # [batch_size,seq_len,layer_hidden_size]

        u = torch.tanh(self.u_linear(gru_out)) # [batch_size,seq_len,layer_hidden_size]
        a = self.e_linear(u).squeeze() # [batch_size,seq_len]
        a = a + softmax_mask # [batch_size,seq_len],如果是pad的话，a的值非常小
        att_score = F.softmax(a,dim=1).unsqueeze(1) # [batch_size,seq_len,1]

        # 将embeds中pad的隐藏向量设置为0
        embeds = embeds * attend_mask.unsqueeze(1) # [batch_size,seq_len,layer_hidden_dim]
        score_embeds = embeds * att_score # [batch_size,seq_len,layer_hidden_dim]
        context_vec = torch.sum(score_embeds,dim=1) # [batch_size,layer_hidden_dim]

        y = self.fc(context_vec)

        return y


