#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-11-14 10:38
# @Author  : 冯佳欣
# @File    : hparams.py
# @Desc    : 超参数的设置

import argparse

class Hprams:
    parser = argparse.ArgumentParser()

    # 每次跑模型可以调整的参数
    # 1. 调整采用的训练数据规模，词向量以及哪个模型的参数以及随机参数
    parser.add_argument('--model_name', default='FastText', type=str,
                        help='model name ,same as models/__init__.py name',choices=['FastText','TextCNN','TextGRU','GRU_Attention','RCNN'])

    parser.add_argument('--model_id', default='0', help='model id')
    parser.add_argument('--max_text_len', default=50, type=int, help='max sentence length')
    parser.add_argument('--label_size', default=19, type=int, help='class num')
    parser.add_argument('--embedding_dim', default=50, type=int, help='len of word vector')
    parser.add_argument('--text_type', default='word', help='use word2vec type ("article"/"word")',choices=['article','word'])
    parser.add_argument('--seed',default=77,type=int,help='random seed number')
    parser.add_argument('--device', default=0, type=int, help='gpu device id')
    parser.add_argument('--aug', default=False, type=bool, help='是否进行数据增强')


    # 2. 关于模型的超参数
    parser.add_argument('--lr1',default=1e-3,type=float,help='learning rate of parameters except embedding')
    parser.add_argument('--lr2',default=0,type=int,help='learning rate of embedding parameters')
    parser.add_argument('--min_lr',default=1e-5,type=float,help='当学习率低于这个值事，就退出训练')
    parser.add_argument('--lr_decay',default=0.8,type=float,help='当一个epoch的损失开始上升时，lr=lr*decay')
    parser.add_argument('--decay_every',default=10000,type=int,help='每多少个batch,查看val acc ，并修改学习率')
    parser.add_argument('--weight_decay',default=0,type=float,help='权重衰减')
    parser.add_argument('--max_epoches',default=10,type=int,help='train epoches')
    parser.add_argument('--linear_hidden_size',default=100,type=int,help='hidden size number')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size number')
    parser.add_argument('--print_every', default=2, type=int, help='print every 不能为1')

    # 3.具体的模型参数
    # TextCNN
    parser.add_argument('--kernel_num', default=2, type=int, help='TextCNN kernel channel number')
    # TextGRU
    parser.add_argument('--layer_hidden_size', default=50, type=int, help='GRU layer hidden size')
    parser.add_argument('--gru_layers', default=2, type=int, help='gru layer number')
    parser.add_argument('--gru_dropout', default=0.5, type=float, help='gru layer dropout')
    parser.add_argument('--k_max', default=2, type=int, help='k max pooling')
    # RCNN
    parser.add_argument('--rcnn_kernel_size', default=5, type=int, help='rcnn kernel window size')
    parser.add_argument('--rcnn_kernel_channels', default=50, type=int, help='rcnn kernel out channels')

    # -------------------------------------------------------------------------------------------
    # 下面的参数不需要手动调节，之后会在程序中更新
    # 1. 文件夹名称信息
    parser.add_argument('--data_dir', default='../data/mini/', help='save data top dir')

    # 2. 参数信息
    parser.add_argument('--best_score',default=0,type=float,help='model best fp1 score')
    parser.add_argument('--vocab_size', default=100, type=int, help='update by later')
    parser.add_argument('--cuda',default=False,help='是否可以使用nvida gpu训练,upadte later')