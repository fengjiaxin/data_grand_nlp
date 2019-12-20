#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-12-19 11:21
# @Author  : 冯佳欣
# @File    : data.py
# @Desc    : 利用torchtext 来获取数据

from torchtext import data
import pandas as pd
from torchtext.vocab import Vectors
from torch.nn import init
import random
import os
import numpy as np
import logging
# 注意本模型训练在单机单gpu上

logging.basicConfig(level=logging.INFO,format = '%(message)s')


# 定义Dataset
class GrandDataset(data.Dataset):
    def __init__(self,path,text_field,label_field,text_type='word',test=False,aug=False,**kwargs):
        fields = [('text',text_field),('label',label_field)]
        examples = []
        csv_data = pd.read_csv(path)
        logging.info('read data from {}'.format(path))

        if text_type == 'word':
            text_type = 'word_seg'

        if test:
            # 如果为测试集，则不加载label
            for text in csv_data[text_type]:
                examples.append(data.Example.fromlist([text,None],fields))

        else:
            for text,label in zip(csv_data[text_type],csv_data['class']):
                if aug:
                    # 数据增强，包括打乱顺序和随机丢弃
                    rate = random.random()
                    if rate > 0.5:
                        text = self.dropout(text)
                    else:
                        text = self.shuffle(text)
                examples.append(data.Example.fromlist([text,label - 1],fields))
        super(GrandDataset,self).__init__(examples,fields,**kwargs)

    def shuffle(self,text):
        text = np.random.permutation(text.strip().split())
        return ' '.join(text)

    def dropout(self,text,p = 0.5):
        # random delete some text
        text = text.strip().split()
        text_len = len(text)
        indexs = np.random.choice(text_len,int(text_len * p))
        for i in indexs:
            text[i] = ''
        return ' '.join(text)

def load_data(hp):
    '''
    负责数据的生成
    :param hp:
        hp.max_text_len
        hp.data_dir
        hp.text_type
        hp.embedding_dim
        hp.device
    :return:
        train_iter
        val_iter
        test_iter
        len(vocab)
        vectors
    '''

    tokenize = lambda x:x.split()
    # text 设置fix_length
    TEXT = data.Field(sequential=True,tokenize=tokenize,batch_first=True,fix_length=hp.max_text_len)
    LABEL = data.Field(sequential=False,batch_first=True,use_vocab=False)

    # load path 训练数据存储在  data_dir/text_type/下
    train_path = os.path.join(hp.data_dir,hp.text_type,'train_set.csv')
    val_path = os.path.join(hp.data_dir,hp.text_type, 'val_set.csv')
    test_path = os.path.join(hp.data_dir, hp.text_type, 'test_set.csv')

    # 数据增强
    if hp.aug:
        logging.info('make augmetiation datasets!')

    train = GrandDataset(train_path,text_field=TEXT,label_field=LABEL,text_type=hp.text_type,test=False,aug=hp.aug)
    val = GrandDataset(val_path,text_field=TEXT,label_field=LABEL,text_type=hp.text_type,test=False)
    test = GrandDataset(test_path,text_field=TEXT,label_field=None,text_type=hp.text_type,test=True)

    cache = '.vector_cache'
    if not os.path.exists(cache):
        os.mkdir(cache)

    # 词向量的位置在 data_dir/word2vec_data/ 下，名称有text_type和 embed_dim 确定
    embedding_path = os.path.join(hp.data_dir,'word2vec_data','{}_{}.txt'.format(hp.text_type,hp.embedding_dim))
    vectors = Vectors(name=embedding_path,cache=cache)
    logging.info('load word2vec vectors from {}'.format(embedding_path))
    # 没有命中的token的初始化方式
    vectors.unk_init = init.xavier_uniform_

    # 构建vocab
    logging.info('building {} vocabulary ....'.format(hp.text_type))
    TEXT.build_vocab(train,val,test,min_freq=5,vectors=vectors)

    # 构建Iterator
    # 在test_iter,val_iter ,shuffle,sort.repeat 一定要设置成False，要不然会被torchtext 搞乱样本顺序
    # 如果输入变长序列，sort_within_batch 需要设置成True

    train_iter = data.BucketIterator(dataset=train,batch_size=hp.batch_size,shuffle=True,sort_within_batch=False,\
                                     repeat=False,device=hp.device)
    val_iter = data.Iterator(dataset=val,batch_size=hp.batch_size,shuffle=False,
                             sort=False,repeat=False,device=hp.device)
    test_iter = data.Iterator(dataset=test,batch_size=hp.batch_size,shuffle=False,sort=False,
                              repeat=False,device=hp.device)

    return train_iter,val_iter,test_iter,len(TEXT.vocab),TEXT.vocab.vectors



