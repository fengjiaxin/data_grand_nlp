#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-11-13 23:06
# @Author  : 冯佳欣
# @File    : data_loader.py
# @Desc    : 载入输入的数据文件

import torch
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np

class MyDataset(Dataset):
    def __init__(self,corpus,labels=None,word2index=None,max_len=50):
        '''
        :param corpus: array of sentence
        :param word2index: word2index_dic,其中<UNK> -> 0
        :param max_len: 截取的长度
        :param labels: array label是从1开始,如果是test data:labels=None
        '''
        self.corpus = corpus
        self.word2index = word2index
        self.max_len = max_len
        if labels is None:
            self.labels = None
        else:
            self.labels = labels - 1

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        if self.labels is not None:
            label = self.labels[index]
            content = self.corpus[index]
            document_encode = [self.word2index.get(word,0) for word in content.strip().split()]
            if len(document_encode) < self.max_len:
                extended_sentences = [0] * (self.max_len - len(document_encode))
                document_encode.extend(extended_sentences)

            document_encode = torch.Tensor(document_encode[:self.max_len])
            return document_encode.long(),label
        else:
            content = self.corpus[index]
            document_encode = [self.word2index.get(word, 0) for word in content.strip().split()]
            if len(document_encode) < self.max_len:
                extended_sentences = [0] * (self.max_len - len(document_encode))
                document_encode.extend(extended_sentences)

            document_encode = torch.Tensor(document_encode[:self.max_len])
            return document_encode.long()

def data_loader(data_file,batch_size,word2index,max_seq_len,column_name,label,mode='train'):
    if mode == 'train':
        train_df = pd.read_csv(data_file,usecols=[column_name,label])
        corpus = train_df[column_name]
        labels = train_df[label]
        train_dataset = MyDataset(corpus,labels,word2index,max_seq_len)
        #train_loader = data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=1)
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        return train_loader
    if mode == 'val':
        val_df = pd.read_csv(data_file, usecols=[column_name, label])
        corpus = val_df[column_name]
        labels = val_df[label]
        val_dataset = MyDataset(corpus, labels, word2index, max_seq_len)
        #val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=1)
        val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return val_loader
    if mode == 'test':
        test_df = pd.read_csv(data_file, usecols=[column_name])
        corpus = test_df[column_name]
        labels = None
        test_dataset = MyDataset(corpus, labels, word2index, max_seq_len)
        #test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return test_loader


def get_data_loader(hp):
    max_len = hp.max_seq_len
    train_path = hp.train_path
    val_path = hp.val_path
    test_path = hp.test_path
    word_vec_path = hp.word2vec_path
    vector_size = hp.embedding_dim
    column_name = hp.column_name
    label = hp.label
    batch_size = hp.batch_size
    #
    word2index= dict()
    word2index['<UNK>'] = 0
    # list 嵌套 list
    word2vector = [[0.0 for i in range(vector_size)] ]

    with open(word_vec_path,'r') as f:
        for line in f:
            vec = line.strip().split()
            word = vec[0]
            word_vec = vec[1:]
            assert len(word_vec) == vector_size
            word2index[word] = len(word2index)
            word_vec_array = [float(x) for x in word_vec]
            word2vector.append(word_vec_array)

    train_iter = data_loader(train_path,batch_size,word2index,max_len,column_name,label,mode='train')
    val_iter = data_loader(val_path, batch_size, word2index, max_len, column_name, label, mode='val')
    test_iter = data_loader(test_path, batch_size, word2index, max_len, column_name, label, mode='test')

    return train_iter,val_iter,test_iter,word2index,np.array(word2vector,dtype=np.float64)



