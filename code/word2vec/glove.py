#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-11-13 15:02
# @Author  : 冯佳欣
# @File    : glove.py
# @Desc    : glove 训练词向量的file


import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader,Dataset
import numpy as np
import scipy.sparse as sparse
from collections import Counter
import pickle
import gc
import time
import logging
import math
logging.basicConfig(level=logging.INFO)
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


# 超参数设置
# params
x_max = 100
alpha = 0.75
epoches = 5
min_count = 5
batch_size = 64
window_size = 5
vector_size = 50
learning_rate = 0.001

# get gpu
use_gpu = torch.cuda.is_available()

# 根据提供的预料库，构建此表
def build_vocab(corpus):
    '''
    build a vocabulary with word frequencies for an entire corpus
    return a dic 'w -> (i,f)'
    '''
    logging.info('building vocab from corpus')

    vocab = Counter()
    for line in corpus:
        tokens = line.split()
        vocab.update(tokens)

    logging.info('Done building vocab from corpus')

    return {word:(i,freq) for i,(word,freq) in enumerate(vocab.items())}



# 构建共现矩阵
def build_cooccur(vocab,corpus,window_size=5):
    '''
    buil a word co-occurrence list for the given corpus
    :param vocab:
    :param corpus:
    :param window_size:
    :param min_count:
    :return:
    '''
    vocab_size = len(vocab)

    # collect cooccurrences internally as a sparse matrix for passable
    # indexing speed;we will convert into a list later
    cooccurrences = sparse.lil_matrix((vocab_size,vocab_size),dtype=np.float64)

    for i,line in enumerate(corpus):
        if i % 1000 == 0:
            logging.info('building cooccurrence matrix: on line %d'%i)

        tokens = line.strip().split()
        token_ids = [vocab[word][0] for word in tokens]

        for center_i,center_id in enumerate(token_ids):
            # collect all word ids in left window of center word
            context_ids = token_ids[max(0,center_i - window_size):center_i]
            contexts_len = len(context_ids)

            for left_i,left_id in enumerate(context_ids):
                # distance from center word
                distance = contexts_len - left_i

                # Weight by inverse of distance between words
                increment = 1.0/float(distance)

                # build co-occurrence matrix symmetrically
                cooccurrences[center_id,left_id] += increment
                cooccurrences[left_id,center_id] += increment

    return cooccurrences

def timeSince(since, percent):
    '''
    :param since: 开始记录的time时刻
    :param percent: 已完成的百分比
    :return:
    '''
    now = time.time()
    pass_time = now - since
    all_time = pass_time / percent
    remain_time = all_time - pass_time
    return '%s (- %s)' % (asMinutes(pass_time), asMinutes(remain_time))

def asMinutes(s):
    '''
    将时间s转换成minute 和 second的组合
    :param s:
    :return:
    '''
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
# calculation weight
def fw(X_c_s):
    return (X_c_s/x_max) ** alpha if X_c_s < x_max else 1

class Glove(nn.Module):
    def __init__(self,vocab_size,vector_size):
        super(Glove,self).__init__()
        # center words weight and bias
        self.center_weight = nn.Embedding(vocab_size,vector_size)
        self.center_biase = nn.Embedding(vocab_size,1)

        # context words weight and bias
        self.context_weight = nn.Embedding(vocab_size,vector_size)
        self.context_biase = nn.Embedding(vocab_size,1)

    def forward(self,center_ids,context_ids):
        '''
        cal v_i^Tv_k _ b_i + b_k
        :param center_ids: [batch]
        :param context_ids: [batch]
        :return:
        '''

        center_w = self.center_weight(center_ids)
        # [batch,1]
        center_b = self.center_biase(center_ids)


        context_w = self.context_weight(center_ids)
        context_b = self.context_biase(center_ids)

        # [batch,1]
        return torch.sum(center_w.mul(context_w),1,keepdim=True) + center_b + context_b


# read data
class TrainData(Dataset):
    def __init__(self, coo_matrix, vocab, id2word, min_count=5):
        '''
        coo_matrix:sparse.lil_matrix
        vocab: word -> (index,freq)
        '''
        # 将coo_matrix的pair 以及权重存储到列表中
        # ((i,j),X_ij)
        self.coo_len = 0
        coo_matrix_list = []
        for i, (row, data) in enumerate(zip(coo_matrix.rows, coo_matrix.data)):
            # 第i个单词，需要确定它是否<= min_count
            word_i = id2word[i]
            if min_count is not None and vocab[word_i][1] < min_count:
                continue
            for (j, x) in zip(row, data):
                word_j = id2word[j]
                if min_count is not None and vocab[word_j][1] < min_count:
                    continue
                coo_matrix_list.append(((i, j), x))
        # 为了方便处理，将c,s,X_c_s,W_c_s都变成numpy矩阵
        c_list = []
        s_list = []
        X_c_s_list = []
        W_c_s_list = []
        for ((c, s), x) in coo_matrix_list:
            self.coo_len += 1
            c_list.append(c)
            s_list.append(s)
            X_c_s_list.append(x)
            W_c_s_list.append(fw(x))
        # 转换成numpy
        c_array = np.array(c_list)
        s_array = np.array(s_list)
        X_c_s_array = np.array(X_c_s_list)
        W_c_s_array = np.array(W_c_s_list)
        self.c = torch.from_numpy(c_array).long()
        self.s = torch.from_numpy(s_array).long()
        self.X_c_s = torch.from_numpy(X_c_s_array).double()
        self.W_c_s = torch.from_numpy(W_c_s_array).double()
        del c_list,s_list,X_c_s_list,W_c_s_list
        del c_array,s_array,X_c_s_array,W_c_s_array
        gc.collect()

    def __len__(self):
        return self.coo_len

    def __getitem__(self, index):
        c = self.c[index]
        s = self.s[index]
        X_c_s = self.X_c_s[index]
        W_c_s = self.W_c_s[index]
        return c, s, X_c_s, W_c_s

def loss_func(X_c_s_hat,X_c_s,W_c_s):
    '''
    计算损失函数
    :param X_c_s_hat: [batch,1]  v_i^Tv_k _ b_i + b_k
    :param X_c_s: [batch,1] X_ij 计数
    :param W_c_s: [batch,1] W_ij 权重
    :return: scalar
    '''
    # X_c_s_hat:[batch,1]
    X_c_s = X_c_s.view(-1,1)
    W_c_s = W_c_s.view(-1,1)

    # [batch,1]
    error_square = (X_c_s_hat.double() - torch.log(X_c_s)) ** 2
    loss = torch.sum(W_c_s.mul(error_square))
    return loss

def save_word_vector(file_name,id2word,glove):
    with open(file_name,'w') as w:
        if use_gpu:
            c_vector = glove.center_weight.weight.data.cpu().numpy()
            s_vector = glove.context_weight.weight.data.cpu().numpy()
            vector = c_vector + s_vector
        else:
            c_vector = glove.center_weight.weight.data.numpy()
            s_vector = glove.context_weight.weight.data.numpy()
            vector = c_vector + s_vector

        for i in range(len(vector)):
            word = id2word[i]
            word_vec = vector[i]
            word_vec = [str(s) for s in word_vec.tolist()]
            write_line = word + ' ' + ' '.join(word_vec) + '\n'
            w.write(write_line)
    logging.info('glove vector save complete')

def train_model(epoches,corpus_file_name,vector_file,coo_matrix_file):
    corpus = []
    with open(corpus_file_name, 'r') as f:
        for line in f:
            corpus.append(line.strip())

    vocab = build_vocab(corpus)
    id2word = dict((i,word) for word,(i,_) in vocab.items())
    if not os.path.exists(coo_matrix_file):
        coo_matrix = build_cooccur(vocab,corpus,window_size=window_size)
        with open(coo_matrix_file,'wb') as w:
            pickle.dump(coo_matrix,w)
        del coo_matrix
        gc.collect()
    with open(coo_matrix_file,'rb') as f:
        coo_matrix = pickle.load(f)

    vocab_size = len(vocab)
    glove = Glove(vocab_size,vector_size)
    logging.info(glove)
    if use_gpu:
        glove.cuda()
    optimizer = torch.optim.Adam(glove.parameters(),lr=learning_rate)

    train_data = TrainData(coo_matrix,vocab,id2word,min_count)
    data_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
    batch_len = len(data_loader)
    n_iters = batch_len * epoches

    start = time.time()

    for epoch in range(epoches):
        logging.info(f'current epoch is {epoch},all epoches is {epoches}')
        avg_epoch_loss = 0
        for batch_idx,(c,s,X_c_s,W_c_s) in enumerate(data_loader):
            c = torch.LongTensor(c)
            s = torch.LongTensor(s)
            X_c_s = torch.DoubleTensor(X_c_s)
            W_c_s = torch.DoubleTensor(W_c_s)

            if use_gpu:
                c = c.cuda()
                s = s.cuda()
                X_c_s = X_c_s.cuda()
                W_c_s = W_c_s.cuda()

            # [batch,1]
            W_c_s_hat = glove(c,s)
            loss = loss_func(W_c_s_hat,X_c_s,W_c_s)
            optimizer.zero_grad()
            loss.backward()
            avg_epoch_loss += loss/len(train_data)
            if batch_idx % 1000 == 1:
                logging.info("train epoch:%s batch_id:%s,time:%s (%d %d%%) %.4f" % \
                             (str(epoch), str(batch_idx), timeSince(start, batch_idx / n_iters), batch_idx,
                              batch_idx / n_iters * 100, loss.item() / batch_size))

        logging.info(f'Epoch {epoch} complete ,avg loss {avg_epoch_loss}')
    save_word_vector(vector_file,id2word,glove)

if __name__ == '__main__':
    # 生成artice 向量
    mode = 'mini'

    if mode == 'mini':
        raw_article_file = '../../data/content_data/mini_raw_article.txt'
        save_raw_article_file = '../../data/word2vec_data/mini_glove_article_' + str(vector_size) + '.txt'
        coo_matrix_article_file = '../../data/pickle_data/mini_article_coo_matrix.pkl'
        train_model(epoches,raw_article_file,save_raw_article_file,coo_matrix_article_file)
        logging.info('generate mini article vector success')

        # 生成word 向量
        raw_word_file = '../../data/content_data/mini_raw_word.txt'
        save_raw_word_file = '../../data/word2vec_data/mini_glove_word_' + str(vector_size) + '.txt'
        coo_matrix_word_file = '../../data/pickle_data/mini_word_coo_matrix.pkl'
        train_model(epoches,raw_word_file,save_raw_word_file,coo_matrix_word_file)
        logging.info('generate mini word vector success')
    else:

        raw_article_file = '../../data/content_data/mini_raw_article.txt'
        save_raw_article_file = '../../data/word2vec_data/glove_article_' + str(vector_size) + '.txt'
        coo_matrix_article_file = '../../data/pickle_data/article_coo_matrix.pkl'
        train_model(epoches, raw_article_file, save_raw_article_file, coo_matrix_article_file)
        logging.info('generate article vector success')

        # 生成word 向量
        raw_word_file = '../../data/content_data/mini_raw_word.txt'
        save_raw_word_file = '../../data/word2vec_data/glove_word_' + str(vector_size) + '.txt'
        coo_matrix_word_file = '../../data/pickle_data/word_coo_matrix.pkl'
        train_model(epoches, raw_word_file, save_raw_word_file, coo_matrix_word_file)
        logging.info('generate word vector success')



