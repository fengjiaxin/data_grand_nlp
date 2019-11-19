#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-11-13 16:01
# @Author  : 冯佳欣
# @File    : skip_gram.py
# @Desc    : skip_gram 模型训练词向量

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import numpy as np
import torch
import math
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import pickle
import gc
import logging
logging.basicConfig(level=logging.INFO)

use_gpu = torch.cuda.is_available()


# 统计中间文件的词频，并构建词表等信息
def construct_vocab(corpus):
    '''
       :param corpus: sentence list
       :return:
           idx_to_word:id和word的对应关系
           word_to_ix:word和ix的对应关系
           word_freqs:array[max_vocab_size] 每个单词的采样权重
       '''
    word_counts = Counter()
    # 文中单词出现的所有次数
    all_word_num = 0
    for line in corpus:
        vec = line.strip().split()
        all_word_num += len(vec)
        word_counts.update(vec)

    # 构建词表，选取出现频率最大的max_vocab_size,-1是为<unk>留位置
    vocab = dict(word_counts)

    idx_to_word = {idx:word for idx,word in enumerate(vocab.keys())}
    word_to_ix = {word:idx for idx,word in enumerate(vocab.keys())}

    # 统计每个单词的词频
    word_counts = np.array([count for count in vocab.values()],dtype=np.float32)
    # 每个单词的频率
    word_freqs = word_counts/np.sum(word_counts)

    # 论文中提到将频率 3/4幂运算后，然后归一化，对negitave sample 有提升
    word_freqs = word_freqs ** (3./4.)
    word_freqs = word_freqs/np.sum(word_freqs)

    return idx_to_word,word_to_ix,word_freqs

def get_context_target_data(corpus,word_to_ix,window_size,target_ids_file,context_ids_file):
    '''
    将sentgence 按照word_to_ix 和window_size，获取每个中心单词的id 以及中心单词周围的context id
    注意：对于一句话 w1 w2 w3 w4 w5
    对于w1来说，没有上文单词，为了方便处理，这里的操作是将w4,w5 当作w1 的上文
    同理，对于w5来说，w1,w2 当作w5的 下文
    :param sentence_file:
    :param word_to_ix:
    :param window_size:
    :return:
        target_id_list:存储中心的id的列表 [w1,w2]
        context_ids_list: 列表里面存储的是target上下文单词id的列表 [[w2,w3,w4,w5],[w1,w5,w3,w4]]
    '''
    if not os.path.exists(target_ids_file) and not os.path.exists(context_ids_file):
        target_ids_list = []
        context_ids_list = []
        for line in corpus:
            sent_vec = line.strip().split()
            # 句子的长度
            sent_len = len(sent_vec)
            for target_word_pos,target_word in enumerate(sent_vec):
                target_word_id = word_to_ix[target_word]
                # context 单词针对 target 位置的偏移
                # 获取目标单词的context_id_list
                temp_context_ids = []
                for context_pos in range(-window_size,window_size + 1):
                    if context_pos == 0:continue
                    context_word = sent_vec[(target_word_pos + context_pos) % sent_len]
                    context_word_id = word_to_ix[context_word]
                    temp_context_ids.append(context_word_id)
                target_ids_list.append(target_word_id)
                context_ids_list.append(temp_context_ids)
        with open(target_ids_file,'wb') as t,open(context_ids_file,'wb') as c:
            pickle.dump(target_ids_list,t)
            pickle.dump(context_ids_list,c)
        return target_ids_list,context_ids_list
    else:
        with open(target_ids_file,'rb') as t,open(context_ids_file,'rb') as c:
            target_ids_list = pickle.load(t)
            context_ids_list = pickle.load(c)

        return target_ids_list,context_ids_list

# 显示时间
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

class MyDataset(Dataset):
    def __init__(self,corpus,word_to_ix,word_freqs,window_size,k,target_ids_file,context_ids_file):
        super(MyDataset,self).__init__()
        self.word_freqs = torch.Tensor(word_freqs)

        target_ids,context_ids_list = get_context_target_data(corpus,word_to_ix,window_size,target_ids_file,context_ids_file)
        # 将列表转换成tensor
        self.target_ids = torch.Tensor(target_ids)
        self.context_ids_list = torch.Tensor(context_ids_list)
        self.window_size = window_size
        self.k = k

    def __len__(self):
        return self.target_ids.size(0)

    def __getitem__(self, index):
        '''
        这个function 返回以下数据进行训练
            - 中心词
            - 这个单词附近的positive 单词
            - 随机采样的K个单词作为negative sample
        '''
        center_word_id = self.target_ids[index]
        pos_indices = self.context_ids_list[index]

        # negative sample
        neg_indices = torch.multinomial(self.word_freqs,self.k * self.window_size * 2,True)

        return center_word_id,pos_indices,neg_indices

def get_data_loader(corpus,word_to_ix,word_freqs,window_size,k,batch_size,target_ids_file,context_ids_file):
    dataset = MyDataset(corpus,word_to_ix,word_freqs,window_size,k,target_ids_file,context_ids_file)
    data_loader = data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
    return data_loader


class Skip_Gram(nn.Module):
    def __init__(self,vocab_size,embed_size):
        '''
        :param vocab_size: 此表数量
        :param embed_size: 词向量维度
        '''
        super(Skip_Gram,self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        init_range = 0.5 / self.embed_size
        # target词嵌入矩阵
        self.target_embed = nn.Embedding(self.vocab_size,self.embed_size)
        self.target_embed.weight.data.uniform_(-init_range,init_range)

        # context词嵌入矩阵
        self.context_embed = nn.Embedding(self.vocab_size,self.embed_size)
        self.context_embed.weight.data.uniform_(-init_range,init_range)

    def forward(self,input_labels,pos_labels,neg_labels):
        '''
        :param input_labels: 中心词 [batch_size]
        :param pos_labels: 中心词周围 context window 出现过的单词 [batch_size,window_size * 2]
        :param neg_labels: 中心词周围没有出现过的单词，从negative sampling [batch_size,window_size * 2 * K]
        :return:
        '''

        # [batch_size,embed_size]
        input_embedding = self.target_embed(input_labels)
        # [batch_size,window_size * 2,embed_size]
        pos_embedding = self.context_embed(pos_labels)
        # [batch_size,window_size * 2 * k,embed_size]
        neg_embedding = self.context_embed(neg_labels)

        log_pos = torch.bmm(pos_embedding,input_embedding.unsqueeze(2)).squeeze()
        log_neg = torch.bmm(neg_embedding,-input_embedding.unsqueeze(2)).squeeze()

        log_pos = F.logsigmoid(log_pos).sum(1)
        log_neg = F.logsigmoid(log_neg).sum(1)

        loss = log_pos + log_neg
        return -loss.mean()

    def input_embeddings(self):
        return self.target_embed.weight.data

# 显示时间
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

# 训练模型
def train_model(model,train_iter,epoches,lr):
    start = time.time()
    logging.info('## 1.begin training  ')
    # 将模式设置为训练模式
    model.train()
    optimizer = optim.Adam(model.parameters(),lr=lr)
    batch_len = len(train_iter)
    logging.info('batch_len:%d'%(batch_len))
    n_iters = batch_len * epoches

    for epoch in range(epoches):
        for batch_idx,(batch_target,batch_pos,batch_neg) in enumerate(train_iter):
            batch_size = batch_target.size()[0]
            # 清除所有优化的梯度
            optimizer.zero_grad()
            # 传入数据并向前传播获取输出
            batch_target = batch_target.long()
            batch_pos = batch_pos.long()
            batch_neg = batch_neg.long()
            if use_gpu:
                batch_target = batch_target.cuda()
                batch_pos = batch_pos.cuda()
                batch_neg = batch_neg.cuda()

            loss = model(batch_target,batch_pos,batch_neg)
            loss.backward()
            optimizer.step()
            if batch_idx % 1000 == 1:
                logging.info("train epoch:%s batch_id:%s,time:%s (%d %d%%) %.4f" %\
                             (str(epoch),str(batch_idx),timeSince(start,batch_idx/n_iters),batch_idx,batch_idx/n_iters * 100,loss.item()/batch_size))

# 根据read_file 生成词向量文件
def main(read_file,write_file,target_ids_file,context_ids_file):
    with open(read_file, 'r') as f:
        corpus = [x.strip() for x in f.readlines()]
    idx_to_word, word_to_ix, word_freqs = construct_vocab(corpus)
    # 获取词向量信息
    data_iter = get_data_loader(corpus, word_to_ix, word_freqs, window_size, k, batch_size)
    vocab_size = len(word_to_ix)

    # 定义模型
    model = Skip_Gram(vocab_size, vector_size)
    if use_gpu:
        model = model.cuda()

    # 训练
    logging.info('开始训练模型')
    train_model(model, data_iter, epoches, lr)

    logging.info('保存word_embedding')
    if use_gpu:
        word_embedding_array = model.input_embeddings().cpu().numpy()
    else:
        word_embedding_array = model.input_embeddings().numpy()
    with open(write_file, 'w') as w:
        for i, word_vec in enumerate(word_embedding_array):
            word = idx_to_word[i]
            vec_str = ' '.join(['%.8f' % x for x in word_vec])
            w.write(word + ' ' + vec_str + '\n')




if __name__ == '__main__':

    # 超参数
    k = 50  # number of negative samples
    window_size = 5
    batch_size = 2048
    epoches = 5
    vector_size = 100 # 词向量维度
    lr = 0.001
    mode = 'all'

    if mode == 'mini':
        # 训练atricle 向量
        raw_article = '../../data/content_data/mini_raw_article.txt'
        skip_article_file = '../../data/word2vec_data/mini_skip2gram_article_' + str(vector_size) + '.txt'
        logging.info('### !!begin generate mini article vector')
        main(raw_article,skip_article_file)
        logging.info('### generate mini article vector success !!!!')

        # 训练 word 向量
        raw_word = '../../data/content_data/mini_raw_word.txt'
        skip_word_file = '../../data/word2vec_data/mini_skip2gram_word_' + str(vector_size) + '.txt'
        logging.info('### !!begin generate mini word vector')
        main(raw_word, skip_word_file)
        logging.info('### generate mini word vector success !!!!')

    else:
        # 训练atricle 向量
        raw_article = '../../data/content_data/mini_raw_article.txt'
        skip_article_file = '../../data/word2vec_data/skip2gram_article_' + str(vector_size) + '.txt'
        target_ids_file = '../../data/pickle_data/article_target_ids_list.pkl'
        context_ids_file = '../../data/pickle_data/article_context_ids_list.pkl'
        logging.info('### !!begin generate mini article vector')
        main(raw_article, skip_article_file,target_ids_file,context_ids_file)
        logging.info('### generate article vector success !!!!')

        # 训练 word 向量
        raw_word = '../../data/content_data/mini_raw_word.txt'
        skip_word_file = '../../data/word2vec_data/skip2gram_word_' + str(vector_size) + '.txt'
        target_ids_file = '../../data/pickle_data/word_target_ids_list.pkl'
        context_ids_file = '../../data/pickle_data/word_context_ids_list.pkl'
        logging.info('### !!begin generate word vector')
        main(raw_word, skip_word_file,target_ids_file,context_ids_file)
        logging.info('### generate word vector success !!!!')