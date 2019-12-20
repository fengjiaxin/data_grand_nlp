#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-11-22 11:43
# @Author  : 冯佳欣
# @File    : train_word2vec.py
# @Desc    : 利用gensim.word2vec 训练词向量

import os
from gensim.models import word2vec

# 向量维度信息
embedding_dim = 50

def train(input_path,output_path):
    print('start handle sentences')
    sentences = word2vec.LineSentence(input_path)
    print('begin train model')
    model = word2vec.Word2Vec(sentences,hs=1,size=embedding_dim)
    print('begin save word vector')
    with open(output_path,'w') as w:
        for word in model.wv.index2word:
            vector = model.wv[word]
            w.write(str(word) + ' ' + ' '.join(map(str,vector)) + '\n')
    print('save success')

if __name__ == '__main__':
    content_dir = '../data/mini/content_data/'

    raw_article = os.path.join(content_dir,'raw_article.txt')
    raw_word = os.path.join(content_dir,'raw_word.txt')

    word2vec_dir = '../data/mini/word2vec_data/'
    if not os.path.exists(word2vec_dir):
        os.mkdir(word2vec_dir)

    article_embedding_path = os.path.join(word2vec_dir,'article_{}.txt'.format(embedding_dim))
    word_embeddding_path = os.path.join(word2vec_dir,'word_{}.txt'.format(embedding_dim))

    print('begin train article word2vec from file {}'.format(raw_article))
    train(raw_article,article_embedding_path)
    print('begin train word word2vec from file {}'.format(raw_word))
    train(raw_word,word_embeddding_path)