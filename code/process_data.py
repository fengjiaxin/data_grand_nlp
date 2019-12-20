#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-12-19 19:41
# @Author  : 冯佳欣
# @File    : process_data.py
# @Desc    : 预处理原始文件 包括三个部分

# 1. 生成 article word 的content data
# 2. 将文件切割成train 和 val
# 3. 生成article 和 word 的 train val test 文件


import pandas as pd
import numpy as np
import os


# 生成content data
def generate_content(datasets_path,content_dir):
    if not os.path.exists(content_dir):
        os.mkdir(content_dir)
    train_path = os.path.join(datasets_path,'train_set.csv')
    test_path = os.path.join(datasets_path,'test_set.csv')

    print('loading datasets......')
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    print('{} lines in train datasets'.format(len(train_data)))
    print('{} lines in test datasets'.format(len(test_data)))


    with open(os.path.join(content_dir,'raw_word.txt'),'w') as w:
        w.writelines([text + '\n' for text in train_data['word_seg']])
        w.writelines([text + '\n' for text in test_data['word_seg']])
    print('making raw word content_data file success')

    with open(os.path.join(content_dir,'raw_article.txt'),'w') as w:
        w.writelines([text + '\n' for text in train_data['article']])
        w.writelines([text + '\n' for text in test_data['article']])
    print('making raw article content_data file success')


def split_train_val(data_path,val_rate=0.1,seed=None):
    print('loading file {} success'.format(data_path))
    df = pd.read_csv(data_path)
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(m * (1.-val_rate))
    train_df = df.iloc[perm[:train_end]]
    val_df = df.iloc[perm[train_end:]]
    return train_df,val_df

def generate_article_word_train_val_test(datasets_path,article_dir,word_dir):
    train_path = os.path.join(datasets_path, 'train_set.csv')
    test_path = os.path.join(datasets_path, 'test_set.csv')
    train_df, val_df = split_train_val(train_path)
    print('split train val success')
    test_df = pd.read_csv(test_path)
    print('raw data load success')
    print('split train val success')

    if not os.path.exists(word_dir):
        os.mkdir(word_dir)

    if not os.path.exists(article_dir):
        os.mkdir(article_dir)

    # begin make word data
    train_df[['word_seg','class']].to_csv(os.path.join(word_dir,'train_set.csv'),index=False)
    val_df[['word_seg', 'class']].to_csv(os.path.join(word_dir, 'val_set.csv'), index=False)
    test_df[['id', 'word_seg']].to_csv(os.path.join(word_dir, 'test_set.csv'), index=False)
    print('make word data success')

    train_df[['article','class']].to_csv(os.path.join(article_dir,'train_set.csv'),index=False)
    val_df[['article', 'class']].to_csv(os.path.join(article_dir, 'val_set.csv'), index=False)
    test_df[['id', 'article']].to_csv(os.path.join(article_dir, 'test_set.csv'), index=False)
    print('make article data success')


if __name__ == '__main__':
    datasets_path = '../data/mini/result/'
    word_dir = '../data/mini/word/'
    article_dir = '../data/mini/article/'
    content_dir = '../data/mini/content_data/'

    generate_content(datasets_path,content_dir)
    generate_article_word_train_val_test(datasets_path,article_dir,word_dir)


