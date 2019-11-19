#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-11-13 14:23
# @Author  : 冯佳欣
# @File    : generate_raw_words.py
# @Desc    : 根据原始文件生成content 文件

import pandas as pd
import os

def main(train_path,test_path,raw_word_file,raw_article_file):
    print('loading datasets......')
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    print('{} lines in train datasets'.format(len(train_data)))
    print('{} lines in test datasets'.format(len(test_data)))

    print('making mini_raw_word.txt.......')
    with open(raw_word_file,'w') as w:
        w.writelines([text + '\n' for text in train_data['word_seg']])
        w.writelines([text + '\n' for text in test_data['word_seg']])

    print('making mini_raw_article.txt.....')
    with open(raw_article_file,'w') as w:
        w.writelines([text + '\n' for text in train_data['article']])
        w.writelines([text + '\n' for text in test_data['article']])

    print('handle all success')

if __name__ == '__main__':
    basic_dir = '../../data/'
    origin_dir = 'origin_data/'
    content_dir = 'content_data'
    train_fp = 'train_set.csv'
    test_fp = 'test_set.csv'
    mini_train_fp = 'mini_train_set.csv'
    mini_test_fp = 'mini_test_set.csv'
    mini_raw_word_fp = 'mini_raw_word.txt'
    mini_raw_article_fp = 'mini_raw_article.txt'
    raw_word_fp = 'mini_raw_word.txt'
    raw_article_fp = 'mini_raw_article.txt'


    print('begin generate mini data')
    mini_train_path = os.path.join(basic_dir,origin_dir,mini_train_fp)
    mini_test_path = os.path.join(basic_dir,origin_dir,mini_test_fp)

    mini_raw_word_path = os.path.join(basic_dir,content_dir,mini_raw_word_fp)
    mini_raw_article_path = os.path.join(basic_dir,content_dir,mini_raw_article_fp)
    main(mini_train_path,mini_test_path,mini_raw_word_path,mini_raw_article_path)

    print('begin generate all data')
    train_path = os.path.join(basic_dir,origin_dir,train_fp)
    test_path = os.path.join(basic_dir,origin_dir,test_fp)

    raw_word_path = os.path.join(basic_dir,content_dir,raw_word_fp)
    raw_article_path = os.path.join(basic_dir,content_dir,raw_article_fp)
    main(train_path,test_path,raw_word_path,raw_article_path)






