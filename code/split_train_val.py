#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-11-19 18:38
# @Author  : 冯佳欣
# @File    : split_train_val.py
# @Desc    : 切割文件

import pandas as pd
from sklearn.model_selection import train_test_split

def split_train_val(data_path,train_path,val_path):
    df = pd.read_csv(data_path)
    print('loading data success')
    X = df.drop(columns=['class'])
    y = df[['class']]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y)
    X_train['class'] = y_train
    X_val['class'] = y_val
    X_train.to_csv(train_path,index=None)
    X_val.to_csv(val_path,index=None)
    print('split train val success')


if __name__ == '__main__':
    data_path = '../data/mini/origin_data/mini_train_set_origin.csv'
    train_path = '../data/mini/origin_data/mini_train_set.csv'
    val_path = '../data/mini/origin_data/mini_val_set.csv'
    split_train_val(data_path,train_path,val_path)