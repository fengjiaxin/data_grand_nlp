#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-11-14 08:28
# @Author  : 冯佳欣
# @File    : main.py
# @Desc    : 训练模型的文件

import torch
import time
import torch.nn.functional as F
import models
import math
import os
import pandas as pd
import json
from sklearn import metrics
import logging
logging.basicConfig(level=logging.INFO,format = '%(message)s')
from hparams import Hprams
from data_loader import get_data_loader
import numpy as np
use_gpu = torch.cuda.is_available()


def val(model,data_iter,hp):
    '''
    计算模型在验证集上的分数
    :param model: 如果cuda is True，model已经在gpu上了
    :param data_iter:
    :param hp:
    :return: f1score
    '''

    # 将模型设为验证模式
    model.eval()

    acc_n = 0
    val_n = 0

    # 建立一个空的array
    true_labels = np.zeros((0,),dtype=np.int32)
    predict = np.zeros((0,),dtype=np.int32)

    with torch.no_grad():
        for (text,label) in data_iter:
            if hp.cuda:
                text = text.cuda()
                label = label.cuda()

            outputs = model(text)
            pred = outputs.max(1)[1]
            acc_n += (pred == label).sum().item()
            val_n += label.size(0)
            if hp.cuda:
                pred = pred.cpu()
                label = label.cpu()
            predict = np.hstack((predict,pred.numpy()))
            true_labels = np.hstack((true_labels,label.numpy()))

    acc = acc_n/val_n
    f1_score = np.mean(metrics.f1_score(true_labels,predict,average=None))
    logging.info('Val Acc %.4f( %d / %d ) ,F1_score: %.4f'%(acc,acc_n,val_n,f1_score))
    return f1_score

def test(model,test_iter,hp):
    # 生成测试提交数据csv
    # 将模型设置为验证模式
    model.eval()

    result = np.zeros((0,),dtype=np.int32)
    probs_list = []
    with torch.no_grad():
        for text in test_iter:
            if hp.cuda:
                text = text.cuda()
            outputs = model(text)
            probs = F.softmax(outputs,dim=1)
            if hp.cuda:
                outputs - outputs.cpu()
                probs = probs.cpu()

            probs_list.append(probs.numpy())
            pred = outputs.max(1)[1]
            result = np.hstack((result,pred.numpy()))

    # 生成概率文件
    prob_cat = np.concatenate(probs_list,axis=0)

    test_df = pd.read_csv(hp.test_path,usecols=['id'])
    test_id = test_df['id'].values
    test_pred = pd.DataFrame({'id':test_id,'class':result})
    test_pred['class'] = (test_pred['class'] + 1).astype(int)

    return prob_cat,test_pred

def save_hparams(path,hp):
    hp_dict = vars(hp)
    with open(path,'w') as w:
        for key,value in hp_dict.items():
            w.write(str(key) + ' : ' + str(value) + '\n')


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




def train():
    logging.info('# hparams')
    hparams = Hprams()
    parser = hparams.parser
    hp = parser.parse_args()

    # 是否可以使用GPU
    if not use_gpu:
        hp.cuda = False
        torch.manual_seed(hp.seed)

    # 更新采用word2vec的相关信息
    if hp.text_type == 'article':
        setattr(hp,"column_name",'article')
    elif hp.text_type == 'word':
        setattr(hp, "column_name", 'word_seg')
    word2vec_name = hp.word2vec_type + '_' + hp.text_type + '_' + str(hp.embedding_dim) + '.txt'

    if hp.data_scale == 'mini':
        train_path = os.path.join(hp.base_dir,hp.data_scale,hp.origin_data,'mini_train_set.csv')
        val_path = os.path.join(hp.base_dir, hp.data_scale, hp.origin_data, 'mini_val_set.csv')
        test_path = os.path.join(hp.base_dir, hp.data_scale, hp.origin_data, 'mini_test_set.csv')
        setattr(hp, "train_path", train_path)
        setattr(hp, "val_path", val_path)
        setattr(hp, "test_path", test_path)
        word2vec_path = os.path.join(hp.base_dir,hp.data_scale,hp.word2vec_data,word2vec_name)
        setattr(hp, "word2vec_path", word2vec_path)
    elif hp.data_scale == 'all':
        train_path = os.path.join(hp.base_dir, hp.data_scale, hp.origin_data, 'train_set.csv')
        val_path = os.path.join(hp.base_dir, hp.data_scale, hp.origin_data, 'val_set.csv')
        test_path = os.path.join(hp.base_dir, hp.data_scale, hp.origin_data, 'test_set.csv')
        setattr(hp, "train_path", train_path)
        setattr(hp, "val_path", val_path)
        setattr(hp, "test_path", test_path)
        word2vec_path = os.path.join(hp.base_dir, hp.data_scale, hp.word2vec_data, word2vec_name)
        setattr(hp, "word2vec_path", word2vec_path)



    train_iter,val_iter,test_iter,word2index,vectors = get_data_loader(hp)
    setattr(hp,"vocab_size",len(word2index))

    global best_score
    best_score = 0.0

    # init model
    model = getattr(models,hp.model_name)(hp,vectors)
    logging.info(model)

    # 更新模型保存位置文件夹 save_dir/model_name/model_id/

    model_path_name_dir = os.path.join(hp.base_dir,hp.data_scale, hp.save_models,hp.model_name)
    if not os.path.exists(model_path_name_dir):
        os.mkdir(model_path_name_dir)
    model_path_dir = os.path.join(model_path_name_dir,hp.model_name+'_'+str(hp.model_id))
    if not os.path.exists(model_path_dir):
        os.mkdir(model_path_dir)

    save_path = os.path.join(model_path_dir,'{}_{}.pth'.format(hp.model_name,hp.model_id))

    # save path  模型之后的存储位置之后在更新
    if hp.cuda:
        torch.cuda.manual_seed(hp.seed)
        model.cuda()

    # 目标函数和优化器
    criterion = F.cross_entropy
    lr1,lr2 = hp.lr1,hp.lr2
    optimizer = model.get_optimizer(lr1,lr2,hp.weight_decay)
    n_iters = hp.max_epoches * len(train_iter)

    start = time.time()
    for epoch in range(hp.max_epoches):
        total_loss = 0.0
        correct = 0
        total = 0

        model.train()

        for idx,(text,label) in enumerate(train_iter):
            # 训练模型参数
            # 使用batchNorm 层，batch_size 不能为1
            if len(label) == 1:continue
            batch_len = len(label)

            if hp.cuda:
                text = text.cuda()
                label = label.cuda()

            optimizer.zero_grad()
            pred = model(text)
            loss = criterion(pred,label)
            loss.backward()
            optimizer.step()

            # 更新统计指标
            total_loss += loss.item()
            predicted = pred.max(1)[1]
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

            if idx % hp.print_every == hp.print_every - 1:
                logging.info("train epoch:%s batch_id:%s,time:%s (%d %d%%) Acc: %.4f (%d / %d) loss:%.4f" % \
                             (str(epoch), str(idx), timeSince(start, idx / n_iters), idx,
                              idx / n_iters * 100, 100. * correct/total,correct,total,loss.item() / batch_len))
                total_loss = 0.0

        # 计算在验证集上的分数，并相应的调整学习率
        f1_score = val(model,val_iter,hp)
        if f1_score > best_score:
            best_score = f1_score
            check_point = {
                'state_dict':model.state_dict(),
                'config':vars(hp)
            }
            torch.save(check_point,save_path)
            logging.info('Best tmp model f1score : %.4f'%best_score)

        # 如果验证集分数表现不好
        if f1_score < best_score:
            model.load_state_dict(torch.load(save_path)['state_dict'])
            lr1 *= hp.lr_decay
            lr2 = 2e-4 if lr2 == 0 else lr2 * 0.8
            optimizer = model.get_optimizer(lr1,lr2,0)

            logging.info('load previous best model : %.4f'%(best_score))
            logging.info('model lr:%.4f, emb lr:%.4f' %(lr1,lr2))

            if lr1 < hp.min_lr:
                logging.info('train over , best f1 score: %.4f'%(best_score))
                break

    # 保存最终的模型
    setattr(hp, "best_score", best_score)
    final_model = {
        'state_dict':model.state_dict(),
        'config':vars(hp)
    }
    best_score_str = "%.8f"%best_score
    best_model_path = os.path.join(model_path_dir,'{}_{}_{}.pth'.format(hp.model_name,hp.text_type,best_score_str))
    torch.save(final_model,best_model_path)
    logging.info('best final model saved in {}'.format(best_model_path))

    # 保存hparams
    hp_path = os.path.join(model_path_dir,'{}_{}_hparams.txt'.format(hp.model_name,hp.model_id))
    save_hparams(hp_path,hp)
    # 在测试集上运行模型并生成概率结果和提交结果
    result_path_dir = os.path.join(hp.base_dir,hp.data_scale, hp.result)
    if not os.path.exists(result_path_dir):
        os.mkdir(result_path_dir)


    probs,test_pred = test(model,test_iter,hp)
    result_path_name = '{}_{}_{}'.format(hp.model_name,hp.model_id,best_score_str)


    np.save(os.path.join(result_path_dir,'{}.npy'.format(result_path_name)),probs)
    logging.info('Prob result {}.npy save success!'.format(result_path_name))

    test_pred[['id','class']].to_csv(os.path.join(result_path_dir,'{}.csv'.format(result_path_name)),index=None)
    logging.info('Result result {}.csv save success!'.format(result_path_name))

    logging.info('all success')


if __name__ == '__main__':
    train()





