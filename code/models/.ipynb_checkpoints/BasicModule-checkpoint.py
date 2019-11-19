#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-11-13 22:03
# @Author  : 冯佳欣
# @File    : BasicModule.py
# @Desc    : 所有模型的基本文件

import torch

class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name = str(type(self))


    def get_optimizer(self,lr1,lr2=0,weight_decay=0):
        '''
        训练的时候参数分为两部分
            1.embedding参数
            2.除embedding参数的其他部分
        :param lr1: 训练除embdding的其他参数
        :param lr2: 训练embedding
        :param weight_decay:
        :return:
        '''

        # id(object):获取对象的内存地址
        # map(function,iter) 返回迭代器
        # 获取embedding参数在内存中的地址的列表
        embed_params_adress = list(map(id,self.embedding.parameters()))

        # filter(function,iterable) 最后将返回True的元素放到新的迭代器中
        base_params = filter(lambda p:id(p) not in embed_params_adress,self.parameters())
        optimizer = torch.optim.Adam([
            {'params':self.embedding.parameters(),'lr':lr2},
            {'params':base_params,'lr':lr1,'wight_decay':weight_decay}
        ])
        return optimizer