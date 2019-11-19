#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-11-14 10:17
# @Author  : 冯佳欣
# @File    : FastText.py
# @Desc    :
class FastText():
    def __init__(self):
        # 模型的默认名字
        self.model_name = str(type(self))
        
    def get_model_name(self):
        return self.model_name