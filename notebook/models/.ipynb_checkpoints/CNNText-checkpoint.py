#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-11-14 10:18
# @Author  : 冯佳欣
# @File    : CNNText.py
# @Desc    :

class CNNText():
    def __init__(self):
        self.model_name = str(type(self))
        
    def get_model_name(self):
        return self.model_name