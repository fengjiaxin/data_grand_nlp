#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-11-14 19:00
# @Author  : 冯佳欣
# @File    : test_argparse.py
# @Desc    :

import argparse

class Hprams:
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed',default=77,type=int,help='random seed number')


hparams = Hprams()
parser = hparams.parser
hp = parser.parse_args()
#hp_dict = dict(vars(hp))
print(hp)
#hp_dict['seed'] = 78
setattr(hp,"seed",78)
print(hp.seed)
print(hp)