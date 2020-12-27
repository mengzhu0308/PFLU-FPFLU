#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2020/10/25 2:42
@File:          pflu.py
'''

from keras import backend as K
import keras

def pflu(x):
    return x * (1 + x / K.sqrt(1 + x * x)) / 2

keras.utils.get_custom_objects()['pflu'] = pflu