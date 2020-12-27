#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2020/10/25 2:42
@File:          fpflu.py
'''

from keras import backend as K
import keras

def fpflu(x):
    return K.maximum(x, x / (1 + x * x))

keras.utils.get_custom_objects()['fpflu'] = fpflu