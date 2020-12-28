#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2020/10/25 2:42
@File:          pflu.py
'''

from keras import backend as K
import keras

'''
https://doi.org/10.1016/j.neucom.2020.11.068
PFLU and FPFLU: Two novel non-monotonic activation functions in convolutional neural networks
'''
def pflu(x):
    return x * (1 + x / K.sqrt(1 + x * x)) / 2

keras.utils.get_custom_objects()['pflu'] = pflu
