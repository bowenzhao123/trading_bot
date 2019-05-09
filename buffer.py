# -*- coding: utf-8 -*-
# @Time : 2019/4/25 10:38 PM
# @Author : ZHAOBOWEN467@pingan.com.cn
# @File : buffer.py
# @Software: PyCharm
# @desc:

from collections import deque,namedtuple
from actor import Actor
from critic import Critic
import random

class ReplayBuffer(object):
    """ Provide the experience buffer for off-line training
    Attr:
        buffer: buffer used to store experiences

    """
    def __init__(self,buff_size):
        self._buffer = deque(maxlen=buff_size)
    @property
    def size(self):
        return len(self._buffer)

    def put(self,elem):
        """ append new experience, if exceed the maximum size, then delete the oldest one

        :param elem:
        :return:
        """
        if isinstance(elem,list):
            if len(elem) > 1:
                elem = list(map(tuple,elem))
                self._buffer.extend(elem)
            else:
                self._buffer.extend(tuple(elem))
        elif isinstance(elem,tuple):
            if len(elem) > 1:
                self._buffer.extend(elem)
            else:
                self._buffer.append(elem)
        else:
            raise TypeError('input element should be list or tuple')

    def enque(self,elem):
        self._buffer.append(elem)

    def print_buffer(self):
        print(self._buffer)

    def batch(self,batch_size):
        """  sample a mini-batch from buffer

        :param batch_size:
        :return:
        """
        return random.sample(self._buffer,batch_size)