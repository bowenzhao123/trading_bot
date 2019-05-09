# -*- coding: utf-8 -*-
# @Time : 2019/4/27 11:39 PM
# @Author : ZHAOBOWEN467@pingan.com.cn
# @File : test.py
# @Software: PyCharm
# @desc:

import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('only_return.csv')

print(df.head())

#plt.plot(df['return_Close'])