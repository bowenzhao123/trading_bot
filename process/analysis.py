# -*- coding: utf-8 -*-
# @Time : 2019/4/23 7:22 PM
# @Author : ZHAOBOWEN467@pingan.com.cn
# @File : analysis.py
# @Software: PyCharm
# @desc:

import tensorflow as tf

#with tf.Session() as sess:
  #  x = tf.placeholder(shape=(),dtype=tf.float32)
   # y = tf.nn.sigmoid(x)
   # res = sess.run(y,{x:1.0})
  #  print(res)

from env.StockTradingEnv import StockTradingEnv
import pandas as pd
import numpy as np
df = pd.read_csv('data/only_return.csv')
# df['return_rate'] = np.random.rand(len(df))
df = df.iloc[:, 1:]
print(df.head())
env = StockTradingEnv(df,10)
state = env.reset()
done = False


while not done:
    action = np.random.rand(2)
    next_state, reward, done, _ = env.step(action)
    # print('reward: {}; netWorth: {}; balance: {}; sell: {}; buy: {}'.format(reward,env.netWorth,env.balance,env.sell_amount, env.buy_amount))
    state = next_state
