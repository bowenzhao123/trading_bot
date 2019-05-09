# -*- coding: utf-8 -*-
# @Time : 2019/4/23 7:23 PM
# @Author : ZHAOBOWEN467@pingan.com.cn
# @File : main.py
# @Software: PyCharm
# @desc:

import tensorflow as tf
from buffer import ReplayBuffer
from actor import Actor
from critic import Critic
import numpy as np
from config import Config
import argparse
from env.StockTradingEnv import StockTradingEnv
import pandas as pd
import keras
config = Config()
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def ddpg(sess,env,epochs,batch_size,buff_size,init_size,tau,state_size,
         time_step,unit_size,lstm_units,unit_size1,lstm_units1,discount):
    """ Implementation of deep determininistic Policy Gradient

    :param sess:  tf.Session()
    :param env:  environment
    :param epochs:  episode number
    :param batch_size:  mini-batch size
    :param buff_size: maximum buffer size
    :param init_size: initial size of buffer
    :param tau: parameter of soft update
    :param state_size: last dimension of input state
    :param time_step: time lag
    :param unit_size: actor network dense layer unit
    :param lstm_units:  actor network lstm layer unit
    :param unit_size1: critic network dense layer unit
    :param lstm_units1: critic network lstm layer unit
    :param discount:
    :return: average reward for each episode
    """
    actor = Actor(sess,tau,state_size,time_step,unit_size,lstm_units)  # init actor object, including oneline and target network
    critic = Critic(sess,tau,state_size,time_step,unit_size1,lstm_units1) # init critic object, including oneline and target network
    buff = ReplayBuffer(buff_size)
    steps = 0
    reward_ls = []  # store average reward for each episode
    for i in range(epochs):
        state = env.reset()
        done_flag = False
        rews = 0
        while not done_flag:
            # use target actor network to select action
            action = actor.model.predict([np.array([state[0]]),np.array([state[1]])])[0]
            print(action)
            noise = np.random.uniform(-.05,.05,2)
            action = np.array([(action[0]+noise[0]),(action[1]+noise[1])])  # add noise

            next_state,reward,done_flag,_ = env.step(action)
            rews += reward

            buff.enque((state,action,reward,next_state,done_flag))
            if buff.size > init_size:
                print('action: {};   reward: {}'.format(action, reward))
                mini_batch = buff.batch(batch_size)  # get mini batch to update network
                states = [x[0] for x in mini_batch]
                input_states = [x[0] for x in states]
                value_states = [x[1] for x in states]
                actions = [x[1] for x in mini_batch]
                rewards = [x[2] for x in mini_batch]
                next_states = [x[3] for x in mini_batch]
                input_next_states = np.array([x[0] for x in next_states])
                value_next_states = np.array([x[1] for x in next_states])
                dones = [x[4] for x in mini_batch]
                # target q calculate q value(s_t+1)
                print('start to train!')
                pred_actions = actor.target.predict([input_next_states,value_next_states])
                target_q_values = critic.target.predict([input_next_states,value_next_states,pred_actions])
                # target q 估计q value(s_t)
                done_index = list(filter(lambda x:dones[x]==True,range(batch_size)))
                target_q_values[done_index] = 0
                target_q_values = np.array(list(map(lambda x,y:discount*x+y,target_q_values,rewards)))
                # update main q
                critic.model.fit([input_states,value_states,actions],target_q_values)
                # update main pg
                pred = actor.model.predict([input_states,value_states])
                grads = critic.get_gradient(input_states,value_states,pred)
                actor.train(input_states,value_states,grads)
                # soft update target pg and target q
                critic.update()
                actor.update()
            state = next_state
            steps += 1
        print('============总收益: {}============'.format(steps,rews))
        reward_ls.append(rews)
    print(reward_ls)
    actor.model.save_weights('model/actor.h5')
    critic.model.save_weights('model/critic.h5')
    actor.target.save_weights('model/actor_target.h5')
    critic.model.save_weights('model/critic_target.h5')
    return reward_ls


def run(args,df):
    env = StockTradingEnv(df,args.time_step)
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    buff_size = int(args.buff_size)
    init_size = int(args.init_size)
    tau = float(args.tau)
    time_step = int(args.time_step)
    discount = .95
    state_size = [2,3]
    unit_size = config.unit_size
    lstm_units = config.lstm_units
    lstm_units1 = config.lstm_units1
    unit_size1 = config.unit_size1
    tf.reset_default_graph()
    with tf.Session() as sess:
        keras.backend.set_session(sess)
        ddpg(sess, env, epochs,batch_size, buff_size, init_size, tau, state_size, time_step,
             unit_size,lstm_units, unit_size1, lstm_units1, discount)
        print('===finish!===')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=200)
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--buff_size", default=10000)
    parser.add_argument("--init_size", default=2000)
    parser.add_argument("--tau",default=.005)
    parser.add_argument("--time_step", default=10)
    parser.add_argument("--disount", default=.95)
    args = parser.parse_args()
    df = pd.read_csv('data/only_return.csv')
    df = df.iloc[:, 1:]
    print(df.head())
    avg_rewards = run(args,df)