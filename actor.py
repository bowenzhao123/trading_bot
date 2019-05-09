# -*- coding: utf-8 -*-
# @Time : 2019/4/25 9:11 PM
# @Author : ZHAOBOWEN467@pingan.com.cn
# @File : actor.py
# @Software: PyCharm
# @desc:

import tensorflow as tf
import numpy as np
import pandas as pd
import keras
from keras.layers import Dense,LSTM,Dropout,Input,concatenate
from keras.optimizers import Adam
from keras.models import Model


class Actor(object):
    """  Creat actor network, including online actor network and target network
    Attr:
        sess: tf.Session()
        tau: parameter for soft update
        state_size: last dimension of state
        time_step: length of time seires
        dropout: dropuot value, default for .5
        unit_size: units of dense layer
        lstm_units: units of lstm layer
        mdoel: online actor network
        target: taret actor network

    """
    def __init__(self,sess,tau,state_size,time_step,unit_size,lstm_units,dropout=.5):
        self.sess = tf.get_default_session()
        keras.backend.set_session(self.sess)
        self.tau = tau
        self.state_size = state_size
        self.time_step = time_step
        self.dropout = dropout
        self.unit_size = unit_size
        self.lstm_units = lstm_units
        self.model, self.weights, self.state_input,self.value_input = self._build()
        self.target, _, _, _ = self._build()
        self.action_gradient = tf.placeholder(tf.float32, [None, 2])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights) # (gradients, variable needed to be updated)
        self.optimize = tf.train.AdamOptimizer().apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def _build(self):
        """ build a actor network graph

        :return:  model, weight of model, sequence input state, account information input
        """
        state_input = Input(shape=(self.time_step,self.state_size[0]),dtype=tf.float32,name='state_input')
        value_input = Input(shape=(self.state_size[1],),dtype=tf.float32,name='value_input')
        lstm = LSTM(units=self.lstm_units,return_sequences=False)
        lstm_output = lstm(state_input)
        input = concatenate([lstm_output,value_input],axis=-1)

        dense = Dense(units=self.unit_size, activation='relu', name='dense')
        dense_output = dense(input)

        dense_discrete = Dense(units=1, activation='sigmoid', name='discrete')
        dense_cont = Dense(units=1,activation='sigmoid',name='continuous')

        action_type = dense_discrete(dense_output)
        amount = dense_cont(dense_output)
        output = concatenate([action_type,amount],axis=-1)

        model = Model(inputs=[state_input,value_input],outputs=output)
        return model, model.trainable_weights, state_input, value_input

    def train(self,input_state,value_state,grads):
        """ update the network

        :param input_state: time series input state, shape = [batch_size, time_step, embeding size]
        :param value_state: account information input (balance, buy_amout, sell-amount)
        :param grads: gradient of the q network
        :return:
        """
        self.sess.run(self.optimize, feed_dict={
            self.state_input: input_state,
            self.value_input: value_state,
            self.action_gradient: grads})

    def update(self):
        """ update target network

        :return:
        """
        model_weights = self.model.get_weights()
        target_weights = self.target.get_weights()
        new_target_weight = list(map(lambda x,y:(x*self.tau+y*(1-self.tau)),model_weights,target_weights))
        self.target.set_weights(new_target_weight)
