# -*- coding: utf-8 -*-
# @Time : 2019/4/25 10:01 PM
# @Author : ZHAOBOWEN467@pingan.com.cn
# @File : critic.py
# @Software: PyCharm
# @desc:

import tensorflow as tf
import keras
from keras.layers import Dense,LSTM,Input,concatenate
from keras.optimizers import Adam
from keras.models import Model

class Critic(object):
    """  Creat actor network, including online actor network and target network
    Attr:
        sess: tf.Session()
        tau: parameter for soft update
        state_size: last dimension of state
        time_step: length of time seires
        dropout: dropuot value, default for .5
        unit_size: units of dense layer
        lstm_units: units of lstm layer
        mdoel: online critic network
        target: taret critic network

    """
    def __init__(self,sess,tau,state_size,time_step,unit_size,lstm_units,dropout=.5):
        self.sess = sess
        keras.backend.set_session(self.sess)
        self.tau = tau
        self.state_size = state_size
        self.time_step = time_step
        self.dropout = dropout
        self.unit_size = unit_size
        self.lstm_units = lstm_units
        self.model,self.input_state, self.value_state, self.action = self._build()
        self.target, _, _, _ = self._build()
        # calculate the gradient of q value versus action
        self.gradients = tf.gradients(self.model.output, self.action)
        self.sess.run(tf.global_variables_initializer())

    def _build(self):
        """ build a actor network graph

        :return:  model, sequence input state, account information input
        """
        state_input = Input(shape=(self.time_step, self.state_size[0]), dtype=tf.float32, name='state_input')
        value_input = Input(shape=(self.state_size[1],),dtype=tf.float32,name='value_input')
        action = Input(shape=(2,),dtype=tf.float32,name='action')

        lstm = LSTM(units=self.lstm_units,return_sequences=False)
        lstm_output = lstm(state_input)

        input = concatenate([lstm_output,value_input, action], axis=-1, name='input')

        dense = Dense(units=self.unit_size,activation='tanh',name='dense')
        dense_output = dense(input)

        dense1 = Dense(units=1,name='dense1')
        output = dense1(dense_output)

        model = Model(inputs=[state_input,value_input,action], outputs=output)
        optimizer = Adam()
        model.compile(optimizer=optimizer, loss='mse')

        return model,state_input,value_input,action


    def get_gradient(self,input_state,value_state,actions):
        """ get the gradient of q value versus action

        :param input_state: time series input state, shape = [batch_size, time_step, embeding size]
        :param value_state: account information input (balance, buy_amout, sell-amount)
        :param actions: action at current time step
        :return:
        """
        return self.sess.run(self.gradients,
                        feed_dict={self.input_state:input_state,
                                   self.value_state:value_state,
                                   self.action:actions})[0]

    def update(self):
        """  update target network

        :return:
        """
        model_weights = self.model.get_weights()
        target_weights = self.target.get_weights()
        new_target_weight = list(map(lambda x, y: (x * self.tau + y * (1 - self.tau)), model_weights, target_weights))
        self.target.set_weights(new_target_weight)
