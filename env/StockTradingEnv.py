# -*- coding: utf-8 -*-
# @Time : 2019/4/23 8:26 PM
# @Author : ZHAOBOWEN467@pingan.com.cn
# @File : StockTradingEnv.py
# @Software: PyCharm
# @desc:


import random
import gym
from gym import spaces
import pandas as pd
import numpy as np

INITIAL_ACCOUNT_BALANCE = 100000
MAX_STEPS = 400


class StockTradingEnv(object):
    """ Stock Trading Environment
    Attr:
        df: time series dataframe
        time_lag: time step
        newWorth: the total asset a trader has, including his balance, his buy account, and his sell account

    """
    def __init__(self,df,time_lag):
        self.df = df
        self.time_lag = time_lag
        self.start = 0
        self.netWorth = INITIAL_ACCOUNT_BALANCE
        self.balance = INITIAL_ACCOUNT_BALANCE


    def reset(self):
        """  reset the environment

        :return: initial state
        """
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.netWorth = INITIAL_ACCOUNT_BALANCE
        self.buy_amount = 0
        self.sell_amount = 0
        self.currentStep = np.random.randint(self.time_lag,2000)
        self.start = self.currentStep
        return self._nextObservation()

    def _nextObservation(self):
        """ generate the next state

        :return:
        """
        frame = np.array(self.df.loc[(self.currentStep - (self.time_lag - 1)):self.currentStep,['return_Low','return_Volume']].values)
        obs = (frame,np.array([self.balance/10000,self.buy_amount/10000,self.sell_amount/10000]))  # standardize the value
        return obs

    def _balance_update(self):
        """  upate the balance, sell acount, buy acount, based on returan rate

        :return:
        """
        return_rate = self.df.loc[self.currentStep, "return_Close"]
        self.buy_amount += return_rate * self.buy_amount
        self.sell_amount -= return_rate * self.sell_amount

    def _takeAction(self,action):
        """

        :param action: (0,1/3): buy, (1/3,2/3): hold, (2/3,0): sell
        :return:
        """
        actionType = action[0]
        amount = self.balance*action[1]  # the amount of money to buy/sell stock

        if actionType < 1/3:  # buy stock
            self.balance += self.sell_amount  # first clear sell account
            self.sell_amount = 0
            self.buy_amount += amount  # add amount to buy account
            self.balance-=amount  # update balance

        if actionType > 2/3: # sell stock
            self.balance += self.buy_amount  # first clear buy account
            self.buy_amount = 0
            self.sell_amount += amount  # add amount to sell account
            self.balance -= amount  # update balance


    def step(self, action):
        """  given action, return new state and reward

        :param action: the action take at this time step
        :return: next state, reward, and done flag
        """

        self.currentStep += 1  # add current step by 1
        self._takeAction(action)  # update buy/sell account
        self._balance_update()  # update balance account
        obs = self._nextObservation()
        newnetWorth = self.balance + self.buy_amount + self.sell_amount  #  calculate new networth

        reward = (newnetWorth - self.netWorth)/(INITIAL_ACCOUNT_BALANCE/MAX_STEPS)  # calculate rewards

        self.netWorth = newnetWorth  # update network
        done = (self.netWorth < INITIAL_ACCOUNT_BALANCE * .8 or (self.currentStep - self.start) > 400)
        return obs, reward, done, {}
