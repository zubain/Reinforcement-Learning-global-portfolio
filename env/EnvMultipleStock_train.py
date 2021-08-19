# This code has been inspired by the FinRL library, with modifications made to tailor the code

########################################################################################
# Loading required packages
import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
########################################################################################

########################################################################################
# Setting parameters

# Max shares per trade is a normalization factor. The action space is limited to [-1, 1]
# Hence this factor translates the action space to the number of shares for each action
# 200 shares per trade is finally chosen
HMAX_NORMALIZE = 200

# Initial amount of money given to the agent
INITIAL_ACCOUNT_BALANCE=1000000

# Total number of stocks allowed in the portfolio
STOCK_DIM = 13

# Transaction fee: 0.05% is finally chosen
TRANSACTION_FEE_PERCENT = 0.0005

REWARD_SCALING = 1e-4

########################################################################################

# Defining the class
class StockEnvTrain(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df,day = 0):
        
        self.day = day
        self.df = df
        self.initial = initial
        self.previous_state = previous_state
        ## action_space normalization and shape is STOCK_DIM. The action space is compressed to [-1,1]
        self.action_space = spaces.Box(low = -1, high = 1,shape = (STOCK_DIM,)) 
        # Observation space comprises of the current portfolio balance + closing prices from the baskets 1-13 + 
        # number of shares owned per basket 1-13 + MACD per basket 1-13 + RSI per basket 1-13 + Volume per basket 1-13 +
        # Bollinger bands per basket 1-13 + VIX prices copied over all the 13 baskets per day 
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (STOCK_DIM*7 + 1,))
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day,:]
        # By default, the terminal state is False. It converts to True at the last iteration

        self.terminal = False     
        # initalize states as described in the observation space
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                      self.data.close.values.tolist() + \
                      [0]*STOCK_DIM + \
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist()  + \
                      self.data.volume.values.tolist() + \
                      self.data.boll.values.tolist() + \
                      self.data.turbulence.values.tolist()

        # initialize reward
        self.reward = 0
        self.cost = 0
        self.trades = 0
        # Store the initial account balance and initial reward balance
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        #self.reset()
        self._seed()
        self.model_name=model_name        
        self.iteration=iteration


    def _sell_stock(self, index, action):
        # perform sell action based on the sign of the action
        if self.state[index+STOCK_DIM+1] > 0:
            #update balance
            self.state[0] += \
            self.state[index+1]*min(abs(action),self.state[index+STOCK_DIM+1]) * \
             (1- TRANSACTION_FEE_PERCENT)

            self.state[index+STOCK_DIM+1] -= min(abs(action), self.state[index+STOCK_DIM+1])
            self.cost +=self.state[index+1]*min(abs(action),self.state[index+STOCK_DIM+1]) * \
             TRANSACTION_FEE_PERCENT
            self.trades+=1
        else:
            pass

    
    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        available_amount = self.state[0] // self.state[index+1]

        #update balance
        self.state[0] -= self.state[index+1]*min(available_amount, action)* \
                          (1+ TRANSACTION_FEE_PERCENT)

        self.state[index+STOCK_DIM+1] += min(available_amount, action)

        self.cost+=self.state[index+1]*min(available_amount, action)* \
                          TRANSACTION_FEE_PERCENT
        self.trades+=1
        
    def step(self, actions):
        # print(self.day)
#         print("Day: ", self.day)
        self.terminal = self.day >= len(self.df.index.unique())-1

        if self.terminal:
            plt.plot(self.asset_memory,'r')
            plt.savefig('results/account_value_train.png')
            plt.close()
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            

            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('results/account_value_train.csv')
            
            df_total_value.columns = ['account_value']
            df_total_value['daily_return']=df_total_value.pct_change(1)
            sharpe = (252**0.5)*df_total_value['daily_return'].mean()/ \
                  df_total_value['daily_return'].std()
#             print("Sharpe: ",sharpe)
            print("=================================")
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.to_csv('results/account_rewards_train.csv')
            
            
            return self.state, self.reward, self.terminal,{}

        else:

            actions = actions * HMAX_NORMALIZE

            
            begin_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))

            
            argsort_actions = np.argsort(actions)

            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]
            

            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = self.df.loc[self.day,:]   
            

            self.state =  [self.state[0]] + \
                    self.data.close.values.tolist() + \
                    list(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]) + \
                    self.data.macd.values.tolist() + \
                    self.data.rsi.values.tolist()  + \
                    self.data.volume.values.tolist() + \
                    self.data.boll.values.tolist() + \
                    self.data.turbulence.values.tolist()

            
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            self.asset_memory.append(end_total_asset)

            
            self.reward = end_total_asset - begin_total_asset            

            self.rewards_memory.append(self.reward)
            
            self.reward = self.reward*REWARD_SCALING
            



        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        self.rewards_memory = []
        #initiate state
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                      self.data.close.values.tolist() + \
                      [0]*STOCK_DIM + \
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist()  + \
                      self.data.volume.values.tolist() + \
                      self.data.boll.values.tolist() + \
                      self.data.turbulence.values.tolist()
        
        # iteration += 1 
        return self.state
    
    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

