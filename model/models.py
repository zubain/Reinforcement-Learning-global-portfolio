# common library
import pandas as pd
import numpy as np
import time
import gym

# # RL models from stable-baselines
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import DDPG

from stable_baselines.ddpg.policies import DDPGPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines.common.vec_env import DummyVecEnv
from preprocessing.preprocessors import *
from config import config

# customized env
from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_validation import StockEnvValidation
from env.EnvMultipleStock_trade import StockEnvTrade


# This function takes in the training environment, name of the model to be saved, the type of policy 
# to be used 'MlpPolicy' and 'MlpLnLstmPolicy' the two tested policy types, and the number of timesteps
# and runs the A2C model that is sourced from the stable baselines library
def train_A2C(env_train, model_name, policy_type, timesteps=25000):

    start = time.time()
    model = A2C(policy_type, env_train, verbose=0)

# In case the model is run with 
#     model = A2C('MlpLnLstmPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model


# This function takes in the training environment, name of the model to be saved, the type of policy 
# to be used 'MlpPolicy' and 'MlpLnLstmPolicy' the two tested policy types, and the number of timesteps
# and runs the DDPG model that is sourced from the stable baselines library
def train_DDPG(env_train, model_name, policy_type, timesteps=10000):

    # add the noise objects for DDPG
    n_actions = env_train.action_space.shape[-1]
    param_noise = None

    # Other noise functions could also be used (Gaussian noise for example), but they are not tested here
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    start = time.time()
    model = DDPG(policy_type, env_train, param_noise=param_noise, action_noise=action_noise)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (DDPG): ', (end-start)/60,' minutes')
    return model

def train_PPO(env_train, model_name, policy_type, timesteps=50000):
    """PPO model"""

    start = time.time()
    model = PPO2(policy_type, env_train, ent_coef = 0.005, nminibatches = 8)
   
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model


def DRL_prediction(df, model, name, last_state, iter_num, unique_trade_date, rebalance_window, initial
#                    , turbulence_threshold
                  ):
    ### make a prediction based on trained model###

    ## trading env
    trade_data = data_split(df, start=unique_trade_date[iter_num - rebalance_window], end=unique_trade_date[iter_num])
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data,
                                                   initial=initial,
                                                   previous_state=last_state,
#                                                    turbulence_threshold=turbulence_threshold,
                                                   model_name=name,
                                                   iteration=iter_num)])
    obs_trade = env_trade.reset()

    for i in range(len(trade_data.index.unique())):
        action, _states = model.predict(obs_trade)
#         print(i)
#         print("action: ", action)
#         print("_states: ", _states)
        obs_trade, rewards, dones, info = env_trade.step(action)
#         print("obs_trade: ", obs_trade)
#         print("rewards: ", rewards)
#         print("dones: ", dones)
#         print("info: ", info)
        if i == (len(trade_data.index.unique()) - 2):
            # print(env_test.render())
            last_state = env_trade.render()

    df_last_state = pd.DataFrame({'last_state': last_state})
    df_last_state.to_csv('results/last_state_{}_{}.csv'.format(name, i), index=False)

    return last_state


def DRL_validation(model, test_data, test_env, test_obs) -> None:
    ###validation process###
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)


def get_validation_sharpe(iteration):
    ###Calculate Sharpe ratio based on validation results###
    df_total_value = pd.read_csv('results/account_value_validation_{}.csv'.format(iteration), index_col=0)
    df_total_value.columns = ['account_value_train']
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / \
             df_total_value['daily_return'].std()
    return sharpe


def run_strategy(df, unique_trade_date, rebalance_window, validation_window, data_start_date, 
                 model_type, timesteps) -> None:
    
    last_state = []
    a2c_sharpe_list = []
    ppo_sharpe_list = []
    ddpg_sharpe_list = []

    # insample_turbulence = df[(df.date<'2015-10-01') & (df.date>=data_start_date)]
    # insample_turbulence = insample_turbulence.drop_duplicates(subset=['date'])
    # insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)
    
    start = time.time()
    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        print("============================================")
        ## initial state is empty
        if i - rebalance_window - validation_window == 0:
            # inital state
            initial = True
        else:
            # previous state
            initial = False
           
        # Tuning turbulence index based on historical data
        # Turbulence lookback window is one quarter

        end_date_index = df.index[df["date"] == unique_trade_date[i - rebalance_window - validation_window]].to_list()[-1]
#         print(end_date_index)
        start_date_index = end_date_index - validation_window*30 + 1
#         print(start_date_index)

#         historical_turbulence = df.iloc[start_date_index:(end_date_index + 1), :]
    
#         historical_turbulence = historical_turbulence.drop_duplicates(subset=['date'])
#         historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

#         if historical_turbulence_mean > insample_turbulence_threshold:
#             # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
#             # then we assume that the current market is volatile,
#             # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
#             # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
#             turbulence_threshold = insample_turbulence_threshold
#         else:
#             # if the mean of the historical data is less than the 90% quantile of insample turbulence data
#             # then we tune up the turbulence_threshold, meaning we lower the risk
#             turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
        
        # Here an empirical threshold level of the vix is taken as the turbulence threshold
#         turbulence_threshold = 30
#         print("turbulence_threshold: ", turbulence_threshold)
        
        ############## Environment Setup starts ##############
        ## training env
        train = data_split(df, start=data_start_date, end=unique_trade_date[i - rebalance_window - validation_window])
        # train = data_split(df, start='2009-01-01', end=unique_trade_date[i - rebalance_window - validation_window])
        print(train.shape)
        env_train = DummyVecEnv([lambda: StockEnvTrain(train)])

        ## validation env
        validation = data_split(df, start=unique_trade_date[i - rebalance_window - validation_window],
                                end=unique_trade_date[i - rebalance_window])

        env_val = DummyVecEnv([lambda: StockEnvValidation(validation,
#                                                           turbulence_threshold=turbulence_threshold,
                                                          iteration=i)])

        obs_val = env_val.reset()
        ############## Environment Setup ends ##############

        ############## Training and Validation starts ##############
        print("======Model training from: " & data_start_date & " to ",
              unique_trade_date[i - rebalance_window - validation_window])
        # print("training: ",len(data_split(df, start=20090000, end=test.datadate.unique()[i-rebalance_window]) ))
        # print("==============Model Training===========")

        print("======" & model_type & " Training========")
        if (model_type=="A2C"):
            model = train_A2C(env_train, model_name="A2C_Z_{}".format(i), timesteps=timesteps)
            print("======A2C Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
                  unique_trade_date[i - rebalance_window])  
            DRL_validation(model=model_a2c, test_data=validation, test_env=env_val, test_obs=obs_val)
            sharpe_a2c = get_validation_sharpe(i)
            print("A2C Sharpe Ratio: ", sharpe_a2c)

            a2c_sharpe_list.append(sharpe_a2c)

        
        if (model_type=="PPO"):
            print("======PPO Training========")
            model_ppo = train_PPO(env_train, model_name="PPO_Z_{}".format(i), timesteps=timesteps)
            print("======PPO Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
                  unique_trade_date[i - rebalance_window])  
            DRL_validation(model=model_ppo, test_data=validation, test_env=env_val, test_obs=obs_val)
            sharpe_ppo = get_validation_sharpe(i)
            print("PPO Sharpe Ratio: ", sharpe_ppo)
        
            ppo_sharpe_list.append(sharpe_ppo)


        if (model_type=="DDPG"):
            print("======DDPG Training========")
            model_ddpg = train_DDPG(env_train, model_name="DDPG_10k_dow_{}".format(i), timesteps=timesteps)
            print("======DDPG Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
                  unique_trade_date[i - rebalance_window])
            DRL_validation(model=model_ddpg, test_data=validation, test_env=env_val, test_obs=obs_val)
            sharpe_ddpg = get_validation_sharpe(i)
            
            ddpg_sharpe_list.append(sharpe_ddpg)
        
        
        ############## Trading starts ##############
        print("======Trading from: ", unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])

        if (model_type=="A2C"):
            last_state = DRL_prediction(df=df, model=model_a2c, name="a2c",
                                        last_state=last_state, iter_num=i,
                                        unique_trade_date=unique_trade_date,
                                        rebalance_window=rebalance_window,
    #                                   turbulence_threshold=turbulence_threshold,
                                        initial=initial)
        

        if (model_type=="PPO"):
            last_state = DRL_prediction(df=df, model=model_ppo, name="ppo",
                                        last_state=last_state, iter_num=i,
                                        unique_trade_date=unique_trade_date,
                                        rebalance_window=rebalance_window,
    #                                   turbulence_threshold=turbulence_threshold,
                                        initial=initial)


        if (model_type=="DDPG"):
            last_state = DRL_prediction(df=df, model=model_ddpg, name="ddpg",
                                        last_state=last_state, iter_num=i,
                                        unique_trade_date=unique_trade_date,
                                        rebalance_window=rebalance_window,
    #                                   turbulence_threshold=turbulence_threshold,
                                        initial=initial)


        print("============Trading Done============")
        ############## Trading ends ##############

    end = time.time()
    print("Strategy took: ", (end - start) / 60, " minutes")
        
 
