# Reinforcement-Learning-global-portfolio
Code for master's thesis titled 'A reinforcement learning approach to optimizing a global stock portfolio'

The code for this thesis draws many elements from the FinRL library, found here - https://github.com/AI4Finance-LLC/FinRL. The library has been adapted for this thesis, with several aspects that are different. 

The structure of this repository is as follows: 

- The Main.ipynb file can be used to run the model and follow the output
- The Utilities.ipynb file can be used to preprocess and prepare the final dataset for the model, perform analyses (such as testing VIX's correlation with the market, calculating the turbulence index, calculating the final returns, Sharpe ratios and so on
- models.py, stored in the models folder contains the implementation for the A2C, PPO and DDPG libraries, built on the stable baselines library. It also contains the 'run_strategy' function that calls upon the models, and the validation and prediction functions. The prediction functions govern the agent's trading behavior, using the trained model as input for each epoch. The models are built upon the custom environments
- The 'env' folder consists of the training, validation and testing environments. The run_strategy function in models.py calls upon the training environment, the validation function utilizes the validation environment and the prediction environment makes use of the trading environment respetively. The environments are defined and utilized in a very similar manner to the FinRL library
- The 'Trained models' folder contains specific models that yielded the results explained in the thesis
- The 'Saved Results' folder contains necessary raw outputs for specific models whose outputs are discussed in the thesis. The function 'prep_result_data' in Utilities.ipynb is used to prepare the output from the raw results. 
- A seperate 'results' folder is needed to output the results of any model that is being run on the Main.ipynb notebook


A formal pacakge requirement file, and instructions to run the code will be added soon. For the moment, the code requires the following packages: numpy, pandas, os, time, stable-baselines and its dependencies, gym and its dependencies.  


PLEASE NOTE: The raw data for this thesis is not available due to copyright. Feel free to get in touch with me at zubain94@gmail.com if you would like to know about the characteristics of the dataset, or for advice on how to tune the model for your dataset. 


References: 
@article{finrl2020,
    author  = {Liu, Xiao-Yang and Yang, Hongyang and Chen, Qian and Zhang, Runjia and Yang, Liuqing and Xiao, Bowen and Wang, Christina Dan},
    journal = {Deep RL Workshop, NeurIPS 2020},
    title   = {FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance},
    url     = {https://arxiv.org/pdf/2011.09607.pdf},
    year    = {2020}
}

Please also note that the current version of stable baselines used here is based on Tensorflow 1.1X. A migration of code to Stable baselines 3.0 using PyTorch may be required in the near future. Using stable baselines with version 2.0 of Tensorflow is still under development. Python 3.6 and under can be used to run the code. 
