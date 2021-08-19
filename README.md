# Reinforcement-Learning-global-portfolio
Code for master's thesis titled 'A reinforcement learning approach to optimizing a global stock portfolio'

The code for this thesis draws many elements from the FinRL library, found here - https://github.com/AI4Finance-LLC/FinRL. The library has been adapted for this thesis, with several aspects that are different. 

The structure of this repository is as follows: 

- The Main.ipynb file can be used to run the model and follow the output
- The Utilities.ipynb file can be used to preprocess and prepare the final dataset for the model, perform analyses (such as testing VIX's correlation with the market, calculating the turbulence index, calculating the final returns, Sharpe ratios and so on
- models.py, stored in the models folder contains the implementation for the A2C, PPO and DDPG libraries, built on the stable baselines library. It also contains the 'run_strategy' function that calls upon the models, and the validation and prediction functions. The prediction functions govern the agent's trading behavior, using the trained model as input for each epoch. The models are built upon the custom environments
- The 'env' folder consists of the training, validation and testing environments. The run_strategy function in models.py calls upon the training environment, the validation function utilizes the validation environment and the prediction environment makes use of the trading environment respetively. The environments are defined and utilized in a very similar manner to the FinRL library
- The 'Trained models' folder contains specific models that yielded the results explained in the thesis
- The 'Results' folder contains necessary raw outputs for specific models whose outputs are discussed in the thesis. The function 'prep_result_data' in Utilities.ipynb is used to prepare the output from the raw results. 


A formal pacakge requirement file, and instructions to run the code will be added soon. For the moment, the code requires the following packages: numpy, pandas, os, time, stable-baselines and its dependencies, gym and its dependencies.  
