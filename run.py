from core.safety_layer import SafetyLayer
from core.envs import Highway
from core.rl_wrappers import SafeDDPG, TensorboardCallback
from stable_baselines3 import DDPG
import os

# config
exp_name = 'exp1'
buffer_path = None  # change if want to save/load file with different name
buffer_size = 5000
batch_size = 32
n_epochs = 5
time_steps = 500  # number of training steps for the agent

# prepare folders and environment
exp_dir = os.path.join('experiments', exp_name)
os.makedirs(exp_dir, exist_ok=True)
env = Highway(mode='discrete')

#  train safety layer
safe_layer = SafetyLayer(env, buffer_size, n_epochs=n_epochs, batch_size=batch_size)
if buffer_path is None:
    buffer_path = os.path.join(exp_dir, 'buffer.obj')
    safe_layer.collect_samples()  # not necessary if there is already a buffer saved
    safe_layer.save_buffer(buffer_path)  # save buffer with collected samples
safe_layer.load_buffer(buffer_path)
safe_layer.train()
safe_layer.save(os.path.join(exp_dir, 'SafetyLayer'))

# train RL agent with safe actions
env.config_mode('continuous')

rl_agent = DDPG("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join('experiments','tensorboard'))
rl_agent.learn(total_timesteps=time_steps, tb_log_name="unsafe_" + exp_name, callback=TensorboardCallback(env))
rl_agent.save(os.path.join(exp_dir, 'unsafe_agent'))

safe_agent = SafeDDPG("MlpPolicy", env, safety_layer=safe_layer, verbose=1, tensorboard_log=os.path.join('experiments','tensorboard'))
safe_agent.learn(total_timesteps=time_steps, tb_log_name="safe_" + exp_name, callback=TensorboardCallback(env))
safe_agent.save(os.path.join(exp_dir, 'safe_agent'))
