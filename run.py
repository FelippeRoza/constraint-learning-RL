from core.safety_layer import SafetyLayer
from core.envs import Highway
from core.callbacks import TensorboardCallback
from stable_baselines3 import DDPG, PPO
import os

# config
exp_name = 'exp_name'
rl_algorithm = 'PPO'
headless = True             # avoid displaying video output, suitable for running on a server
use_safety_layer = True
sl_weight_folder = None     # folder with safety layer weights for loading. If None it'll train a new model
buffer_path = None          # change if there is a buffer saved
buffer_size = 500000
batch_size = 32
n_epochs = 5
training_steps = 100000           # number of training steps for the agent

# prepare folders and environment
exp_dir = os.path.join('experiments', rl_algorithm + '_' + exp_name)
os.makedirs(exp_dir, exist_ok=True)

env = Highway(mode='continuous', safety_layer=None)
if headless:
    env.config_mode('headless')

if use_safety_layer:
    env.config_mode('discrete')
    safe_layer = SafetyLayer(env, buffer_size, n_epochs=n_epochs, batch_size=batch_size)
    if sl_weight_folder:
        safe_layer.load(sl_weight_folder)
    else:
        if buffer_path is None:
            buffer_path = os.path.join(exp_dir, 'buffer.obj')
            safe_layer.collect_samples()  # not necessary if there is already a buffer saved
            safe_layer.save_buffer(buffer_path)  # save buffer with collected samples
        safe_layer.load_buffer(buffer_path)
        safe_layer.train()
        safe_layer.save(os.path.join(exp_dir, 'SafetyLayer'))

    safety_status = 'safe'
    env.config_mode('continuous')
    env.add_safety_layer(safe_layer)
else:
    safety_status = 'unsafe'

# train RL agent
if rl_algorithm == 'PPO':
    rl_agent = PPO("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join('experiments','tensorboard'))
elif rl_algorithm == 'DDPG':
    rl_agent = DDPG("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join('experiments', 'tensorboard'))

rl_agent.learn(total_timesteps=training_steps, tb_log_name=safety_status + rl_algorithm + exp_name,
               callback=TensorboardCallback(env))
rl_agent.save(os.path.join(exp_dir, safety_status))
