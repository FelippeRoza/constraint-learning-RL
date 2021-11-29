from core.safety_layer import SafetyLayer
from core.envs import Highway
from core.rl_wrappers import SafeDDPG, TensorboardCallback
from stable_baselines3 import DDPG

# config
exp_name = 'exp1'
buffer_path = 'data/buffer' + exp_name + '.obj'  # change if want to save/load file with different name
buffer_size = 5000
batch_size = 32
n_epochs = 5
env = Highway(mode='discrete')

#  train safety layer
safe_layer = SafetyLayer(env, buffer_size, n_epochs=n_epochs, batch_size=batch_size)
safe_layer.collect_samples()  # not necessary if there is already a buffer saved

safe_layer.save_buffer(buffer_path)  # save buffer with collected samples
safe_layer.load_buffer(buffer_path)
safe_layer.train()

# train RL agent with safe actions
env.config_mode('continuous')
time_steps = 500

rl_agent = DDPG("MlpPolicy", env, verbose=1, tensorboard_log="data/tensorboard/")
rl_agent.learn(total_timesteps=time_steps, tb_log_name="unsafe_" + exp_name, callback=TensorboardCallback(env))
rl_agent.save("data/unsafe_" + exp_name)

safe_rl_agent = SafeDDPG("MlpPolicy", env, safety_layer=safe_layer, verbose=1, tensorboard_log="data/tensorboard/")
safe_rl_agent.learn(total_timesteps=time_steps, tb_log_name="safe_" + exp_name, callback=TensorboardCallback(env))
safe_rl_agent.save("data/safe_" + exp_name)
