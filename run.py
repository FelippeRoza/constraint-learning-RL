from core.safety_layer import SafetyLayer
from core.envs import Highway
from core.rl_wrappers import SafeDDPG
from stable_baselines3 import DDPG

# config
buffer_path = 'data/buffer.obj'
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

rl_agent = DDPG("MlpPolicy", env, verbose=1)
rl_agent.learn(total_timesteps=time_steps)
rl_agent.save("data/DDPG")

safe_rl_agent = SafeDDPG("MlpPolicy", env, safe_layer, verbose=1)
safe_rl_agent.learn(total_timesteps=time_steps)
safe_rl_agent.save("data/safe_DDPG")
