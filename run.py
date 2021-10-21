from core.safety_layer import SafetyLayer
from core.envs import Highway
import numpy as np

# config
buffer_path = 'data/buffer.obj'
buffer_size = 5000
batch_size = 32
n_epochs = 5
env = Highway()

#  train safety layer
safe_layer = SafetyLayer(env, buffer_size, n_epochs=n_epochs, batch_size=batch_size)
safe_layer.collect_samples()
safe_layer.save_buffer(buffer_path)
safe_layer.load_buffer(buffer_path)
safe_layer.train()

# TODO: train RL agent with safe actions
done = True
for i in range(1000):
    if done:
        observation = env.reset()
    c = env.get_constraint_values()
    observation_next, _, done, _ = env.step(0)  # action comes from keyboard

    action = np.array(list(env.vehicle.action.values()))
    safe_action = safe_layer.get_safe_action(observation, action, c)

    print(action, safe_action)

    env.render()
