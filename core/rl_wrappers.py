from stable_baselines3 import DDPG
import numpy as np
from typing import Optional, Tuple
from stable_baselines3.common.noise import ActionNoise


class SafeDDPG(DDPG):
    def __init__(self, *args, **kwargs):
        self.safety_layer = kwargs.pop('safety_layer')
        super(SafeDDPG, self).__init__(*args, **kwargs)
        self.real_env = args[1]  # because stables_baselines create a dummy representation

    def _sample_action(
            self, learning_starts: int, action_noise: Optional[ActionNoise] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        action, buffer_action = super()._sample_action(learning_starts, action_noise)

        obs = self.real_env.observation_type.observe()
        c = self.real_env.get_constraint_values()
        safe_action = self.safety_layer.get_safe_action(obs, action[0], c)
        safe_action = np.expand_dims(safe_action, axis=0)
        return safe_action, buffer_action
