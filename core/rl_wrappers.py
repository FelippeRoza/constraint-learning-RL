from stable_baselines3 import DDPG
import numpy as np
from typing import Optional, Tuple
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
from typing import Any, Dict
import torch

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

    def load(self, path):
        raise Exception('Load not defined')

    def save(self, *args, **kwargs):
        real_env = self.real_env
        safety_layer = self.safety_layer
        delattr(self, 'real_env')  # quick and dirty solution
        delattr(self, 'safety_layer')
        super(SafeDDPG, self).save(*args, **kwargs)
        self.real_env = real_env
        self.safety_layer = safety_layer


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, env, verbose=0, video_freq=10000):
        super(TensorboardCallback, self).__init__(verbose)
        self.env = env
        self.cum_violations = np.zeros(env.num_constraints)
        self.episode_c = [[] for i in range(self.env.num_constraints)]
        self._render_freq = video_freq

    def _on_step(self) -> bool:
        c = self.env.get_constraint_values()
        for i, c in enumerate(c):
            self.episode_c[i].append(c)
            if c > 0:
                self.cum_violations[i] += 1
            self.logger.record('safety/c'+str(i)+'_cum_violations', self.cum_violations[i])

        if self.n_calls % self._render_freq == 0:
            screens = []
            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                screen = self.env.render(mode="rgb_array")
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self.env,
                callback=grab_screens,
                n_eval_episodes=2,
                deterministic=True,
            )
            self.logger.record(
                "trajectory/video",
                Video(torch.ByteTensor([screens]), fps=1),
                exclude=("stdout", "log", "json", "csv"),
            )

        return True

    def _on_rollout_end(self) -> None:
        for i, c in enumerate(self.episode_c):
            self.logger.record('safety/c' + str(i) + '_max', np.max(c))
        self.episode_c = [[] for i in range(self.env.num_constraints)]
