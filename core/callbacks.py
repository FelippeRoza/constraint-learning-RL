import numpy as np
import torch
from typing import Any, Dict
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, env, verbose=0, video_freq=10000):
        super(TensorboardCallback, self).__init__(verbose)
        self.env = env
        self.thresh_violations = np.zeros(env.num_constraints)
        self.collisions = 0
        self.out_of_road = 0
        self.episode_c = [[] for i in range(self.env.num_constraints)]
        self._render_freq = video_freq

    def _on_step(self) -> bool:
        constraints = self.env.get_constraint_values()

        for i, c in enumerate(constraints):
            self.episode_c[i].append(c)
            if c > 0:
                self.thresh_violations[i] += 1
            self.logger.record('safety/c'+str(i)+'_cum_violations', self.thresh_violations[i])

        if self.env.vehicle.crashed:
            self.collisions += 1

        if not self.env.on_road:
            self.out_of_road += 1

        self.logger.record('safety/cum_collisions', self.collisions)
        self.logger.record('safety/cum_out_of_road', self.out_of_road)

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
                Video(torch.ByteTensor([screens]), fps=5),
                exclude=("stdout", "log", "json", "csv"),
            )

        return True

    def _on_rollout_end(self) -> None:
        for i, c in enumerate(self.episode_c):
            self.logger.record('safety/c' + str(i) + '_max', np.max(c))
        self.episode_c = [[] for i in range(self.env.num_constraints)]
