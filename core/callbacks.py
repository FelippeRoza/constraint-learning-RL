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

    def __init__(self, env, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.env = env
        self.thresh_violations = np.zeros(env.num_constraints)
        self.collisions = 0
        self.out_of_road = 0
        self.correction = []  # cummulative action correction
        self.episode_c = [[] for i in range(self.env.num_constraints)]

    def _on_step(self) -> bool:
        constraints = self.env.get_constraint_values()
        for i, c in enumerate(constraints):
            self.episode_c[i].append(c)
            if c > 0:
                self.thresh_violations[i] += 1
            self.logger.record('safety/c'+str(i)+'_cum_violations', self.thresh_violations[i])

        if self.env.crashed:
            self.collisions += 1
        if not self.env.on_road:
            self.out_of_road += 1
        self.correction.append(self.env.action_correction)

        self.logger.record('safety/cum_collisions', self.collisions)
        self.logger.record('safety/cum_out_of_road', self.out_of_road)
        self.logger.record('safety/not_solved', self.env.not_solved)  # optimization did not find solution

        return True

    def _on_rollout_end(self) -> None:
        for i, c in enumerate(self.episode_c):
            self.logger.record('safety/c' + str(i) + '_max', np.max(c))
        self.logger.record('safety/mean_action_correction', np.mean(self.correction))
        self.correction = []
        self.episode_c = [[] for i in range(self.env.num_constraints)]

