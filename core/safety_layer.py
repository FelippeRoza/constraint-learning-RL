import numpy as np
import pickle
import torch
import cvxpy as cp

from core.replay_buffer import ReplayBuffer
from core.net import Net

def for_each(f, l):
    for x in l:
        f(x)

class SafetyLayer:
    def __init__(self, env, buffer_size, n_epochs=10, batch_size=32, lr=1e-4, layer_dims=[64, 20], env_mode='manual'):
        self.env = env
        self.lr = lr  # learning rate
        self.layer_dims = layer_dims
        self.buffer_size = buffer_size
        self.buffer = ReplayBuffer(buffer_size)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_constraints = 0
        self.models = None

    @staticmethod
    def _batch_as_tensor(batch):
        return [torch.Tensor(i) for i in (batch["action"], batch["observation"], batch["c"], batch["c_next"])]

    def _create_models(self):
        if len(self.buffer) <= 0:
            raise Exception("Replay buffer is empty. Collect samples before training the constraint models.")
        sample = self.buffer.sample(1)
        in_dim = sample["observation"][0].shape[0]
        out_dim = sample["action"][0].shape[0]

        self.models = [Net(in_dim=in_dim, out_dim=out_dim, layer_dims=self.layer_dims) for i in range(self.env.num_constraints)]
        self.optimizers = [torch.optim.Adam(model.parameters(), lr=self.lr) for model in self.models]
        self.n_constraints = len(self.models)

    def collect_samples(self, render=True):
        done = True
        for i in range(self.buffer_size):
            if done:
                observation = self.env.reset()
            c = self.env.get_constraint_values()
            observation_next, _, done, _ = self.env.step(self.env.action_space.sample())
            c_next = self.env.get_constraint_values()
            if render:
                self.env.render()
            self.buffer.add({
                "action": np.array(list(self.env.vehicle.action.values())),
                "observation": observation.flatten(),  # observation represented as an 1D array
                "c": c,
                "c_next": c_next
            })
            observation = observation_next
        print('Buffer is full')

    def get_safe_action(self, observation, action, const):
        obs, a, c = (torch.Tensor(observation), torch.Tensor(action), torch.Tensor(const))
        g = [model(obs.view(1, -1)).squeeze() for model in self.models]
        g = [x.detach().cpu().numpy() for x in g]

        # TODO: use Lagrangian closed form solution when only one constraint is used
        # optimization problem
        x = cp.Variable(self.n_constraints)
        cost = cp.sum_squares((x - action) * 0.5)
        prob = cp.Problem(cp.Minimize(cost),
                          [g @ x + c <= 0])  # [g_i[0].T @ x <= -c_i[0]])
        prob.solve()
        action_new = x.value

        return action_new

    def train(self):
        self._create_models()
        loss_fn = torch.nn.MSELoss()

        for epoch in range(self.n_epochs):
            for i, model in enumerate(self.models):
                loss_list = []
                for batch in self.buffer.get_sequential(self.batch_size):

                    # compute loss
                    action, obs, c, c_next = self._batch_as_tensor(batch)
                    g = model(obs)
                    c_next_pred = c[:, i] + torch.bmm(g.view(g.shape[0],1,-1), action.view(action.shape[0],-1,1)).squeeze()
                    loss = loss_fn(c_next_pred, c_next[:, i])

                    # Backpropagation
                    self.optimizers[i].zero_grad()
                    loss.backward()
                    self.optimizers[i].step()
                    loss_list.append(np.asarray(loss.item()))

                print('Loss model', i, 'epoch', epoch, ':', np.mean(loss_list))

    def save_buffer(self, path):
        file_handler = open(path, 'wb')
        pickle.dump(self.buffer, file_handler)
        print('Saved buffer at', path)

    def load_buffer(self, path):
        file_handler = open(path, 'rb')
        self.buffer = pickle.load(file_handler)
        if self.models is None:
            self._create_models()
        print('Loaded buffer from', path)