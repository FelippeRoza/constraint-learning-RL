import copy
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
import cvxpy as cp
import os
from tqdm import tqdm
import multiprocessing
from core.dataset import Buffer
from core.net import Net


def for_each(f, l):
    for x in l:
        f(x)


class SafetyLayer:
    def __init__(self, env, buffer_size=10000, buffer_path='buffer.obj', n_epochs=10, batch_size=32,
                 lr=1e-4, layer_dims=[10, 5], n_workers=1):
        self.env = env
        self.lr = lr  # learning rate
        self.layer_dims = layer_dims
        self.buffer_size = buffer_size
        self.buffer_path = buffer_path
        self.n_workers = n_workers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_constraints = len(env.get_constraint_values())
        self.model = None

    def _create_model(self, obs_dim, action_dim):
        in_dim = obs_dim
        out_dim = action_dim * self.n_constraints

        self.model = Net(in_dim=in_dim, out_dim=out_dim, layer_dims=self.layer_dims)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def sample_collection_worker(self, n_samples, env):
        done = True
        a_list, s_list, c_list, c_next_list = [], [], [], []
        for i in tqdm(range(n_samples)):
            if i > n_samples/2:
                env.config_mode('continuous')
            if done:
                observation = env.reset()
            c = env.get_constraint_values()
            action = env.action_space.sample()
            observation_next, _, done, _ = env.step(action)
            c_next = env.get_constraint_values()

            if env.__class__.__name__ == 'Highway':
                a_list.append(np.array(list(env.vehicle.action.values())))
            else:
                a_list.append(action)
            s_list.append(observation)
            c_list.append(c)
            c_next_list.append(c_next)

            observation = observation_next

        return {'action': a_list, 'observation': s_list, 'c': c_list, 'c_next': c_next_list}

    def collect_samples(self):
        print('Started collecting samples')
        n_samples = int(self.buffer_size/self.n_workers)

        with multiprocessing.Pool(processes=self.n_workers) as pool:
            results = [pool.apply_async(self.sample_collection_worker, (n_samples, copy.deepcopy(self.env))) for i in range(self.n_workers)]
            pool.close()
            pool.join()
        results = [result.get() for result in results]

        buffer = {'action': [], 'observation': [], 'c': [], 'c_next': []}
        for r in results:
            buffer['action'].extend(r['action'])
            buffer['observation'].extend(r['observation'])
            buffer['c'].extend(r['c'])
            buffer['c_next'].extend(r['c_next'])
        self.save_buffer(buffer)

    def get_safe_action(self, observation, action, const):
        obs, a, c = (torch.Tensor(observation), torch.Tensor(action), torch.Tensor(const))
        g = self.model(obs.unsqueeze(0).unsqueeze(0)).view((c.shape[0], a.shape[0]))
        g = g.detach().cpu().numpy()

        # use Lagrangian closed form solution when only one constraint is used
        if len(const) == 1:
            multiplier = (np.dot(g, action) + const) / np.dot(g, g)
            multiplier = np.clip(multiplier, 0, np.inf)
            # Calculate correction
            correction = np.max(multiplier) * g[np.argmax(multiplier)]
            action_new = action - correction
        else:
            x = cp.Variable(len(action))
            cost = cp.sum_squares((x - action) * 0.5)
            prob = cp.Problem(cp.Minimize(cost),
                              [g @ x + c <= 0])  # [g_i[0].T @ x <= -c_i[0]])
            prob.solve()
            action_new = x.value

        if np.isnan(action_new).any():
            action_new = None
        return action_new

    def train(self):
        loss_fn = torch.nn.MSELoss()
        if not os.path.isfile(self.buffer_path):
            print('Buffer in', self.buffer_path, 'does not exist.')
            self.collect_samples()
        dataset = Buffer(buffer_path=self.buffer_path)

        self._create_model(dataset[1]['observation'].shape[0], dataset[1]['action'].shape[0])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        for epoch in range(self.n_epochs):
            loss_list = []
            for i_batch, batch in enumerate(dataloader):

                # compute loss
                action, obs, c, c_next = batch['action'].to(device), batch['observation'].to(device), \
                                         batch['c'].to(device), batch['c_next'].to(device)
                g = self.model(obs)
                c_next_pred = c + torch.bmm(g.view(g.shape[0], c.shape[1], action.shape[1]),
                                                  action.view(action.shape[0], -1, 1)).squeeze()
                loss = loss_fn(c_next_pred, c_next)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_list.append(np.asarray(loss.item()))

            print('Loss epoch', epoch, ':', np.mean(loss_list))
        self.model.to("cpu")

    def save_buffer(self, buffer):
        os.makedirs(os.path.dirname(self.buffer_path), exist_ok=True)
        file_handler = open(self.buffer_path, 'wb')
        pickle.dump(buffer, file_handler)
        print('Saved buffer at', self.buffer_path)

    def save(self, path):
        torch.save(self.model, path)
        print('Saved safety layer model at', path)

    def load(self, path):
        self.model = torch.load(path)
        print('Loaded safety layer model from', path)
