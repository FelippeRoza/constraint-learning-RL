from core.safety_layer import SafetyLayer
from core.envs import Highway
from core.callbacks import TensorboardCallback
from stable_baselines3 import DDPG, PPO, A2C, SAC
import argparse
import distutils
import os


parser = argparse.ArgumentParser(description='Constrained Learning RL.')
parser.add_argument('--exp_name', default='exp1', help='name of the experiment.')
parser.add_argument('--rl_alg', default='DDPG', choices=['DDPG', 'PPO', 'A2C', 'SAC'], help='RL training algorithm.')
parser.add_argument('--train_steps', default=100000, type=int, help='RL algorithm training steps.')
parser.add_argument('--headless', default='True', help='No video output, suitable for running on a server.')
parser.add_argument('--use_safety_layer', default='True', help='Use a safety layer or not.')
parser.add_argument('--sl_load_folder', default=None,
                    help='Folder with safety layer weights. If not given, a new safety layer will be trained.')
parser.add_argument('--buffer_path', default='experiments/buffer/buffer.obj', help='Path with buffer collected previously.')
parser.add_argument('--buffer_size', default=500000, type=int, help='Buffer size (number of samples).')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size for training safety layer.')
parser.add_argument('--n_epochs', default=20, type=int, help='Number of epochs to train safety layer.')
args = parser.parse_args()

# prepare folders and environment
exp_dir = os.path.join('experiments', args.rl_alg + '_' + args.exp_name)
os.makedirs(exp_dir, exist_ok=True)

env = Highway(mode='continuous', safety_layer=None)
if bool(distutils.util.strtobool(args.headless)):
    env.config_mode('headless')

if bool(distutils.util.strtobool(args.use_safety_layer)):
    env.config_mode('discrete')
    safe_layer = SafetyLayer(env, buffer_size=args.buffer_size, buffer_path=args.buffer_path, n_epochs=args.n_epochs,
                             batch_size=args.batch_size)
    if args.sl_load_folder:
        safe_layer.load(args.sl_load_folder)
    else:
        safe_layer.train()
        safe_layer.save(os.path.join(exp_dir, 'SafetyLayer'))

    safety_status = 'safe'
    env.config_mode('continuous')
    env.add_safety_layer(safe_layer)
else:
    safety_status = 'unsafe'

# train RL agent
if args.rl_alg == 'PPO':
    rl_agent = PPO("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join('experiments', 'tensorboard'))
elif args.rl_alg == 'DDPG':
    rl_agent = DDPG("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join('experiments', 'tensorboard'))
elif args.rl_alg == 'A2C':
    rl_agent = A2C("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join('experiments', 'tensorboard'))
elif args.rl_alg == 'SAC':
    rl_agent = SAC("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join('experiments', 'tensorboard'))

rl_agent.learn(total_timesteps=args.train_steps, tb_log_name=safety_status + args.rl_alg + args.exp_name,
               callback=TensorboardCallback(env))
rl_agent.save(os.path.join(exp_dir, safety_status))
