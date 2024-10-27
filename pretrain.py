import argparse
import numpy as np
import torch
import torch.nn.functional as F
import gym
from tqdm import tqdm
from model import Dynamics
import d4rl
import utils
import random
import os
from torch.utils.tensorboard import SummaryWriter
from main import snapshot_src
import json


parser = argparse.ArgumentParser()

parser.add_argument('--env', type=str, default='hopper-medium-v2')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_iters', type=int, default=int(5e5))
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--no_normalize', default=False, action='store_true')
parser.add_argument('--eval_data', default=0.0, type=float) # proportion of data used for evaluation
args = parser.parse_args()

work_dir = '../runs/{}/{}/{}/eval{}_norm{}_seed{}'.format(
     os.getcwd().split('/')[-1], 'pretrain', args.env, args.eval_data, not args.no_normalize, args.seed)
writer = SummaryWriter(work_dir)
with open(os.path.join(work_dir, 'args.json'), 'w') as f:
    json.dump(vars(args), f, sort_keys=True, indent=4)
snapshot_src('.', os.path.join(work_dir, 'src'), '.gitignore')
# Set seeds
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = 'cuda'
env = gym.make(args.env)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
if not args.no_normalize:
    mean, std = replay_buffer.normalize_states()
else:
    print("No normalize")
states = replay_buffer.state
actions = replay_buffer.action
next_states = replay_buffer.next_state

if args.eval_data:
    eval_size = int(states.shape[0] * args.eval_data)
    eval_idx = np.random.choice(states.shape[0], eval_size, replace=False)
    train_idx = np.setdiff1d(np.arange(states.shape[0]), eval_idx)
    eval_states = states[eval_idx]
    eval_actions = actions[eval_idx]
    eval_next_states = next_states[eval_idx]
    states = states[train_idx]
    actions = actions[train_idx]
    next_states = next_states[train_idx]
else:
    eval_states = None
    eval_actions = None
    eval_next_states = None

dynamics = Dynamics(state_dim, action_dim).to(device)

optimizer = torch.optim.Adam(dynamics.parameters(), lr=args.lr)

for step in tqdm(range(args.num_iters + 1), desc='train'):
    idx = np.random.choice(states.shape[0], args.batch_size)
    train_states = torch.from_numpy(states[idx]).to(device)
    train_actions = torch.from_numpy(actions[idx]).to(device)
    train_next_states = torch.from_numpy(next_states[idx]).to(device)
    
    pred_next_states = dynamics(train_states, train_actions)
    loss = F.mse_loss(pred_next_states, train_next_states)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 5000 == 0:
        print('step: %s, train_loss: %s' % (step,loss.item()))
        writer.add_scalar('pretrain/train_loss', loss.item(), step)
        if eval_states is not None and eval_actions is not None:
            dynamics.eval()
            with torch.no_grad():
                eval_states_tensor = torch.from_numpy(eval_states).to(device)
                eval_actions_tensor = torch.from_numpy(eval_actions).to(device)
                eval_next_states_tensor = torch.from_numpy(eval_next_states).to(device)
                pred = dynamics(eval_states_tensor, eval_actions_tensor)
                loss = F.mse_loss(pred, eval_next_states_tensor)
                print('step: %s, eval_loss: %s' % (step,loss.item()))
                writer.add_scalar('pretrain/eval_loss', loss.item(), step)
            dynamics.train()
    if step == args.num_iters:
        if not os.path.isdir('../SCAS_dynamics'):
            os.makedirs('../SCAS_dynamics')
        torch.save(dynamics.state_dict(), '../SCAS_dynamics/dynamics_%s_norm%s.pt' % (args.env, not args.no_normalize))
