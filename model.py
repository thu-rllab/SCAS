import torch
import torch.nn.functional as F
from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dynamics(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Dynamics, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 256)
		self.l4 = nn.Linear(256, 256)
		self.l5 = nn.Linear(256, state_dim)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		out = F.relu(self.l1(sa))
		out = F.relu(self.l2(out))
		out = F.relu(self.l3(out))
		out = F.relu(self.l4(out))
		out = self.l5(out)
		return out