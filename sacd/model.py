import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical


def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class QNetwork(BaseNetwork):

    def __init__(self, d_model, obs_dim, num_actions, dueling_net=False):
        super().__init__()

        if not dueling_net:
            self.head = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, num_actions))
        else:
            self.a_head = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, num_actions))
            self.v_head = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, num_actions))

        self.dueling_net = dueling_net

    def forward(self, states):
        if not self.dueling_net:
            return self.head(states)
        else:
            a = self.a_head(states)
            v = self.v_head(states)
            return v + a - a.mean(1, keepdim=True)


class TwinnedQNetwork(BaseNetwork):
    def __init__(self, d_model, obs_dim, num_actions, dueling_net=False):
        super().__init__()
        self.Q1 = QNetwork(d_model, obs_dim, num_actions, dueling_net)
        self.Q2 = QNetwork(d_model, obs_dim, num_actions, dueling_net)

    def forward(self, states):
        q1 = self.Q1(states)
        q2 = self.Q2(states)
        return q1, q2


class CateoricalPolicy(BaseNetwork):

    def __init__(self, d_model, obs_dim, num_actions):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, num_actions)
        )

    def act(self, states):
        action_logits = self.head(states)
        greedy_actions = torch.argmax(
            action_logits, dim=1, keepdim=True)
        return greedy_actions

    def sample(self, states):
        if not self.shared:
            states = self.conv(states)

        action_probs = F.softmax(self.head(states), dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability.
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs