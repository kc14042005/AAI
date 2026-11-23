import random, time, copy
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

print("Gymnasium:", gym.__version__, "| PyTorch:", torch.__version__)
SEED = 42
random.seed(SEED); np.random.seed(SEED); _ = torch.manual_seed(SEED)
torch.set_num_threads(1)

TAXI_ID = max((k for k in gym.registry if k.startswith("Taxi-v")), key=lambda k: int(k.split("-v")[-1]))
print("Environment:", TAXI_ID)
env = gym.make(TAXI_ID)
n_states, n_actions = env.observation_space.n, env.action_space.n
print("States:", n_states, "| Actions:", n_actions, "(0 south, 1 north, 2 east, 3 west, 4 pickup, 5 drop-off)")

Q = np.zeros((n_states, n_actions))
alpha, gamma = 0.1, 0.99
epsilon, eps_min, eps_decay = 1.0, 0.01, 0.9995
episodes = 6000
rewards = []

for ep in range(episodes):
    state, _ = env.reset(seed=SEED + ep)
    total, done = 0, False
    while not done:
        action = env.action_space.sample() if random.random() < epsilon else int(np.argmax(Q[state]))
        next_state, reward, terminated, truncated, _ = env.step(action)
        target = reward + gamma * np.max(Q[next_state]) * (not terminated)
        Q[state, action] += alpha * (target - Q[state, action])
        state, total, done = next_state, total + reward, terminated or truncated
    epsilon = max(eps_min, epsilon * eps_decay)
    rewards.append(total)
    if (ep + 1) % 1000 == 0:
        print(f"Episode {ep + 1:5d} average reward (last 1000) {np.mean(rewards[-1000:]):7.2f} epsilon {epsilon:.3f}")

plt.figure(figsize=(8, 3.5))
plt.plot(np.convolve(rewards, np.ones(100) / 100, mode="valid"))
plt.xlabel("Episode"); plt.ylabel("Reward (100-episode moving average)")
plt.title("Q-learning on " + TAXI_ID); plt.show()

test_rewards, steps = [], []
for ep in range(100):
    s, _ = env.reset(seed=10000 + ep); done, tot, n = False, 0, 0
    while not done:
        s, r, term, trunc, _ = env.step(int(np.argmax(Q[s]))); tot += r; n += 1; done = term or trunc
    test_rewards.append(tot); steps.append(n)
print(f"Greedy policy over 100 test episodes: mean reward {np.mean(test_rewards):.2f}, "
      f"mean steps {np.mean(steps):.1f}, success rate {np.mean(np.array(test_rewards) > 0):.0%}")

env_vis = gym.make(TAXI_ID, render_mode="ansi")
s, _ = env_vis.reset(seed=7)
print(env_vis.render())
done, n = False, 0
while not done:
    s, r, term, trunc, _ = env_vis.step(int(np.argmax(Q[s]))); n += 1; done = term or trunc
print(env_vis.render()); print("Passenger delivered in", n, "steps, final reward", r)

env = gym.make("CartPole-v1")
obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n
print("Observation: cart position, cart velocity, pole angle, pole angular velocity | Actions:", n_actions)

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, n_actions))

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity)

    def push(self, *t):
        self.buf.append(t)

    def sample(self, n):
        s, a, r, s2, d = zip(*random.sample(self.buf, n))
        return (torch.tensor(np.array(s), dtype=torch.float32), torch.tensor(a),
                torch.tensor(r, dtype=torch.float32), torch.tensor(np.array(s2), dtype=torch.float32),
                torch.tensor(d, dtype=torch.float32))

random.seed(SEED); np.random.seed(SEED); _ = torch.manual_seed(SEED)
policy_net, target_net = QNetwork(), QNetwork()
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=3e-4, amsgrad=True)
memory = ReplayBuffer(20000)
GAMMA, BATCH, TAU = 0.99, 128, 0.005
eps, EPS_MIN, EPS_DECAY = 1.0, 0.01, 0.98

def evaluate(net, n=5, seed0=9000):
    scores = []
    for k in range(n):
        s, _ = env.reset(seed=seed0 + k); done, tot = False, 0
        while not done:
            with torch.no_grad():
                s, r, term, trunc, _ = env.step(int(net(torch.tensor(s, dtype=torch.float32)).argmax()))
            tot += r; done = term or trunc
        scores.append(tot)
    return np.mean(scores)

episode_rewards, start = [], time.time()
best_score, best_weights = -1, None

for episode in range(1, 501):
    state, _ = env.reset(seed=SEED + episode); total, done = 0, False
    while not done:
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = int(policy_net(torch.tensor(state, dtype=torch.float32)).argmax())
        next_state, reward, terminated, truncated, _ = env.step(action)
        memory.push(state, action, reward, next_state, float(terminated))
        state, total, done = next_state, total + reward, terminated or truncated
        if len(memory.buf) >= 500:
            s, a, r, s2, d = memory.sample(BATCH)
            q = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q_target = r + GAMMA * target_net(s2).max(1).values * (1 - d)
            loss = nn.functional.smooth_l1_loss(q, q_target)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_value_(policy_net.parameters(), 100); optimizer.step()
            with torch.no_grad():
                for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
                    tp.mul_(1 - TAU).add_(TAU * pp)
    eps = max(EPS_MIN, eps * EPS_DECAY)
    episode_rewards.append(total)
    if episode % 25 == 0:
        score = evaluate(policy_net)
        if score >= best_score:
            best_score, best_weights = score, copy.deepcopy(policy_net.state_dict())
        if episode % 50 == 0:
            print(f"Episode {episode:3d} mean reward (last 50) {np.mean(episode_rewards[-50:]):6.1f} "
                  f"greedy eval {score:6.1f} epsilon {eps:.3f} ({time.time() - start:.0f}s)")

policy_net.load_state_dict(best_weights)
print("Restored best checkpoint, greedy evaluation score:", best_score)

plt.figure(figsize=(8, 3.5))
plt.plot(episode_rewards, alpha=0.4, label="episode reward")
plt.plot(np.convolve(episode_rewards, np.ones(20) / 20, mode="valid"), label="20-episode average")
plt.axhline(500, color="grey", ls="--", label="maximum (500)")
plt.xlabel("Episode"); plt.ylabel("Reward"); plt.title("DQN on CartPole-v1"); plt.legend(); plt.show()

test = []
for ep in range(20):
    s, _ = env.reset(seed=5000 + ep); done, tot = False, 0
    while not done:
        with torch.no_grad():
            s, r, term, trunc, _ = env.step(int(policy_net(torch.tensor(s, dtype=torch.float32)).argmax()))
        tot += r; done = term or trunc
    test.append(tot)
print("Greedy DQN policy over 20 test episodes:", test)
print(f"Mean test reward: {np.mean(test):.1f} (CartPole is considered solved at an average of 475)")
