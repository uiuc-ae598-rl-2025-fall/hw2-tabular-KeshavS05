import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


ACTIONS = [0, 1, 2, 3]  # maps to left, down, right, up
DELTAS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

nS = 16
nA = 4

def make_frozenlake(is_slippery: bool):
   env = gym.make(
      'FrozenLake-v1',
      desc=None,
      map_name="4x4",
      is_slippery=is_slippery,
   )
   return env


def policy_return(env, Q, gamma=0.95, episodes=100, max_steps=100):
   returns = []
   episode_length = []
   for _ in range(episodes):
      epi_len = 0
      state, _ = env.reset()
      G = 0.0
      for t in range(max_steps):
         action = np.argmax(Q[state])
      
         next_state, reward, terminated, truncated, _ = env.step(action)
         epi_len += 1
         G += (gamma**t) * reward
         if terminated or truncated:
            break
         
         state = next_state
         
      returns.append(G)
      episode_length.append(epi_len)
      
   return np.mean(returns), np.mean(episode_length)

def generate_episode(env, pi):
   state, _ = env.reset()
   done = False
   episode = []
   while not done:
      action = np.random.choice(ACTIONS, p=pi[state])
      next_state, reward, terminated, truncated, _ = env.step(action)
      episode.append((state, action, reward))
      
      done = terminated or truncated
      state = next_state
   
   return episode

def monte_carlo(env, gamma=0.95, eps=0.4, episodes=5000):
   
   # Initializations
   Q = np.zeros((nS, nA))      
   returns = [[[] for _ in range(nA)] for _ in range(nS)]
   pi = np.full((nS, nA), 1.0/nA, dtype=float)
   episode_length = []
   evalation_returns = []
   total_steps = []
   total_steps_count = 0
   for ep in range(episodes):
      episode = generate_episode(env, pi)
      total_steps_count += len(episode)
      episode_rewardless = [(s, a) for (s, a, _) in episode]
         
      G = 0.0
      for t in range(len(episode) - 1, -1, -1):
         state, action, reward = episode[t]
         G = gamma * G + reward
         if ((state, action) not in episode_rewardless[:t]):
            returns[state][action].append(G)
            Q[state][action] = np.mean(returns[state][action])
            new_action = np.argmax(Q[state])
            
            for a in ACTIONS:
               if a == new_action:
                  pi[state][a] = 1 - eps + eps/len(ACTIONS)
               else:
                  pi[state][a] = eps/len(ACTIONS)
                  
      if (ep % 100 == 0):
         total_steps.append(total_steps_count)
         eval_ret, epi_len = policy_return(env, Q)
         evalation_returns.append(eval_ret)
         episode_length.append(epi_len)
                  
   return Q, pi, episode_length, evalation_returns, total_steps


def epsilon_greedy(Q, state, eps):
   greedy_action = np.argmax(Q[state])
   probs = [1 - eps + eps/len(ACTIONS) if a == greedy_action else eps/len(ACTIONS) for a in ACTIONS]
   return np.random.choice(ACTIONS, p=probs) 


def sarsa(env, gamma=0.95, eps=0.2, episodes=5000, alpha = 0.002):
   
   # Initializations
   Q = np.full((nS, nA), 1/nA)
   for ind in [5, 7, 11, 12, 15]:
      Q[ind] = np.array([0, 0, 0, 0])
   episode_length = []
   evalation_returns = []
   total_steps = []
   total_steps_count = 0
   
   for ep in range(episodes):
      state, _ = env.reset()
      action = epsilon_greedy(Q, state, eps)
      done = False
         
      while not done:
         next_state, reward, terminated, truncated, _ = env.step(action)
         total_steps_count += 1
         
         if terminated or truncated:
            done = True
            Q[state][action] = Q[state][action] + alpha * (reward - Q[state][action])
         
         else:
            next_action = epsilon_greedy(Q, next_state, eps)
            Q[state][action] = Q[state][action] + alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            state = next_state
            action = next_action
            
      if (ep % 100 == 0):
         total_steps.append(total_steps_count)
         eval_ret, epi_len = policy_return(env, Q)
         evalation_returns.append(eval_ret)
         episode_length.append(epi_len)
            
   return Q, episode_length, evalation_returns, total_steps


def q_learning(env, gamma=0.95, eps=0.2, episodes=5000, alpha = 0.002):
   
   # Initialization
   Q = np.full((nS, nA), 0.25)
   for ind in [5, 7, 11, 12, 15]:
      Q[ind] = np.array([0, 0, 0, 0])
   episode_length = []
   evalation_returns = []
   total_steps = []
   total_steps_count = 0
   
   for ep in range(episodes):
      state, _ = env.reset()
      done = False
         
      while not done:
         action = epsilon_greedy(Q, state, eps)
         next_state, reward, terminated, truncated, _ = env.step(action)
         total_steps_count += 1
         
         if terminated or truncated:
            done = True
            Q[state][action] = Q[state][action] + alpha * (reward - Q[state][action])
            
         else:
            Q[state][action] = Q[state][action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
            state = next_state 
            
      if (ep % 100 == 0):
         total_steps.append(total_steps_count)
         eval_ret, epi_len = policy_return(env, Q)
         evalation_returns.append(eval_ret)
         episode_length.append(epi_len)
            
   return Q, episode_length, evalation_returns, total_steps
      

def plot_eval_return_vs_steps(steps, eval_returns, title):
   plt.figure()
   plt.plot(steps, eval_returns)
   plt.xlabel("Training env steps")
   plt.ylabel("Evaluation return (discounted)")
   plt.title(title)
   plt.show()

def plot_avg_length_vs_steps(steps, avg_lengths, title):
   plt.figure()
   plt.plot(steps, avg_lengths)
   plt.xlabel("Training env steps")
   plt.ylabel("Avg evaluation episode length")
   plt.title(title)
   plt.show()


ARROWS = {0:"←", 1:"↓", 2:"→", 3:"↑"}  # left, down, right, up

def plot_q_heatmaps(Q, algo_name=""):
   fig, axes = plt.subplots(1, 4, figsize=(12, 3))
   for a in range(4):
      grid = Q[:, a].reshape(4, 4)
      im = axes[a].imshow(grid)
      for r in range(4):
         for c in range(4):
               axes[a].text(c, r, f"{grid[r,c]:.2f}", ha="center", va="center")
      axes[a].set_title(f"Q(s,a={a})")
      fig.colorbar(im, ax=axes[a])
   plt.suptitle(f"{algo_name} — Q(s,a) heatmaps")
   plt.show()


def plot_policy_grid_from_pi(pi, title="MC policy (ε-soft argmax)"):
   greedy = np.argmax(pi, axis=1).reshape(4, 4)
   plt.figure(figsize=(4, 4))
   plt.imshow(np.zeros((4,4)), cmap='summer')
   for r in range(4):
      for c in range(4):
         a = int(greedy[r, c])
         plt.text(c, r, ARROWS[a], ha="center", va="center", fontsize=12)
   plt.title(title)
   plt.xticks([])
   plt.yticks([])
   plt.show()

def plot_policy_grid_from_Q(Q, title="Greedy policy from Q"):
   J = np.argmax(Q, axis=1).reshape(4, 4)
   plt.figure(figsize=(4, 4))
   plt.imshow(np.zeros((4,4)), cmap='summer')
   for r in range(4):
      for c in range(4):
         a = int(J[r, c])
         plt.text(c, r, ARROWS[a], ha="center", va="center", fontsize=12)
   plt.title(title)
   plt.xticks([])
   plt.yticks([])
   plt.show()











env = make_frozenlake(is_slippery=True)

# MC
Q, pi, episode_length, evalation_returns, total_steps,  = monte_carlo(env, episodes=5000)
print("done")
plot_eval_return_vs_steps(total_steps, evalation_returns, "MC (slippery=True)")
plot_avg_length_vs_steps(total_steps, episode_length, "MC avg episode length (slippery=True)")
plot_q_heatmaps(Q, "MC Q-heatmap (slippery)")
plot_policy_grid_from_pi(pi, "MC policy (slippery)(ε-soft argmax)")

# SARSA
Q, episode_length, evalation_returns, total_steps = sarsa(env, episodes=5000)
plot_eval_return_vs_steps(total_steps, evalation_returns, "SARSA (slippery=True)")
plot_avg_length_vs_steps(total_steps, episode_length, "SARSA avg episode length (slippery=True)")
plot_q_heatmaps(Q, "SARSA Q-heatmap (slippery)")
plot_policy_grid_from_Q(Q, "SARSA policy (slippery)(ε-soft argmax)")

# Q-learning
Q, episode_length, evalation_returns, total_steps = q_learning(env, episodes=5000)
plot_eval_return_vs_steps(total_steps, evalation_returns, "Q-learning (slippery=True)")
plot_avg_length_vs_steps(total_steps, episode_length, "Q-learning avg episode length (slippery=True)")
plot_q_heatmaps(Q, "Q-learning Q-heatmap (slippery)")
plot_policy_grid_from_Q(Q, "Q-learning policy (slippery)(ε-soft argmax)")


env = make_frozenlake(is_slippery=False)

# MC
Q, pi, episode_length, evalation_returns, total_steps = monte_carlo(env, episodes=5000)
plot_eval_return_vs_steps(total_steps, evalation_returns, "MC (slippery=False)")
plot_avg_length_vs_steps(total_steps, episode_length, "MC avg episode length (slippery=False)")
plot_q_heatmaps(Q, "MC Q-heatmap (not slippery)")
plot_policy_grid_from_pi(pi, "MC policy (not slippery)(ε-soft argmax)")

# SARSA
Q, episode_length, evalation_returns, total_steps = sarsa(env, episodes=5000)
plot_eval_return_vs_steps(total_steps, evalation_returns, "SARSA (slippery=False)")
plot_avg_length_vs_steps(total_steps, episode_length, "SARSA avg episode length (slippery=False)")
plot_q_heatmaps(Q, "SARSA Q-heatmap (not slippery)")
plot_policy_grid_from_Q(Q, "SARSA policy (not slippery)(ε-soft argmax)")

# Q-learning
Q, episode_length, evalation_returns, total_steps = q_learning(env, episodes=5000)
plot_eval_return_vs_steps(total_steps, evalation_returns, "Q-learning (slippery=False)")
plot_avg_length_vs_steps(total_steps, episode_length, "Q-learning avg episode length (slippery=False)")
plot_q_heatmaps(Q, "Q-learning Q-heatmap (not slippery)")
plot_policy_grid_from_Q(Q, "Q-learning policy (not slippery)(ε-soft argmax)")