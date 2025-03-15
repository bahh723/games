import random
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os



import argparse
# Argument parser setup
parser = argparse.ArgumentParser(description="Q-learning Prisoner's Dilemma Simulation")
parser.add_argument("--init", type=str, required=True, help="Initial state (e.g., 'CCCC', 'CDCD').")
parser.add_argument("--id", type=int, required=True, help="Run number (e.g., 1, 2, 3).")
parser.add_argument("--eps", type=float, required=True, help="epsilon parameter in Q learning")
parser.add_argument("--memory", type=int, required=True, help="number of memory")
parser.add_argument("--type", type=str, required=True, help="type of game (on or off)")
parser.add_argument("--rounds", type=int, default=40000, help="type of game (on or off)")
args = parser.parse_args()
current_state = args.init
# Create output directory if it doesn't exist
output_dir = f"results_{args.type}_memory{args.memory}_eps{args.eps}"
os.makedirs(output_dir, exist_ok=True)



# -------------------------------
# Game Parameters (ensure t > r > p > s)
# -------------------------------
if args.type == "off": 
    T = 1.00   # Temptation payoff (when defecting against cooperation)
    S = 0.00   # Sucker’s payoff (when cooperating against defection)
    R = 0.10   # Reward for mutual cooperation (r < 0.5)
    P = 0.05   # Punishment for mutual defection
elif args.type == "on": 
    T = 0.6   # Temptation payoff (when defecting against cooperation)
    S = 0.0   # Sucker’s payoff (when cooperating against defection)
    R = 0.5   # Reward for mutual cooperation (r < 0.5)
    P = 0.1   # Punishment for mutual defection




# -------------------------------
# Q-learning Parameters
# -------------------------------
ALPHA = 0.1      # Learning rate
GAMMA = 0.9      # Discount factor
EPSILON = args.eps    # Exploration rate
NUM_ROUNDS = args.rounds  # Total number of game rounds
W = 50
MEMORY = args.memory

# -------------------------------
# Option to fix a player's policy:
# Set fix_p1 = True to fix P1's policy, or fix_p2 = True to fix P2's.
# (If a policy is fixed, that agent always follows its initial rule and does not update Q-values.)
# -------------------------------
fix_p1 = False
fix_p2 = False

# -------------------------------
# Define all possible states.
# Each state is a 6-character string representing the joint actions of the last 3 rounds.
# For example, "DCCDCC" represents rounds: (D,C), (CD), (CC).
# There are 2^6 = 64 possible states.
# -------------------------------
states = [''.join(x) for x in itertools.product('CD', repeat=2*MEMORY)]

# -------------------------------
# Helper function: Format a state for printing (split rounds by "|")
# -------------------------------
def format_state(state):
    s = state[0:2]
    for i in range(1, MEMORY):
         s += '|' + state[2*i: 2*(i+1)]
    return s



# -------------------------------
# Q-Learning Agent Class
#
# Each agent maintains a Q-table for every state (64 states, 2 actions per state).
#
# Parameters:
# - fixed_policy: if not None, get_action always returns fixed_policy(state)
#                 and update() will do nothing.
# - init_policy: if provided, the Q-table is initialized so that the preferred action
#                (given by init_policy(state)) starts with a higher Q-value.
# -------------------------------
class QLearningAgent:
    def __init__(self, epsilon=EPSILON, alpha=ALPHA, gamma=GAMMA,
                 fixed_policy=None, init_policy=None):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.fixed_policy = fixed_policy
        self.Q = {}
        for state in states:
            self.Q[state] = {'C': 1.0/(1-GAMMA), 'D': 1.0/(1-GAMMA)}
    
    def get_action(self, state):
        # If a fixed policy is set, always use it.
        if self.fixed_policy is not None:
            return self.fixed_policy(state)
        # Otherwise, use an epsilon-greedy selection based on Q-values.
        if random.random() < self.epsilon:
            return random.choice(['C', 'D'])
        q_vals = self.Q[state]
        if q_vals['C'] == q_vals['D']:
            # return 'D'
            return random.choice(['C', 'D'])
        return 'C' if q_vals['C'] > q_vals['D'] else 'D'
    
    def update(self, state, action, reward, next_state):
        # If using a fixed policy, skip Q-value updates.
        if self.fixed_policy is not None:
            return
        max_next = max(self.Q[next_state].values())
        self.Q[state][action] += self.alpha * (reward + self.gamma * max_next - self.Q[state][action])

    def appendQdiff(self, Qdiffs): 
        for state in states: 
            Qdiffs[state].append(self.Q[state]['C'] - self.Q[state]['D'])
        return Qdiffs
# -------------------------------
# Create Agents with the desired initial policies.
#
# P1: Off-diagonal cooperation policy.
# P2: All defect policy.
#
# If fix_p1 (or fix_p2) is True, that player's get_action will always return the fixed policy,
# and its Q-values will not be updated.
# -------------------------------
agent1 = QLearningAgent(epsilon=EPSILON, alpha=ALPHA, gamma=GAMMA,
                        fixed_policy=None,
                        init_policy=None)
agent2 = QLearningAgent(epsilon=EPSILON, alpha=ALPHA, gamma=GAMMA,
                        fixed_policy=None,
                        init_policy=None)

# -------------------------------
# Initialize the starting state.
# We assume the game starts with 3 rounds of mutual cooperation: "CCCCCC"
# -------------------------------
rewards1, rewards2 = [], []
Qdiffs1, Qdiffs2 = {}, {}
for s in states: 
    Qdiffs1[s] = []
    Qdiffs2[s] = []  


# Tracking state occurrences
state_counts = {"CC": [], "CD": [], "DC": [], "DD": []}
window_states = []


# -------------------------------
# Run the Q-learning simulation.
# -------------------------------
for i in range(NUM_ROUNDS):
    # Each agent selects an action based on the current state.
    action1 = agent1.get_action(current_state)
    action2 = agent2.get_action(current_state)
    
    # Determine rewards from the one-shot Prisoner's Dilemma matrix.
    if action1 == 'C' and action2 == 'C':
        reward1, reward2, state_label = R, R, "CC"
    elif action1 == 'C' and action2 == 'D':
        reward1, reward2, state_label = S, T, "CD"
    elif action1 == 'D' and action2 == 'C':
        reward1, reward2, state_label = T, S, "DC"
    elif action1 == 'D' and action2 == 'D':
        reward1, reward2, state_label = P, P, "DD"
    rewards1.append(reward1)
    rewards2.append(reward2)
    
    # Update the state: drop the oldest round (first 2 characters) and append the current round's actions.
    next_state = current_state[2:] + action1 + action2
    
    # Each agent updates its Q-table.
    agent1.update(current_state, action1, reward1, next_state)
    agent2.update(current_state, action2, reward2, next_state)
    
    current_state = next_state

    window_states.append(state_label)
    if len(window_states) > W:
        window_states.pop(0)
    
    # Compute rolling proportions
    if i >= W:
        for key in state_counts:
            state_counts[key].append(window_states.count(key) / W)

    # compute Q diff
    Qdiffs1 = agent1.appendQdiff(Qdiffs1)
    Qdiffs2 = agent2.appendQdiff(Qdiffs2)

# Compute moving averages
def moving_avg(data, window):
    return np.convolve(data, np.ones(window)/window, mode='valid')

smooth_r1, smooth_r2 = moving_avg(rewards1, W), moving_avg(rewards2, W)

# -------------------------------
# Extract the learned (or fixed) policy from an agent's Q-table.
# For each state, choose the action with the higher Q-value (if tied, default to 'C').
# -------------------------------
def extract_policy(agent):
    policy = {}
    for state in states:
        q_vals = agent.Q[state]
        if q_vals['C'] >= q_vals['D']:
            policy[state] = 'C'
        else:
            policy[state] = 'D'
    return policy

policy1 = extract_policy(agent1)
policy2 = extract_policy(agent2)

# -------------------------------
# Separate the states into two groups:
# - States whose joint policy (P1 + P2) is not "DD".
# - States whose joint policy is "DD".
# -------------------------------
non_dd_states = []
dd_states = []
for state in sorted(states):
    joint_policy = policy1[state] + policy2[state]
    if joint_policy == "DD":
        dd_states.append(state)
    else:
        non_dd_states.append(state)

# -------------------------------
# Print the joint policies for both agents.
# For each state, we print the state in "XX|XX|XX" format followed by the concatenated actions from P1 and P2.
# The non-"DD" policies are printed first, then a separator, then the "DD" policies.
# -------------------------------
print("Learned joint policy (P1, P2):")
for state in non_dd_states:
    print(f"{format_state(state)}: {policy1[state]}{policy2[state]}")
print("----------")
for state in dd_states:
    print(f"{format_state(state)}: {policy1[state]}{policy2[state]}")






fig, axes = plt.subplots(4, 1, figsize=(8.5, 7))  # Create a figure with two subplots (stacked vertically)

# First subplot (Moving Average Rewards)
axes[0].plot(smooth_r1, label='Agent 1 Reward', alpha=0.7)
axes[0].plot(smooth_r2, label='Agent 2 Reward', alpha=0.7)
# axes[0].set_xlabel('Rounds')
axes[0].set_ylabel('Average Reward')
axes[0].set_title(f'Moving Average Rewards (Window={W})')
axes[0].legend()

# Second subplot (Proportion of States)
for key, values in state_counts.items():
    axes[1].plot(range(W, NUM_ROUNDS), values, label=key, alpha=0.5)
# axes[1].set_xlabel("Rounds")
axes[1].set_ylabel("Proportion in Window")
axes[1].set_title("Proportion of States in CC, CD, DC, DD over Time")
axes[1].legend()

for state in states: 
    axes[2].plot(Qdiffs1[state], label=state, alpha=0.7)
# axes[2].set_xlabel("Rounds")
axes[2].set_ylabel("Q[state][C] - Q[state][D]")
axes[2].set_title("Q[state][C] - Q[state][D]")
# axes[2].set_ylim(-0.5, 0.5)  # Set y-axis range
axes[2].legend(fontsize=4, markerscale=0.4, frameon=False, handlelength=1, loc="upper right", bbox_to_anchor=(1, 1))

# Fourth "plot" (Displaying text output)
axes[3].axis("off")  # Hide axes for text display
text_output = "Learned joint policy (P1, P2):\n"
count = 0
for state in states:
    text_output += f"{format_state(state)}: {policy1[state]}{policy2[state]}       "
    count += 1
    if count % 4 == 0: 
        text_output += "\n"
# text_output += "\n"
# for state in dd_states:
#     text_output += f"{format_state(state)}: {policy1[state]}{policy2[state]}       "
axes[3].text(0.1, 0.4, text_output, fontsize=10, verticalalignment="center", family="monospace")



plt.tight_layout()
filename = os.path.join(output_dir, f"{args.init}{args.id}.png")
plt.savefig(filename, dpi=300)
plt.close()
print(f"{args.init}{args.id}.png")