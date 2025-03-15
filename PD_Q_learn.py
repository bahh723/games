import random
import itertools
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Game Parameters (ensure t > r > p > s)
# -------------------------------
T = 1.00   # Temptation payoff (when defecting against cooperation)
S = 0.00   # Sucker’s payoff (when cooperating against defection)
R = 0.10   # Reward for mutual cooperation (r < 0.5)
P = 0.05   # Punishment for mutual defection

# -------------------------------
# Q-learning Parameters
# -------------------------------
ALPHA = 0.1      # Learning rate
GAMMA = 0.9      # Discount factor
EPSILON = 0.1    # Exploration rate
NUM_ROUNDS = 500000  # Total number of game rounds
W = 500
MEMORY = 3

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
# Off-Diagonal Cooperation Policy for P1
#
# This function implements the following rules:
# 1. If the three rounds form a cyclic permutation of ["CC", "DC", "CD"],
#    then P1 plays the first letter of the next pair in that cycle.
#    (E.g., state "DCCDCC" splits into ["DC", "CD", "CC"] which is a rotation of the cycle,
#     so the next expected pair is "DC" and P1 plays 'D'.)
# 2. Else if the state ends with "CC" (pattern: XXXXCC), then play D.
# 3. Else if the state matches pattern "XXCCDC" (i.e. positions 2-3 are "CC" and positions 4-5 are "DC"), then play C.
# 4. Otherwise, play D.
# -------------------------------
def off_diagonal_coop_policy(state, player):
    if state[-2:] == "CD":  
        if player=="P1": 
            return 'D'
        if player=="P2": 
            return 'C'
    elif state[-2:] == "DC": 
        if player=="P1":
            return 'C'
        if player=="P2": 
            return 'D'
    else: 
        return 'D'
    
    
    #rounds = []
    #for i in range(MEMORY): 
    #    rounds.append(state[2*i: 2*(i+1)])
    #canonical_cycle = ["CC", "DC", "CD"]
    # Check if rounds follow a cyclic permutation of the canonical cycle.
    #for r in range(3):
    #    rotated = canonical_cycle[r:] + canonical_cycle[:r]
    #    if rounds == rotated:
    #        # Determine the next expected pair in the cycle.
    #        next_pair = rotated[(2 + 1) % 3]
    #        if player=="P1":
    #            return next_pair[0]  # P1's action is the first letter of the pair.
    #        if player=="P2":
    #            return next_pair[1]  # P2's action is the second letter of the pair.
    # If not following the cycle:
    #if state[-2:] == "CC":  # Matches XXXXCC
    #    if player=="P1":
    #        return 'D'
    #    if player=="P2":
    #        return 'C'
    #if state[2:4] == "CC" and state[4:6] == "DC":  # Matches XXCCDC
    #    if player=="P1":
    #        return 'C'
    #    if player=="P2":
    #        return 'D'
    #return 'D'

def off_diagonal_coop_policy_player(player):
    return lambda state: off_diagonal_coop_policy(state, player)

# -------------------------------
# All Defect Policy for P2: always play D.
# -------------------------------
def all_defect_policy(state):
    return 'D'

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
            if init_policy is not None:
                best_action = init_policy(state)
                # Initialize the Q-value for the preferred action to 1.0 and the other to 0.0.
                self.Q[state] = {'C': 1.0 if best_action == 'C' else 0.0,
                                 'D': 1.0 if best_action == 'D' else 0.0}
            else:
                self.Q[state] = {'C': 0.0, 'D': 0.0}
    
    def get_action(self, state):
        # If a fixed policy is set, always use it.
        if self.fixed_policy is not None:
            return self.fixed_policy(state)
        # Otherwise, use an epsilon-greedy selection based on Q-values.
        if random.random() < self.epsilon:
            return random.choice(['C', 'D'])
        q_vals = self.Q[state]
        if q_vals['C'] == q_vals['D']:
            return random.choice(['C', 'D'])
        return 'C' if q_vals['C'] > q_vals['D'] else 'D'
    
    def update(self, state, action, reward, next_state):
        # If using a fixed policy, skip Q-value updates.
        if self.fixed_policy is not None:
            return
        max_next = max(self.Q[next_state].values())
        self.Q[state][action] += self.alpha * (reward + self.gamma * max_next - self.Q[state][action])

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
                        fixed_policy=off_diagonal_coop_policy_player("P1") if fix_p1 else None,
                        init_policy=off_diagonal_coop_policy_player("P1"))
agent2 = QLearningAgent(epsilon=EPSILON, alpha=ALPHA, gamma=GAMMA,
                        fixed_policy=off_diagonal_coop_policy_player("P2") if fix_p2 else None,
                        init_policy=off_diagonal_coop_policy_player("P2"))

# -------------------------------
# Initialize the starting state.
# We assume the game starts with 3 rounds of mutual cooperation: "CCCCCC"
# -------------------------------
current_state = 'C' * (2*MEMORY) 
rewards1, rewards2 = [], []


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


# Plot results
plt.figure(figsize=(10, 5))
plt.plot(smooth_r1, label='Agent 1 Reward', alpha=0.7)
plt.plot(smooth_r2, label='Agent 2 Reward', alpha=0.7)
plt.xlabel('Rounds')
plt.ylabel('Average Reward')
plt.title(f'Moving Average Rewards (Window={W})')
plt.legend()
plt.show()

# Plot the results
plt.figure(figsize=(10, 5))
for key, values in state_counts.items():
    plt.plot(range(W, NUM_ROUNDS), values, label=key)
plt.xlabel("Rounds")
plt.ylabel("Proportion in Window")
plt.title("Proportion of States in CC, CD, DC, DD over Time")
plt.legend()
plt.show()


