import numpy as np
import random
import matplotlib.pyplot as plt

def define_rewards(g):
    # Define the reward matrix dynamically using parameter g
    return {
        ("CC", "C", "C"): 2 * g, ("CC", "C", "D"): g,
        ("CC", "D", "C"): 2 + g, ("CC", "D", "D"): 2,
        ("CD", "C", "C"): 2 * g, ("CD", "C", "D"): g,
        ("CD", "D", "C"): 2 + g, ("CD", "D", "D"): 2,
        ("DC", "C", "C"): 2 * g, ("DC", "C", "D"): g,
        ("DC", "D", "C"): 2 + g, ("DC", "D", "D"): 2,
        ("DD", "C", "C"): 2 * g, ("DD", "C", "D"): g,
        ("DD", "D", "C"): 2 + g, ("DD", "D", "D"): 2,
    }

# Initialize Q-values
def initialize_q():
    actions = ["C", "D"]
    states = ["CC", "CD", "DC", "DD"]
    q_values = {state: {action: 0 for action in actions} for state in states}
    return q_values

# Choose action using epsilon-greedy policy
def choose_action(state, q_values, epsilon):
    if random.random() < epsilon:
        return random.choice(["C", "D"])
    return max(q_values[state], key=q_values[state].get)

# Update Q-values using Q-learning formula
def update_q(q_values, state, action, reward, next_state, alpha, gamma):
    max_next_q = max(q_values[next_state].values())
    q_values[state][action] += alpha * (reward + gamma * max_next_q - q_values[state][action])

def test_different_gammas(g, gammas, iterations=1000, alpha=0.1, epsilon=0.1, runs=10):
    cooperation_rates = []
    for gamma in gammas:
        total_cooperation_rate = 0

        # Run the simulation multiple times for the current gamma
        for _ in range(runs):
            cooperation_count = 0
            rewards = define_rewards(g)
            q1 = initialize_q()
            q2 = initialize_q()

            actions = ["C", "D"]
            states = ["CC", "CD", "DC", "DD"]
            state = "DD"  # Initial state

            for _ in range(iterations):
                # Choose actions for both players
                action1 = choose_action(state, q1, epsilon)
                action2 = choose_action(state[::-1], q2, epsilon)  # Reverse state for the opponent

                # Get rewards
                reward1 = rewards[(state, action1, action2)]
                reward2 = rewards[(state[::-1], action2, action1)]

                # Update Q-values
                next_state = action1 + action2
                update_q(q1, state, action1, reward1, next_state, alpha, gamma)
                update_q(q2, state[::-1], action2, reward2, next_state[::-1], alpha, gamma)

                # Track cooperation
                if action1 == "C" and action2 == "C":
                    cooperation_count += 1

                # Update state
                state = next_state

            # Calculate cooperation rate for this run
            total_cooperation_rate += cooperation_count / iterations

        # Average cooperation rate over all runs
        average_cooperation_rate = total_cooperation_rate / runs
        cooperation_rates.append(average_cooperation_rate)

    return cooperation_rates

# Define parameters
g = 1.5
gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Different gamma values to test
iterations = 10000
alpha = 0.1
epsilon = 0.1
runs = 20

# Run the test
cooperation_rates = test_different_gammas(g, gammas, iterations, alpha, epsilon, runs)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(gammas, cooperation_rates, marker='o')
plt.title("Average Cooperation Rate vs. Gamma")
plt.xlabel("Gamma (Discount Factor)")
plt.ylabel("Average Cooperation Rate")
plt.grid(True)
plt.show()