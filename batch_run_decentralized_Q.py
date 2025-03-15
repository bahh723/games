import itertools
import subprocess

# Define all initial states (e.g., 'CCCC' to 'DDDD')
MEMORY = 1
type = "on"
eps = 0.01
states = [''.join(x) for x in itertools.product('CD', repeat=2 * MEMORY)]


# Run each state 3 times
for initial_state in states:
    for run_id in range(1, 4):
        print(f"Running simulation for {initial_state}, Run {run_id}...")
        subprocess.run(["python", "PD_Q_learn.py", "--init", initial_state, "--id", str(run_id), 
                        "--type", type, "--eps", str(eps), "--memory", str(MEMORY)])
