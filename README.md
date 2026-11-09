# Malis Project

Fishing Season simulator and agent experiments for the MALIS project. The simulator is in `fishing.py`, agent implementations and the `play()` helper are in `agents.py`, and the experiments are in `experiment-2-contextual-completed.ipynb`.

## Start the project

The repository includes a Python virtual environment in `.malis_project` with the packages used by the project and notebook. From the project directory, activate it before running code.

### macOS and Linux

```bash
source .malis_project/bin/activate
```

### Windows PowerShell

```powershell
.malis_project\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.malis_project\Scripts\activate.bat
```

Once activated, the `python` and `jupyter` commands use this environment. To leave it, run `deactivate`.

> Python virtual environments can contain machine-specific paths and executables. If `.malis_project` does not run on your operating system or after moving the project, recreate it with Python 3.10 and install NumPy, Jupyter, pandas, and Matplotlib.

## Run an agent

With the environment activated, run this from the project directory. It creates a reproducible Season 1 bay, runs the example random agent, and prints its total catch:

```bash
python -c 'from fishing import FishingBay; from agents import RandomAgent, play; bay = FishingBay(season=1, seed=0); total = play(RandomAgent(seed=0), bay); print(f"Total catch: {total} kg over {bay.days} days")'
```

Replace `RandomAgent` with another agent defined in `agents.py`, such as `GreedyAgent`, `EpsilonGreedyAgent`, `UCBAgent`, or `HumanRuleAgent`. For example:

```python
from fishing import FishingBay
from agents import EpsilonGreedyAgent, play

bay = FishingBay(season=3, seed=42)
agent = EpsilonGreedyAgent(seed=42, epsilon=0.1, step_size=0.2)
total = play(agent, bay)
print(f"Total catch: {total} kg")
```

Save the example as a `.py` file and run it with `python your_file.py`, or enter the lines in a Python session. Seasons 1, 2, and 3 have 30, 10, and 30 days respectively. Use `ContextualBay` with a `ContextualAgent` (for example, `ContextualLinUCBAgent`) for contextual experiments.

## Run the experiment notebook

Start Jupyter from the project directory with the virtual environment activated:

```bash
jupyter lab
```

Alternatively, run `jupyter notebook`. Open `experiment-2-contextual-completed.ipynb` and run its cells from top to bottom. The notebook is associated with the `malis_project` kernel; if prompted to choose a kernel in Jupyter, select that kernel (it may be displayed as `.malis_project (3.10.12)`). The notebook loads the included `fishing_log_*.json` files and runs simulations and comparisons; some cells evaluate many bays and may take a while.

## Use the simulator and logs directly

`fishing.py` can be imported without running an agent framework. For example, rebuild the bay from a saved log and inspect the choices and catches:

```python
from fishing import load_log, choices, FishingBay

log = load_log("fishing_log_dakio.json")
spots, catches = choices(log, season=1)
bay = FishingBay.from_log(log, season=1)
print("Human total:", sum(catches), "kg")
print("Best possible:", bay.best_possible(), "kg")
```

Use `load_logs(path)` to load a JSON file containing multiple logs or all `.json` logs in a folder. The included logs are examples; replace their paths with your own game logs as needed.

## Project files

- `fishing.py` — simulator, contextual simulator, and log-loading helpers.
- `agents.py` — agent base classes, example agents, and `play()`.
- `experiment-2-contextual-completed.ipynb` — simulation experiments, comparisons, and plots.
- `.malis_project/` — included Python virtual environment.
- `fishing_log_*.json` — sample game logs used by the notebook.
