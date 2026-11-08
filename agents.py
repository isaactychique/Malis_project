"""Agents for Fishing Season (MALIS, Project 1).

This file is a skeleton. Write all your agents in it.

Do NOT modify the classes Agent and ContextualAgent, nor the function play():
they define how every agent interacts with the simulator, and they are used
to run your agents during grading.

How an agent works
------------------
For each season, play(agent, bay) calls:

    agent.reset()                     once, before the first day
    spot = agent.choose()             every day: return a spot, 0 to 4
    agent.update(spot, catch)         every day: the catch you observed

An agent only learns through update(). It must not look at the bay itself.

Contextual agents (Part IV, optional) follow the same pattern, but receive the
morning conditions x (numpy array [wind_kmh, waves_m, tide_m]):

    spot = agent.choose(x)
    agent.update(x, spot, catch)

Example
-------
    from fishing import FishingBay
    from agents import RandomAgent, play

    bay = FishingBay(season=1, seed=0)
    total = play(RandomAgent(seed=0), bay)
"""
import numpy as np
from fishing import ContextualBay, FishingBay


class Agent:
    """Base class for the agents of Parts II and III.

    Parameters
    ----------
    n_spots : int
        Number of fishing spots (5 in the game).
    seed : int or None
        Seed of the agent's own random generator, self.rng. Use self.rng for
        any random decision, so that results can be reproduced.
    """

    def __init__(self, n_spots=5, seed=None):
        self.n_spots = n_spots
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """Called once at the start of every season. Clear what the agent learned."""

    def choose(self):
        """Return the spot to fish today, an int between 0 and n_spots - 1."""
        raise NotImplementedError

    def update(self, spot, catch):
        """Called after each day with the spot chosen and the catch observed (kg)."""


class ContextualAgent:
    """Base class for the agents of Part IV (contextual bay).

    Same as Agent, except that choose() and update() also receive x, the
    conditions of the day: numpy array [wind_kmh, waves_m, tide_m].
    """

    def __init__(self, n_spots=5, seed=None):
        self.n_spots = n_spots
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """Called once at the start of every season. Clear what the agent learned."""

    def choose(self, x):
        """Return the spot to fish today given the conditions x."""
        raise NotImplementedError

    def update(self, x, spot, catch):
        """Called after each day with the conditions, the spot chosen and the catch observed."""


def play(agent, bay):
    """Play one full season of bay with agent. Returns the total catch (kg).

    Works with FishingBay and Agent, or with ContextualBay and ContextualAgent.
    The bay is reset first, so the same bay can be played by several agents.
    """
    contextual = isinstance(bay, ContextualBay)
    if contextual and not isinstance(agent, ContextualAgent):
        raise TypeError("a ContextualBay must be played by a ContextualAgent")
    if not contextual and not isinstance(agent, Agent):
        raise TypeError("a FishingBay must be played by an Agent")
    bay.reset()
    agent.reset()
    while not bay.done:
        if contextual:
            x = bay.observe_array()
            spot = int(agent.choose(x))
            catch = bay.fish(spot)
            agent.update(x, spot, catch)
        else:
            spot = int(agent.choose())
            catch = bay.fish(spot)
            agent.update(spot, catch)
    return bay.total


# ---------------------------------------------------------------------------
# Provided example agents (baselines)
# ---------------------------------------------------------------------------


class RandomAgent(Agent):
    """Chooses a spot uniformly at random every day. Learns nothing."""

    def choose(self):
        return int(self.rng.integers(self.n_spots))


class RandomContextualAgent(ContextualAgent):
    """Ignores the conditions and chooses a spot uniformly at random every day."""

    def choose(self, x):
        return int(self.rng.integers(self.n_spots))


# ---------------------------------------------------------------------------
# Your agents
# ---------------------------------------------------------------------------


class HumanRuleAgent(Agent):
    """Explore, commit to the best spot, re-check the runner-up if catches drop.

    Rule (applicable by hand):
      1. Fish each spot n_init times, in order 0..4.
      2. Fish the spot with the highest average of its last `memory` catches.
      3. Every day at that spot, look at the mean of the last `window` catches
         there. If it is below `floor` kg (and at least `window` days have
         passed since the last check), fish the `n_check` best other spots
         once each, then go back to step 2.

    Memory: the catches seen at each spot. Parameters:
      n_init : visits per spot in the initial exploration
      memory : number of recent catches per spot used for its estimate
               (small = adapts to the Season 3 change, large = less noisy)
      window : recent catches at the current spot used to detect a drop
      floor  : kg below which the spot is considered to have dropped
      n_check: number of other spots re-sampled after a drop
    """

    def __init__(self, n_spots=5, seed=None, n_init=1, memory=3,
                 window=2, floor=3.0, n_check=1):
        super().__init__(n_spots, seed)
        self.n_init, self.memory = n_init, memory
        self.window, self.floor, self.n_check = window, floor, n_check

    def reset(self):
        self.hist = [[] for _ in range(self.n_spots)]
        self.queue = [k for _ in range(self.n_init) for k in range(self.n_spots)]
        self.current = None
        self.since_check = 0

    def _estimate(self, k):
        h = self.hist[k][-self.memory:]
        return np.mean(h) if h else -np.inf

    def _best(self):
        return int(np.argmax([self._estimate(k) for k in range(self.n_spots)]))

    def choose(self):
        return self.queue[0] if self.queue else self.current

    def update(self, spot, catch):
        self.hist[spot].append(catch)
        if self.queue:
            self.queue.pop(0)
            if not self.queue:                 # exploration or re-check finished
                self.current = self._best()
                self.since_check = 0
            return
        self.since_check += 1
        recent = self.hist[self.current][-self.window:]
        if self.since_check >= self.window and np.mean(recent) < self.floor:
            others = sorted((k for k in range(self.n_spots) if k != self.current),
                            key=self._estimate, reverse=True)
            self.queue = others[:self.n_check]

# Task 4: add below at least two agents implementing methods from the reading
# (Sutton and Barto, Chapter 2), as subclasses of Agent. Document each class:
# what it keeps in memory, how it chooses, what its parameters mean.


class _ValueAgent(Agent):
    """Shared machinery for the action-value agents below (not an agent itself).

    Memory
    ------
    Q[a] : estimated mean catch (kg) of spot a.
    N[a] : number of times spot a has been fished this season.
    t    : number of days played this season.

    Update rule (Sutton and Barto, Sec. 2.4)
    ----------------------------------------
        Q[a] <- Q[a] + alpha * (catch - Q[a])
    with alpha = 1 / N[a] (sample average, default) or a constant step_size
    (better when the fish populations can change, as in season 3).

    Parameters
    ----------
    initial_value : float
        Starting value of every Q[a]. Values above the best possible mean
        (about 10 kg) give "optimistic initial values" (Sec. 2.6).
    step_size : float or None
        None -> sample average (alpha = 1/N[a]); a float in (0, 1] -> constant alpha.
    """

    def __init__(self, n_spots=5, seed=None, initial_value=0.0, step_size=None):
        super().__init__(n_spots=n_spots, seed=seed)
        self.initial_value = float(initial_value)
        self.step_size = step_size
        self.reset()

    def reset(self):
        self.Q = np.full(self.n_spots, self.initial_value, dtype=float)
        self.N = np.zeros(self.n_spots, dtype=int)
        self.t = 0

    def _argmax(self, values):
        """Index of the maximum, ties broken uniformly at random."""
        best = np.flatnonzero(values == np.max(values))
        return int(self.rng.choice(best))

    def update(self, spot, catch):
        self.t += 1
        self.N[spot] += 1
        alpha = 1.0 / self.N[spot] if self.step_size is None else self.step_size
        self.Q[spot] += alpha * (catch - self.Q[spot])


class GreedyAgent(_ValueAgent):
    """Pure greedy: always fish the spot with the highest estimated mean catch.

    Memory: Q (estimated means) and N (visit counts).
    Choice: argmax_a Q[a], ties broken at random. No exploration of its own.
    Parameters:
        initial_value : starting Q for every spot. With 0 (default) the agent
            locks onto the first spot that gives a positive catch; with an
            optimistic value (e.g. 10-12 kg) every spot is tried early on.
        step_size : None for sample averages, or a constant alpha.
    """

    def choose(self):
        return self._argmax(self.Q)


class EpsilonGreedyAgent(_ValueAgent):
    """Epsilon-greedy: explore with probability epsilon, otherwise act greedily.

    Memory: Q (estimated means) and N (visit counts).
    Choice: with probability epsilon pick a spot uniformly at random;
        otherwise pick argmax_a Q[a] (ties broken at random).
    Parameters:
        epsilon : exploration probability in [0, 1] (default 0.1).
        initial_value, step_size : see _ValueAgent. A constant step_size
            (e.g. 0.1-0.2) lets the agent track the change in season 3.
    """

    def __init__(self, n_spots=5, seed=None, epsilon=0.1, initial_value=0.0, step_size=None):
        self.epsilon = float(epsilon)
        super().__init__(n_spots=n_spots, seed=seed,
                         initial_value=initial_value, step_size=step_size)

    def choose(self):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_spots))
        return self._argmax(self.Q)


class UCBAgent(_ValueAgent):
    """Upper Confidence Bound (Sutton and Barto, Sec. 2.7).

    Memory: Q (estimated means), N (visit counts) and t (days played).
    Choice: every spot is fished once first (random order); then
        a = argmax_a [ Q[a] + c * sqrt( ln(t) / N[a] ) ]
        where t is the number of days played so far. The bonus is large for
        rarely fished spots and shrinks as N[a] grows.
    Parameters:
        c : exploration strength, in kg. The catch noise has SD 2.5 kg, so
            values around 1-3 are sensible (default 2.0). Larger c explores more.
        step_size : None for sample averages, or a constant alpha.
    """

    def __init__(self, n_spots=5, seed=None, c=2.0, step_size=None):
        self.c = float(c)
        super().__init__(n_spots=n_spots, seed=seed, initial_value=0.0, step_size=step_size)

    def choose(self):
        untried = np.flatnonzero(self.N == 0)
        if untried.size > 0:
            return int(self.rng.choice(untried))
        bonus = self.c * np.sqrt(np.log(max(self.t, 1)) / self.N)
        return self._argmax(self.Q + bonus)


# Task 6 (optional): add your contextual agents below, as subclasses of
# ContextualAgent.


class ContextualLinUCBAgent(ContextualAgent):
    """Linear UCB for Part IV: learn how conditions affect each fishing spot.

    Each spot has its own linear model for expected catch. The feature vector is
    [1, wind_kmh/40, waves_m/3, tide_m/2], so features are on comparable scales.

    Memory
    ------
    P[a] : inverse regularized feature covariance matrix for spot a.
    b[a] : feature-weighted sum of observed catches for spot a.

    Choice
    ------
    Predict the catch for each spot and add an uncertainty bonus:
        x.T @ theta[a] + alpha * sqrt(x.T @ inv(A[a]) @ x)
    The bonus encourages trying spots where the model is still uncertain.

    Parameters
    ----------
    alpha : float
        Exploration strength. Larger values explore more; alpha=1.0 is a
        reasonable starting point for catches measured in kilograms.
    ridge : float
        Positive regularization added to the diagonal of each A matrix.
    """

    def __init__(self, n_spots=5, seed=None, alpha=1.0, ridge=1.0):
        super().__init__(n_spots=n_spots, seed=seed)
        self.alpha = float(alpha)
        self.ridge = float(ridge)
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.ridge <= 0:
            raise ValueError("ridge must be positive")
        self.reset()

    def reset(self):
        self.P = np.repeat((np.eye(4) / self.ridge)[None, :, :], self.n_spots, axis=0)
        self.b = np.zeros((self.n_spots, 4), dtype=float)

    @staticmethod
    def _features(x):
        x = np.asarray(x, dtype=float)
        return np.array([1.0, x[0] / 40.0, x[1] / 3.0, x[2] / 2.0])

    def choose(self, x):
        z = self._features(x)
        scores = np.empty(self.n_spots, dtype=float)
        for a in range(self.n_spots):
            P = self.P[a]
            theta = P @ self.b[a]
            uncertainty = np.sqrt(max(float(z @ P @ z), 0.0))
            scores[a] = z @ theta + self.alpha * uncertainty
        best = np.flatnonzero(np.isclose(scores, scores.max()))
        return int(self.rng.choice(best))

    def update(self, x, spot, catch):
        z = self._features(x)
        # Sherman-Morrison update avoids inverting a matrix on every decision.
        P = self.P[spot]
        Pz = P @ z
        denominator = 1.0 + float(z @ Pz)
        self.P[spot] = P - np.outer(Pz, Pz) / denominator
        self.b[spot] += float(catch) * z

class ContextualGreedyAgent(ContextualLinUCBAgent):
    """Prediction-only contextual policy with no uncertainty bonus.

    It fits the same regularized linear model as ContextualLinUCBAgent, then
    chooses the spot with the highest predicted catch. With alpha=0 it does not
    deliberately explore uncertain spots, making it a useful baseline for
    demonstrating why prediction quality alone can be insufficient online.
    """

    def __init__(self, n_spots=5, seed=None, ridge=1.0):
        super().__init__(n_spots=n_spots, seed=seed, alpha=0.0, ridge=ridge)


class ContextFreeAdapter(ContextualAgent):
    """Expose a Part III Agent as a contextual policy that ignores conditions.

    Parameters
    ----------
    agent_factory : callable
        Function taking a seed and returning an Agent, such as
        ``lambda seed: UCBAgent(seed=seed, c=2.0)``.
    """

    def __init__(self, agent_factory, n_spots=5, seed=None):
        super().__init__(n_spots=n_spots, seed=seed)
        self.agent = agent_factory(seed)

    def reset(self):
        self.agent.reset()

    def choose(self, x):
        return self.agent.choose()

    def update(self, x, spot, catch):
        self.agent.update(spot, catch)

