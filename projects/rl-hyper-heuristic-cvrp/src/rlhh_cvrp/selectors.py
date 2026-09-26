from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class SearchState:
    features: np.ndarray
    key: tuple[int, ...]


@dataclass(frozen=True)
class TransitionInfo:
    accepted: bool
    improved_current: bool
    new_best: bool


class Selector(Protocol):
    n_actions: int

    def reset_episode(self) -> None: ...

    def select(self, state: SearchState, rng: np.random.Generator) -> int: ...

    def update(
        self,
        state: SearchState,
        action: int,
        reward: float,
        next_state: SearchState,
        done: bool,
        info: TransitionInfo,
    ) -> None: ...

    def end_episode(self) -> None: ...


class BaseSelector:
    def __init__(self, n_actions: int) -> None:
        self.n_actions = n_actions

    def reset_episode(self) -> None:
        return None

    def update(
        self,
        state: SearchState,
        action: int,
        reward: float,
        next_state: SearchState,
        done: bool,
        info: TransitionInfo,
    ) -> None:
        return None

    def end_episode(self) -> None:
        return None


class FixedSelector(BaseSelector):
    def __init__(self, n_actions: int, action: int) -> None:
        super().__init__(n_actions)
        self.action = action

    def select(self, state: SearchState, rng: np.random.Generator) -> int:
        return self.action


class RandomSelector(BaseSelector):
    def select(self, state: SearchState, rng: np.random.Generator) -> int:
        return int(rng.integers(0, self.n_actions))


class RoundRobinSelector(BaseSelector):
    def __init__(self, n_actions: int) -> None:
        super().__init__(n_actions)
        self.index = 0

    def reset_episode(self) -> None:
        self.index = 0

    def select(self, state: SearchState, rng: np.random.Generator) -> int:
        action = self.index % self.n_actions
        self.index += 1
        return action


class AdaptiveWeightSelector(BaseSelector):
    def __init__(self, n_actions: int, reaction: float = 0.2) -> None:
        super().__init__(n_actions)
        if not 0.0 < reaction <= 1.0:
            raise ValueError("reaction must be in (0, 1]")
        self.reaction = reaction
        self.weights = np.ones(n_actions, dtype=float)

    def reset_episode(self) -> None:
        self.weights = np.ones(self.n_actions, dtype=float)

    def select(self, state: SearchState, rng: np.random.Generator) -> int:
        probabilities = self.weights / self.weights.sum()
        return int(rng.choice(self.n_actions, p=probabilities))

    def update(
        self,
        state: SearchState,
        action: int,
        reward: float,
        next_state: SearchState,
        done: bool,
        info: TransitionInfo,
    ) -> None:
        score = 0.0
        if info.new_best:
            score = 5.0
        elif info.improved_current:
            score = 3.0
        elif info.accepted:
            score = 1.0
        target = max(0.1, score)
        self.weights[action] = (
            (1.0 - self.reaction) * self.weights[action] + self.reaction * target
        )


class QLearningSelector(BaseSelector):
    def __init__(
        self,
        n_actions: int,
        *,
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon: float = 0.3,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.97,
    ) -> None:
        super().__init__(n_actions)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q: dict[tuple[int, ...], np.ndarray] = {}
        self.learning = True

    def _values(self, key: tuple[int, ...]) -> np.ndarray:
        if key not in self.q:
            self.q[key] = np.zeros(self.n_actions, dtype=float)
        return self.q[key]

    def select(self, state: SearchState, rng: np.random.Generator) -> int:
        values = self._values(state.key)
        if self.learning and rng.random() < self.epsilon:
            return int(rng.integers(0, self.n_actions))
        best = np.flatnonzero(values == values.max())
        return int(rng.choice(best))

    def update(
        self,
        state: SearchState,
        action: int,
        reward: float,
        next_state: SearchState,
        done: bool,
        info: TransitionInfo,
    ) -> None:
        if not self.learning:
            return
        current = self._values(state.key)
        bootstrap = 0.0 if done else float(self._values(next_state.key).max())
        target = reward + self.gamma * bootstrap
        current[action] += self.alpha * (target - current[action])

    def end_episode(self) -> None:
        if self.learning:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def freeze(self) -> "QLearningSelector":
        frozen = QLearningSelector(
            self.n_actions,
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon=0.0,
            epsilon_min=0.0,
            epsilon_decay=1.0,
        )
        frozen.q = {key: value.copy() for key, value in self.q.items()}
        frozen.learning = False
        return frozen
