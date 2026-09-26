import numpy as np

from rlhh_cvrp.selectors import QLearningSelector, SearchState, TransitionInfo


def test_q_learning_updates_selected_action_value() -> None:
    selector = QLearningSelector(3, alpha=1.0, gamma=0.0, epsilon=0.0)
    state = SearchState(np.zeros(2), (0, 0))
    nxt = SearchState(np.ones(2), (1, 0))
    selector.update(
        state,
        2,
        0.5,
        nxt,
        False,
        TransitionInfo(accepted=True, improved_current=True, new_best=True),
    )
    assert selector.q[state.key][2] == 0.5
    assert selector.select(state, np.random.default_rng(1)) == 2


def test_frozen_q_policy_does_not_learn() -> None:
    selector = QLearningSelector(2, epsilon=0.0)
    state = SearchState(np.zeros(1), (0,))
    selector.q[state.key] = np.array([1.0, 0.0])
    frozen = selector.freeze()
    frozen.update(
        state,
        1,
        99.0,
        state,
        False,
        TransitionInfo(accepted=True, improved_current=True, new_best=True),
    )
    assert np.array_equal(frozen.q[state.key], np.array([1.0, 0.0]))
