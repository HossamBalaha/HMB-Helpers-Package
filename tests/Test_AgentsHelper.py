import unittest
import numpy as np
from HMB.AgentsHelper import (
  RandomAgent,
  QLearningAgent,
  SARSAAgent,
  ExpectedSARSAAgent,
  GreedyAgent,
  SoftmaxPolicyAgent,
  DoubleQLearningAgent,
)


def make_env(n_states=5, n_actions=3):
  trans = np.zeros((n_states, n_actions), dtype=int)
  rewards = np.random.randn(n_states, n_actions)
  for s in range(n_states):
    for a in range(n_actions):
      trans[s, a] = (s + a + 1) % n_states
  return trans, rewards


class TestAgentsHelper(unittest.TestCase):
  '''
  Unit tests for core tabular agents using a tiny environment.
  '''

  def setUp(self):
    self.nStates = 6
    self.nActions = 4
    self.trans, self.rewards = make_env(self.nStates, self.nActions)
    self.sampler = lambda: np.random.randint(0, self.nActions)
    self.alpha = 0.5
    self.gamma = 0.9
    self.epsilon = 0.1

  def _one_step(self, agent, state):
    action = agent.ChooseAction(state)
    self.assertTrue(0 <= action < self.nActions)
    nextState = self.trans[state, action]
    reward = self.rewards[state, action]
    # Use appropriate UpdateParameters signature
    if isinstance(agent, QLearningAgent) or isinstance(agent, DoubleQLearningAgent):
      agent.UpdateParameters(state, nextState, reward, action, None)
    elif isinstance(agent, SARSAAgent) or isinstance(agent, ExpectedSARSAAgent):
      # derive next action via policy
      nextAction = agent.ChooseAction(nextState)
      self.assertTrue(0 <= nextAction < self.nActions)
      agent.UpdateParameters(state, nextState, reward, action, nextAction)
    return nextState

  def test_random_agent(self):
    ra = RandomAgent(self.sampler)
    a = ra.ChooseAction(state=None)
    self.assertTrue(0 <= a < self.nActions)

  def test_greedy_and_softmax_policy_agents(self):
    greedy = GreedyAgent(self.sampler, self.alpha, self.gamma, self.nStates, self.nActions)
    s = 0
    for _ in range(5):
      s = self._one_step(greedy, s)
    softmax = SoftmaxPolicyAgent(self.sampler, self.alpha, self.gamma, self.nStates, self.nActions,
                                 epsilon=self.epsilon)
    s = 0
    for _ in range(5):
      s = self._one_step(softmax, s)

  def test_q_learning_variants(self):
    ql = QLearningAgent(self.sampler, self.alpha, self.gamma, self.nStates, self.nActions, epsilon=self.epsilon)
    s = 0
    for _ in range(10):
      s = self._one_step(ql, s)
    dql = DoubleQLearningAgent(self.sampler, self.alpha, self.gamma, self.nStates, self.nActions, epsilon=self.epsilon)
    s = 0
    for _ in range(10):
      s = self._one_step(dql, s)

  def test_sarsa_variants(self):
    sarsa = SARSAAgent(self.sampler, self.alpha, self.gamma, self.nStates, self.nActions, epsilon=self.epsilon)
    s = 0
    for _ in range(10):
      s = self._one_step(sarsa, s)
    expSarsa = ExpectedSARSAAgent(self.sampler, self.alpha, self.gamma, self.nStates, self.nActions,
                                  epsilon=self.epsilon)
    s = 0
    for _ in range(10):
      s = self._one_step(expSarsa, s)

  def test_epsilon_extremes_and_deterministic_sampler(self):
    # Deterministic sampler always returns 0
    det_sampler = lambda: 0
    ql_zero_eps = QLearningAgent(det_sampler, self.alpha, self.gamma, self.nStates, self.nActions, epsilon=0.0)
    a = ql_zero_eps.ChooseAction(0)
    self.assertTrue(0 <= a < self.nActions)
    ql_full_eps = QLearningAgent(det_sampler, self.alpha, self.gamma, self.nStates, self.nActions, epsilon=1.0)
    a2 = ql_full_eps.ChooseAction(0)
    self.assertTrue(0 <= a2 < self.nActions)

  def test_invalid_params_raise_or_clip(self):
    # Negative alpha/gamma should either raise or be clipped; ensure no crash on init
    _ = QLearningAgent(self.sampler, -0.1, -0.5, self.nStates, self.nActions, epsilon=self.epsilon)
    # Out-of-range state/action should not crash ChooseAction
    agent = GreedyAgent(self.sampler, self.alpha, self.gamma, self.nStates, self.nActions)
    a = agent.ChooseAction(state=self.nStates + 10)
    self.assertTrue(0 <= a < self.nActions)


if __name__ == "__main__":
  unittest.main()
