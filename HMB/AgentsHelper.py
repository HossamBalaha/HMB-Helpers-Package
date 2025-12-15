import numpy as np


class RandomAgent(object):
  r'''
  Random policy baseline that always samples from a provided sampler.
  Useful as a non-learning baseline or to perform pure exploration.
  '''

  def __init__(self, ActionSpaceSampleFunc):
    r'''
    Initialize RandomAgent.

    Parameters:
      ActionSpaceSampleFunc (callable): Callable returning a random action index when called.
    '''

    # Store the action sampler for later use.
    self.ActionSpaceSampleFunc = ActionSpaceSampleFunc

  def ChooseAction(self, state=None):
    r'''
    Return a randomly sampled action.

    Parameters:
      state: Ignored by RandomAgent but kept for API compatibility.

    Returns:
      int: Random action index returned by ActionSpaceSampleFunc().
    '''

    # Delegate to the provided sampler and return the result.
    return self.ActionSpaceSampleFunc()


class QAgent(object):
  r'''
  Base tabular Q-agent with epsilon-greedy action selection.

  This base class stores a Q-table and implements a simple epsilon-greedy
  action selection policy. Subclasses should implement UpdateParameters
  to perform learning updates (Q-learning, SARSA, Expected SARSA, ...).

  Parameters:
    ActionSpaceSampleFunc (callable): Callable returning a sampled action index from the environment's action space; used for exploration.
    alpha (float): Learning rate (0 < alpha <= 1).
    gamma (float): Discount factor for future rewards (0 <= gamma <= 1).
    noOfStates (int): Number of discrete states (size of first Q dimension).
    noOfActions (int): Number of discrete actions (size of second Q dimension).
    epsilon (float, optional): Exploration probability for epsilon-greedy policy (default 0.1).

  Attributes:
    qTable (numpy.ndarray): Array of shape (noOfStates, noOfActions) storing Q-values.
    alpha, gamma, epsilon (float): Learning and policy hyperparameters.
    ActionSpaceSampleFunc (callable): Stored reference to the sampler.
  '''

  def __init__(self, ActionSpaceSampleFunc, alpha, gamma, noOfStates, noOfActions, epsilon=0.1):
    r'''
    Initialize the QAgent and allocate the Q-table.

    Parameters:
      ActionSpaceSampleFunc (callable): Function to sample random actions.
      alpha (float): Learning rate.
      gamma (float): Discount factor.
      noOfStates (int): Number of discrete states.
      noOfActions (int): Number of discrete actions.
      epsilon (float, optional): Exploration probability for epsilon-greedy policy.
    '''

    # Store the action sampler function for exploration.
    self.ActionSpaceSampleFunc = ActionSpaceSampleFunc
    # Store the learning rate.
    self.alpha = alpha
    # Store the discount factor.
    self.gamma = gamma
    # Store the exploration probability.
    self.epsilon = epsilon
    # Store the number of available actions.
    self.noOfActions = noOfActions
    # Store the number of possible states.
    self.noOfStates = noOfStates
    # Allocate the Q-table initialized to zeros.
    self.qTable = np.zeros([noOfStates, noOfActions])

  def ChooseAction(self, state):
    r'''
    Select an action using an epsilon-greedy strategy.

    Parameters:
      state (int): Current discrete state index.

    Returns:
      int: Chosen action index. If exploring, result of ActionSpaceSampleFunc; otherwise the greedy action (argmax over Q-values).
    '''

    # Draw a uniform random number to decide exploration vs. exploitation.
    if (np.random.uniform(0, 1) < self.epsilon):
      # When exploring, sample a random action from the provided sampler.
      action = self.ActionSpaceSampleFunc()
    else:
      # When exploiting, choose the greedy action from the Q-table.
      action = self.GetAction(state)
    # Return the selected action.
    return action

  def GetAction(self, state):
    r'''
    Return the greedy action for the given state using the Q-table.

    Parameters:
      state (int): State index to query.

    Returns:
      int: Index of the action with maximal Q-value for the state. Ties are resolved by numpy.argmax (first occurrence).
    '''

    # Clip out-of-range state index to last valid.
    if ((state < 0) or (state >= self.qTable.shape[0])):
      state = min(max(state, 0), self.qTable.shape[0] - 1)
    return np.argmax(self.qTable[state, :])


class QLearningAgent(QAgent):
  r'''
  Tabular Q-learning agent implementing the standard off-policy update.

  Implements the classical Q-learning update which bootstraps using the maximum action-value in the next state.
  '''

  def UpdateParameters(self, state, nextState, reward, action, nextAction):
    r'''
    Perform a Q-learning update for a single transition.

    Parameters:
      state (int): Previous state index.
      nextState (int): Next state index after taking the action.
      reward (float): Observed reward for the transition.
      action (int): Action index taken in ``state``.
      nextAction (int or None): Present for API parity with on-policy agents but unused by Q-learning.
    '''

    # Read the old Q-value for the state-action pair.
    oldValue = self.qTable[state, action]
    # Compute the target using the reward and the max Q in the next state.
    target = reward + self.gamma * np.max(self.qTable[nextState, :])
    # Apply the learning update rule.
    newValue = oldValue + self.alpha * (target - oldValue)
    # Write the updated Q-value back into the table.
    self.qTable[state, action] = newValue


class SARSAAgent(QAgent):
  r'''
  On-policy SARSA agent that updates using the Q-value of the next taken action.

  Uses the (state, action, reward, nextState, nextAction) tuple to form the TD target for on-policy updates.
  '''

  def UpdateParameters(self, prevState, nextState, reward, prevAction, nextAction):
    r'''
    Perform a SARSA update for the observed transition.

    Parameters:
      prevState (int): The previous state index.
      nextState (int): The subsequent state index.
      reward (float): Observed reward for the transition.
      prevAction (int): Action taken in ``prevState``.
      nextAction (int): Action taken in ``nextState`` (on-policy).
    '''

    # Read the old Q-value for the previous state-action pair.
    oldValue = self.qTable[prevState, prevAction]
    # Compute the SARSA target using the Q-value of the next (state, action) pair.
    target = reward + self.gamma * self.qTable[nextState, nextAction]
    # Apply the update step.
    newValue = oldValue + self.alpha * (target - oldValue)
    # Store the updated Q-value.
    self.qTable[prevState, prevAction] = newValue


class ExpectedSARSAAgent(QAgent):
  r'''
  Expected SARSA agent that uses the expectation under the epsilon-greedy policy as the bootstrap target.

  The expected value is computed exactly by accounting for the epsilon mass that is shared between greedy and non-greedy actions and handling ties.
  '''

  def UpdateParameters(self, prevState, nextState, reward, prevAction, nextAction):
    r'''
    Perform an Expected SARSA update using the expectation over the policy in the next state.

    Parameters:
      prevState (int): Previous state index.
      nextState (int): Next state index.
      reward (float): Observed reward.
      prevAction (int): Action taken in previous state.
      nextAction (int or None): Present for API parity but unused here.
    '''

    # Read the current Q-value for the previous state and action.
    oldValue = self.qTable[prevState, prevAction]
    # Initialize accumulator for expected Q-value in nextState.
    expectedQ = 0
    # Find the maximum Q-value in the next state.
    qMax = np.max(self.qTable[nextState, :])
    # Count how many actions achieve the maximal Q-value (tie handling).
    greedyActions = 0
    for i in range(self.noOfActions):
      # Count greedy actions when their Q equals the maximum.
      if (self.qTable[nextState, i] == qMax):
        greedyActions += 1

    # Probability assigned to any non-greedy action under epsilon-greedy policy.
    nonGreedyActionProbability = self.epsilon / float(self.noOfActions)
    # Probability for each greedy action is the remaining mass divided by number of greedy actions,
    # plus the non-greedy probability that every action keeps.
    greedyActionProbability = ((1.0 - self.epsilon) / greedyActions) + nonGreedyActionProbability

    # Accumulate the expected value under the epsilon-greedy policy.
    for i in range(self.noOfActions):
      if (self.qTable[nextState, i] == qMax):
        expectedQ += self.qTable[nextState, i] * greedyActionProbability
      else:
        expectedQ += self.qTable[nextState, i] * nonGreedyActionProbability

    # Compute the expected SARSA target using expected Q of next state.
    target = reward + self.gamma * expectedQ
    # Apply the learning update.
    newValue = oldValue + self.alpha * (target - oldValue)
    # Write the updated value back to the Q-table.
    self.qTable[prevState, prevAction] = newValue


class GreedyAgent(QAgent):
  r'''
  Deterministic greedy policy that always picks the current best action.
  This convenience subclass of QAgent enforces epsilon=0 for deterministic action selection.
  '''

  def __init__(self, ActionSpaceSampleFunc, alpha, gamma, noOfStates, noOfActions):
    r'''
    Initialize GreedyAgent by calling QAgent with epsilon forced to 0.

    Parameters:
      See QAgent.__init__ for parameter descriptions; epsilon is forced to 0.
    '''

    # Initialize parent QAgent with epsilon disabled for pure exploitation.
    super().__init__(ActionSpaceSampleFunc, alpha, gamma, noOfStates, noOfActions, epsilon=0.0)

  def ChooseAction(self, state):
    r'''
    Choose the greedy action deterministically.

    Parameters:
      state (int): Current state index.

    Returns:
      int: Index of the greedy action according to the stored Q-table.
    '''

    # Use QAgent's GetAction to return the greedy choice.
    return self.GetAction(state)


class SoftmaxPolicyAgent(QAgent):
  r'''
  Softmax (Boltzmann) policy over Q-values for stochastic action selection.

  The temperature parameter controls exploration: lower values concentrate
  probability mass on higher-valued actions, higher values approach a
  uniform distribution. An optional small epsilon mixes the softmax
  probabilities with a uniform distribution.
  '''

  def __init__(self, ActionSpaceSampleFunc, alpha, gamma, noOfStates, noOfActions, temperature=1.0, epsilon=0.0):
    r'''
    Initialize the SoftmaxPolicyAgent.

    Parameters:
      temperature (float): Softmax temperature (must be > 0). Lower favors greedy actions.
      epsilon (float, optional): Small probability to mix the softmax distribution with a uniform distribution (default 0.0).
    '''

    # Initialize base QAgent, use epsilon as an optional uniform-mix parameter.
    super().__init__(ActionSpaceSampleFunc, alpha, gamma, noOfStates, noOfActions, epsilon=epsilon)
    # Store the softmax temperature.
    self.temperature = max(1e-8, float(temperature))

  def _softmax(self, logits):
    r'''
    Numerically-stable softmax that returns a probability vector.

    Parameters:
      logits (array-like): Values to convert to probabilities.

    Returns:
      numpy.ndarray: Probability vector summing to 1.
    '''

    # Shift logits by their max for numerical stability.
    shifted = logits - np.max(logits)
    # Exponentiate scaled logits.
    exp = np.exp(shifted / self.temperature)
    # Normalize to get probabilities.
    probs = exp / np.sum(exp)
    return probs

  def ChooseAction(self, state):
    r'''
    Sample an action according to the softmax policy (optionally mixed
    with epsilon uniform).

    Parameters:
      state (int): Current state index.

    Returns:
      int: Sampled action index.
    '''

    # Read Q-values for the state.
    qvals = self.qTable[state, :]
    # Compute softmax probabilities.
    probs = self._softmax(qvals)
    # If small epsilon mixing is enabled, mix with uniform distribution.
    if (self.epsilon > 0.0):
      # Create uniform probabilities.
      uniform = np.ones_like(probs) / float(len(probs))
      # Mix the distributions.
      probs = (1.0 - self.epsilon) * probs + self.epsilon * uniform
    # Sample from the probability vector using numpy choice.
    action = np.random.choice(len(probs), p=probs)
    # Return the sampled action.
    return int(action)


class DoubleQLearningAgent(QAgent):
  r'''
  Double Q-learning agent implementing two Q-tables to reduce overestimation.

  On each update a random coin flip decides which table is updated; the
  other table is used to evaluate the chosen action (the Double Q-learning
  scheme described by Van Hasselt et al.).
  '''

  def __init__(self, ActionSpaceSampleFunc, alpha, gamma, noOfStates, noOfActions, epsilon=0.1):
    r'''
    Initialize double-Q tables and parameters.

    Parameters:
      See QAgent.__init__ for parameter descriptions; an additional second Q-table is allocated internally.
    '''

    # Initialize the base single QAgent with epsilon for action selection.
    super().__init__(ActionSpaceSampleFunc, alpha, gamma, noOfStates, noOfActions, epsilon=epsilon)
    # Allocate the second Q-table initialized to zeros.
    self.qTable2 = np.zeros_like(self.qTable)

  def ChooseAction(self, state):
    r'''
    Choose action using the sum of both Q-tables for improved selection.

    Parameters:
      state (int): Current state index.

    Returns:
      int: Action index chosen either randomly (with probability epsilon) or as the argmax of qTable + qTable2.
    '''

    # Use the sum of both tables as an estimate for action values.
    combined = self.qTable[state, :] + self.qTable2[state, :]
    # With probability epsilon choose a random action.
    if (np.random.uniform(0, 1) < self.epsilon):
      # Exploration: use provided sampler.
      return self.ActionSpaceSampleFunc()
    # Exploitation: choose argmax of combined Q-values.
    return int(np.argmax(combined))

  def UpdateParameters(self, state, nextState, reward, action, nextAction=None):
    r'''
    Perform Double Q-learning update by randomly choosing which table to update.

    If the chosen table is A then the argmax is computed on A and evaluated
    using B (and vice-versa). This reduces the maximization bias present in
    standard Q-learning.

    Parameters:
      state (int): Current state index.
      nextState (int): Next state index.
      reward (float): Observed reward.
      action (int): Action taken in ``state``.
      nextAction (int or None): Present for API parity; unused here.
    '''

    # Randomly choose which table to update this step.
    if (np.random.rand() < 0.5):
      # Update primary table qTable using qTable2 for evaluation.
      old = self.qTable[state, action]
      # Select greedy action according to qTable.
      aStar = int(np.argmax(self.qTable[nextState, :]))
      # Evaluate selected action using qTable2.
      target = reward + self.gamma * self.qTable2[nextState, aStar]
      # Apply learning rule.
      self.qTable[state, action] = old + self.alpha * (target - old)
    else:
      # Update qTable2 using qTable for evaluation.
      old = self.qTable2[state, action]
      aStar = int(np.argmax(self.qTable2[nextState, :]))
      target = reward + self.gamma * self.qTable[nextState, aStar]
      self.qTable2[state, action] = old + self.alpha * (target - old)


class QLambdaAgent(QAgent):
  r'''
  Q(\u03BB) agent with accumulating eligibility traces (tabular).

  Implements off-policy Q(\u03BB) with accumulating traces. The agent
  maintains an eligibility matrix of the same shape as the Q-table and
  updates all Q-values proportionally to their eligibility on each step.
  '''

  def __init__(self, ActionSpaceSampleFunc, alpha, gamma, noOfStates, noOfActions, lambd=0.9, epsilon=0.1):
    r'''
    Initialize QLambdaAgent with eligibility traces.

    Parameters:
      lambd (float): Eligibility trace decay parameter (\u03BB), typically in [0, 1].
    '''

    # Initialize base QAgent.
    super().__init__(ActionSpaceSampleFunc, alpha, gamma, noOfStates, noOfActions, epsilon=epsilon)
    # Store lambda parameter.
    self.lambd = float(lambd)
    # Allocate eligibility traces initialized to zeros.
    self.eligibility = np.zeros_like(self.qTable)

  def ResetTraces(self):
    r'''
    Reset eligibility traces to zero.

    Call this at the start of each episode if episodic resets are used.
    '''

    # Zero the eligibility trace matrix.
    self.eligibility.fill(0.0)

  def UpdateParameters(self, state, nextState, reward, action, nextAction):
    r'''
    Perform Q(\u03BB) update with accumulating traces for a single transition.

    This implementation uses the off-policy max over next-state actions
    when forming the TD target (i.e., Q-based bootstrap).

    Parameters:
      state (int): Current state index.
      nextState (int): Next state index after taking action.
      reward (float): Observed reward.
      action (int): Action taken in ``state``.
      nextAction (int or None): Present for API parity but unused for the off-policy Q(\u03BB) update.
    '''

    # Read current Q-value for the (state, action) pair.
    qSa = self.qTable[state, action]
    # Compute TD target using max over next state's actions (off-policy Q(\u03BB)).
    tdTarget = reward + self.gamma * np.max(self.qTable[nextState, :])
    # Compute TD error.
    delta = tdTarget - qSa

    # Increment eligibility for the active state-action pair.
    self.eligibility[state, action] += 1.0

    # Update all Q-values proportionally to their eligibility.
    self.qTable += self.alpha * delta * self.eligibility

    # Decay eligibility traces by gamma * lambda.
    self.eligibility *= (self.gamma * self.lambd)


class SARSALambdaAgent(QAgent):
  r'''
  On-policy SARSA(\u03BB) agent with eligibility traces.

  This agent implements accumulating or replacing eligibility traces for
  SARSA-style on-policy learning. Call ResetTraces() at episode start.

  Parameters:
    ActionSpaceSampleFunc (callable): Action sampler used for exploration.
    alpha (float): Learning rate.
    gamma (float): Discount factor.
    noOfStates (int): Number of discrete states.
    noOfActions (int): Number of discrete actions.
    lambd (float, optional): Trace-decay parameter (default 0.9).
    epsilon (float, optional): Epsilon for epsilon-greedy policy (default 0.1).
    trace_type (str, optional): "accumulating" or "replacing" (default "accumulating").
  '''

  def __init__(
    self, ActionSpaceSampleFunc, alpha, gamma, noOfStates, noOfActions,
    lambd=0.9, epsilon=0.1, traceType="accumulating"
  ):
    r'''
    Initialize the SARSALambdaAgent.

    Parameters:
      ActionSpaceSampleFunc (callable): Function to sample random actions.
      alpha (float): Learning rate.
      gamma (float): Discount factor.
      noOfStates (int): Number of discrete states.
      noOfActions (int): Number of discrete actions.
      lambd (float, optional): Trace-decay parameter for eligibility traces.
      epsilon (float, optional): Exploration probability for epsilon-greedy policy.
      traceType (str, optional): Type of eligibility traces ("accumulating" or "replacing").
    '''

    # Initialize parent QAgent.
    super().__init__(ActionSpaceSampleFunc, alpha, gamma, noOfStates, noOfActions, epsilon=epsilon)
    self.lambd = float(lambd)
    self.traceType = traceType
    self.eligibility = np.zeros_like(self.qTable)

  def ResetTraces(self):
    r'''
    Zero the eligibility traces (call at episode start).
    '''

    # Zero the eligibility trace matrix.
    self.eligibility.fill(0.0)

  def UpdateParameters(self, prevState, nextState, reward, prevAction, nextAction):
    r'''
    Perform SARSA(\u03BB) update using the on-policy TD error.

    Parameters follow the SARSA convention; nextAction is used for the
    on-policy bootstrap.
    '''

    # TD error using Q(prevState, prevAction) and Q(nextState, nextAction)
    old = self.qTable[prevState, prevAction]
    target = reward + self.gamma * self.qTable[nextState, nextAction]
    delta = target - old

    # Update eligibility traces
    if (self.traceType == "replacing"):
      self.eligibility[prevState, prevAction] = 1.0
    else:
      self.eligibility[prevState, prevAction] += 1.0

    # Update Q-values proportionally to eligibility
    self.qTable += self.alpha * delta * self.eligibility

    # Decay traces
    self.eligibility *= (self.gamma * self.lambd)


class MonteCarloAgent(object):
  r'''
  First-visit Monte Carlo control (episodic) with incremental averaging.

  Stores an episode buffer and updates Q at episode end using returns.

  Parameters:
    ActionSpaceSampleFunc (callable): Action sampler for exploration.
    gamma (float): Discount factor.
    noOfStates (int): Number of discrete states.
    noOfActions (int): Number of discrete actions.
    epsilon (float, optional): Epsilon for epsilon-greedy policy (default 0.1).
    useFirstVisit (bool, optional): If True use first-visit MC, otherwise every-visit.
  '''

  def __init__(self, ActionSpaceSampleFunc, gamma, noOfStates, noOfActions, epsilon=0.1, useFirstVisit=True):
    r'''
    Initialize the MonteCarloAgent.

    Parameters:
      ActionSpaceSampleFunc (callable): Function to sample random actions.
      gamma (float): Discount factor.
      noOfStates (int): Number of discrete states.
      noOfActions (int): Number of discrete actions.
      epsilon (float, optional): Exploration probability for epsilon-greedy policy.
      useFirstVisit (bool, optional): If True use first-visit MC, otherwise every-visit.
    '''

    # Store the action sampler function for exploration.
    self.ActionSpaceSampleFunc = ActionSpaceSampleFunc
    # Store the discount factor.
    self.gamma = gamma
    # Store the number of available actions.
    self.noOfActions = noOfActions
    # Store the number of possible states.
    self.noOfStates = noOfStates
    # Store the exploration probability.
    self.epsilon = epsilon
    # Store the first-visit flag.
    self.useFirstVisit = useFirstVisit

    # Q-table and counts for incremental averaging
    self.qTable = np.zeros([noOfStates, noOfActions])
    self.counts = np.zeros([noOfStates, noOfActions], dtype=np.int64)

    # Episode memory
    self.episode = []  # list of (state, action, reward)

  def ChooseAction(self, state):
    r'''
    Epsilon-greedy action selection based on current Q-table.

    Parameters:
      state (int): Current discrete state index.

    Returns:
      int: Chosen action index. If exploring, result of ActionSpaceSampleFunc; otherwise the greedy action (argmax over Q-values).
    '''

    # Draw a uniform random number to decide exploration vs. exploitation.
    if (np.random.uniform() < self.epsilon):
      # When exploring, sample a random action from the provided sampler.
      action = self.ActionSpaceSampleFunc()
    else:
      # When exploiting, choose the greedy action from the Q-table.
      action = int(np.argmax(self.qTable[state, :]))
    # Return the selected action.
    return action

  def StoreTransition(self, state, action, reward):
    r'''
    Append a transition to the current episode buffer.

    Parameters:
      state (int): State index.
      action (int): Action index.
      reward (float): Reward value.
    '''

    # Append the (state, action, reward) tuple to the episode list.
    self.episode.append((state, action, reward))

  def EndEpisodeAndUpdate(self):
    r'''
    Process stored episode and perform first-visit (or every-visit)
    Monte Carlo updates, then clear the episode buffer.
    '''

    G = 0.0
    visited = set()
    # iterate backwards.
    for t in reversed(range(len(self.episode))):
      s, a, r = self.episode[t]
      G = self.gamma * G + r
      if (self.useFirstVisit):
        if ((s, a) in visited):
          continue
        visited.add((s, a))
      # incremental average.
      self.counts[s, a] += 1
      n = self.counts[s, a]
      self.qTable[s, a] += (G - self.qTable[s, a]) / float(n)
    # clear episode.
    self.episode = []


class DynaQAgent(QAgent):
  r'''
  Dyna-Q agent: model-based planning with a simple tabular model.

  Learns a one-step model (last observed next state and reward per (s,a))
  and performs planning updates by sampling previously observed pairs.

  Parameters:
    ActionSpaceSampleFunc (callable): Action sampler for exploration.
    alpha (float): Learning rate.
    gamma (float): Discount factor.
    noOfStates (int): Number of discrete states.
    noOfActions (int): Number of discrete actions.
    epsilon (float, optional): Epsilon for epsilon-greedy policy (default 0.1).
    planningSteps (int, optional): Number of model-based planning updates per real step.
  '''

  def __init__(self, ActionSpaceSampleFunc, alpha, gamma, noOfStates, noOfActions, epsilon=0.1, planningSteps=5):
    r'''
    Initialize the DynaQAgent.

    Parameters:
      ActionSpaceSampleFunc (callable): Function to sample random actions.
      alpha (float): Learning rate.
      gamma (float): Discount factor.
      noOfStates (int): Number of discrete states.
      noOfActions (int): Number of discrete actions.
      epsilon (float, optional): Exploration probability for epsilon-greedy policy.
      planningSteps (int, optional): Number of planning steps to perform.
    '''

    # Initialize the base QAgent.
    super().__init__(ActionSpaceSampleFunc, alpha, gamma, noOfStates, noOfActions, epsilon=epsilon)
    # Store the number of planning steps.
    self.planningSteps = int(planningSteps)
    # Simple model: store last seen reward and nextState for each (s,a)
    self.modelReward = np.zeros_like(self.qTable)
    self.modelNext = np.zeros_like(self.qTable, dtype=int)
    self.observed = np.zeros_like(self.qTable, dtype=bool)

  def UpdateParameters(self, state, nextState, reward, action, nextAction=None):
    r'''
    Perform a real experience Q-learning update, update the model, and
    run planningSteps simulated updates sampled from observed (s,a) pairs.

    Parameters:
      state (int): Current state index.
      nextState (int): Next state index.
      reward (float): Observed reward.
      action (int): Action taken in ``state``.
      nextAction (int or None): Present for API parity; unused here.
    '''

    # Real experience Q-learning update
    old = self.qTable[state, action]
    target = reward + self.gamma * np.max(self.qTable[nextState, :])
    self.qTable[state, action] = old + self.alpha * (target - old)

    # Update model (simple last-observed model)
    self.modelReward[state, action] = reward
    self.modelNext[state, action] = int(nextState)
    self.observed[state, action] = True

    # Planning: sample previously observed (s,a) uniformly
    seenIndices = np.argwhere(self.observed)
    if (seenIndices.size == 0):
      return
    for _ in range(self.planningSteps):
      idx = np.random.randint(len(seenIndices))
      sP, aP = seenIndices[idx]
      rP = float(self.modelReward[sP, aP])
      sPNext = int(self.modelNext[sP, aP])
      oldP = self.qTable[sP, aP]
      targetP = rP + self.gamma * np.max(self.qTable[sPNext, :])
      self.qTable[sP, aP] = oldP + self.alpha * (targetP - oldP)


class UCB1Agent(object):
  r'''
  UCB1 multi-armed bandit agent (no states).

  Uses the UCB1 formula to select actions in pure bandit problems.
  '''

  def __init__(self, ActionSpaceSampleFunc, noOfActions, c=1.0):
    r'''
    Initialize the UCB1Agent.

    Parameters:
      ActionSpaceSampleFunc (callable): Function to sample random actions.
      noOfActions (int): Number of available actions (arms).
      c (float, optional): Exploration parameter for UCB1 (default 1.0).
    '''

    # Store the action sampler function for exploration.
    self.ActionSpaceSampleFunc = ActionSpaceSampleFunc
    # Store the number of available actions.
    self.noOfActions = noOfActions
    # Store the exploration parameter.
    self.c = float(c)
    # Initialize counts and values for each action.
    self.counts = np.zeros(noOfActions, dtype=np.int64)
    self.values = np.zeros(noOfActions, dtype=float)

  def ChooseAction(self, state=None):
    r'''
    Choose an action using the UCB1 rule.

    Parameters:
      state: Ignored by UCB1Agent but kept for API compatibility.

    Returns:
      int: Action index selected by the UCB1 algorithm.
    '''

    total = np.sum(self.counts)
    # choose any untried action first
    for a in range(self.noOfActions):
      if (self.counts[a] == 0):
        return a
    # compute ucb values
    ucb = self.values + self.c * np.sqrt(np.log(total) / (self.counts + 1e-12))
    return int(np.argmax(ucb))

  def UpdateParameters(self, action, reward):
    r'''
    Update running average for the chosen arm.

    Parameters:
      action (int): Chosen action index.
      reward (float): Observed reward.
    '''

    # Increment the count for the selected action.
    self.counts[action] += 1
    n = self.counts[action]
    # Update the value estimate for the selected action using incremental average.
    self.values[action] += (reward - self.values[action]) / float(n)


class CountBonusQLAgent(QAgent):
  r'''
  Q-learning augmented with a count-based intrinsic bonus to encourage exploration.

  The bonus can be computed per-state or per state-action pair.
  '''

  def __init__(
    self, ActionSpaceSampleFunc, alpha, gamma, noOfStates, noOfActions,
    epsilon=0.1, bonus_coef=1.0, countType="state"
  ):
    r'''
    Initialize the CountBonusQLAgent.

    Parameters:
      ActionSpaceSampleFunc (callable): Function to sample random actions.
      alpha (float): Learning rate.
      gamma (float): Discount factor.
      noOfStates (int): Number of discrete states.
      noOfActions (int): Number of discrete actions.
      epsilon (float, optional): Exploration probability for epsilon-greedy policy.
      bonus_coef (float, optional): Coefficient for the exploration bonus.
      countType (str, optional): Type of count-based bonus ("state" or "state_action").
    '''

    # Initialize the base QAgent.
    super().__init__(ActionSpaceSampleFunc, alpha, gamma, noOfStates, noOfActions, epsilon=epsilon)
    # Store the bonus coefficient and count type.
    self.bonus_coef = float(bonus_coef)
    self.countType = countType
    # Initialize visitation counts.
    if (countType == "state"):
      self.counts = np.zeros(noOfStates, dtype=np.int64)
    else:
      self.counts = np.zeros([noOfStates, noOfActions], dtype=np.int64)

  def UpdateParameters(self, state, nextState, reward, action, nextAction=None):
    r'''
    Perform Q-learning update with an added exploration bonus computed
    from visitation counts.

    Parameters:
      state (int): Current state index.
      nextState (int): Next state index.
      reward (float): Observed reward.
      action (int): Action taken in ``state``.
      nextAction (int or None): Present for API parity; unused here.
    '''

    # increment counts for the nextState or (state,action)
    if (self.countType == "state"):
      self.counts[nextState] += 1
      cnt = self.counts[nextState]
    else:
      self.counts[state, action] += 1
      cnt = self.counts[state, action]

    # Compute the exploration bonus based on visitation count.
    bonus = self.bonus_coef / np.sqrt(float(cnt))
    # Augment the reward with the exploration bonus.
    augReward = reward + bonus

    # Perform the standard Q-learning update with the augmented reward.
    old = self.qTable[state, action]
    target = augReward + self.gamma * np.max(self.qTable[nextState, :])
    self.qTable[state, action] = old + self.alpha * (target - old)


class NStepTDAgent(object):
  r'''
  n-step on-policy TD agent (n-step SARSA style).

  Maintains a buffer of recent transitions and performs n-step updates.
  '''

  def __init__(self, ActionSpaceSampleFunc, alpha, gamma, noOfStates, noOfActions, n=3, epsilon=0.1):
    r'''
    Initialize the NStepTDAgent.

    Parameters:
      ActionSpaceSampleFunc (callable): Function to sample random actions.
      alpha (float): Learning rate.
      gamma (float): Discount factor.
      noOfStates (int): Number of discrete states.
      noOfActions (int): Number of discrete actions.
      n (int, optional): Number of steps for n-step TD updates (default 3).
      epsilon (float, optional): Exploration probability for epsilon-greedy policy.
    '''

    # Store the action sampler function for exploration.
    self.ActionSpaceSampleFunc = ActionSpaceSampleFunc
    # Store the learning rate.
    self.alpha = alpha
    # Store the discount factor.
    self.gamma = gamma
    # Store the number of available actions.
    self.noOfActions = noOfActions
    # Store the number of possible states.
    self.noOfStates = noOfStates
    # Store the n-step value.
    self.n = int(max(1, n))
    # Store the exploration probability.
    self.epsilon = epsilon

    # Allocate the Q-table initialized to zeros.
    self.qTable = np.zeros([noOfStates, noOfActions])
    # buffers
    self.states = []
    self.actions = []
    self.rewards = []

  def ChooseAction(self, state):
    r'''
    Select an action using an epsilon-greedy strategy.

    Parameters:
      state (int): Current discrete state index.

    Returns:
      int: Chosen action index. If exploring, result of ActionSpaceSampleFunc; otherwise the greedy action (argmax over Q-values).
    '''

    # Draw a uniform random number to decide exploration vs. exploitation.
    if (np.random.uniform() < self.epsilon):
      # When exploring, sample a random action from the provided sampler.
      action = self.ActionSpaceSampleFunc()
    else:
      # When exploiting, choose the greedy action from the Q-table.
      action = int(np.argmax(self.qTable[state, :]))
    # Return the selected action.
    return action

  def UpdateParameters(self, state, nextState, reward, action, nextAction, done=False):
    r'''
    Step the agent with a (state, action, reward) sample and perform any
    ready n-step updates. If done==True, flush remaining updates.

    Parameters:
      state (int): Current state index.
      nextState (int): Next state index.
      reward (float): Observed reward.
      action (int): Action taken in ``state``.
      nextAction (int): Action taken in ``nextState`` (on-policy).
      done (bool, optional): Flag indicating episode termination (default False).
    '''

    # Append transition.
    self.states.append(state)
    self.actions.append(action)
    self.rewards.append(reward)

    # Perform updates while we have at least n rewards or if episode ended.
    while len(self.rewards) >= self.n or (done and len(self.rewards) > 0):
      # Compute n-step return for the oldest stored transition.
      G = 0.0
      for i in range(self.n):
        if (i < len(self.rewards)):
          G += (self.gamma ** i) * self.rewards[i]
        else:
          break
      # Bootstrap term.
      if (len(self.rewards) >= self.n and not (done and len(self.rewards) == self.n and nextState is None)):
        # if we have a nextState/action to bootstrap from.
        G += (self.gamma ** self.n) * self.qTable[nextState, nextAction]

      # Pop the oldest transition from the buffers.
      s0 = self.states.pop(0)
      a0 = self.actions.pop(0)
      self.rewards.pop(0)

      # Perform the Q-value update for the oldest transition.
      old = self.qTable[s0, a0]
      self.qTable[s0, a0] = old + self.alpha * (G - old)

    # If episode ended, clear buffers.
    if (done):
      self.states = []
      self.actions = []
      self.rewards = []


class PrioritizedSweepingAgent(QAgent):
  r'''
  Prioritized Sweeping agent (model-based planning with prioritized updates).

  This agent maintains a one-step model (reward and next-state for each
  observed (s,a)) and a predecessor map for states. When a real transition
  is observed it computes a priority for the state and uses a priority
  queue to selectively perform planning updates on predecessor state-action
  pairs with largest expected TD error first. This is useful for
  sample-efficient planning in small tabular MDPs.

  Parameters:
    ActionSpaceSampleFunc (callable): Action sampler used for exploration.
    alpha (float): Learning rate for Q-value updates.
    gamma (float): Discount factor.
    noOfStates (int): Number of discrete states.
    noOfActions (int): Number of discrete actions.
    epsilon (float, optional): Epsilon for epsilon-greedy action selection (default 0.1).
    planningSteps (int, optional): Number of prioritized-planning iterations to perform per real update (default 5).
    theta (float, optional): Priority threshold; only priorities greater than theta are pushed (default 1e-4).
  '''

  def __init__(self, ActionSpaceSampleFunc, alpha, gamma, noOfStates, noOfActions, epsilon=0.1, planningSteps=5,
               theta=1e-4):
    r'''
    Initialize PrioritizedSweepingAgent and allocate model/priorities.
    '''

    super().__init__(ActionSpaceSampleFunc, alpha, gamma, noOfStates, noOfActions, epsilon=epsilon)
    import heapq
    self._heapq = heapq
    self.planningSteps = int(planningSteps)
    self.theta = float(theta)
    # Model tables.
    self.modelReward = np.zeros_like(self.qTable)
    self.modelNext = np.zeros_like(self.qTable, dtype=int)
    self.observed = np.zeros_like(self.qTable, dtype=bool)
    # Predecessors: mapping state -> set of (s_prev, a_prev).
    self.predecessors = {s: set() for s in range(self.noOfStates)}
    # Priority queue storing tuples (-priority, state) to simulate max-heap.
    self.pq = []

  def _push_state(self, state, priority):
    # Only push if priority exceeds threshold.
    if (priority > self.theta):
      # Store negative priority for max-heap behavior.
      self._heapq.heappush(self.pq, (-float(priority), int(state)))

  def _pop_state(self):
    if (not self.pq):
      return None
    pr, s = self._heapq.heappop(self.pq)
    return int(s), -float(pr)

  def ResetModel(self):
    r'''
    Clear the learned model and predecessor information.
    '''

    self.modelReward.fill(0.0)
    self.modelNext.fill(0)
    self.observed.fill(False)
    self.predecessors = {s: set() for s in range(self.noOfStates)}
    self.pq = []

  def ChooseAction(self, state):
    r'''
    Choose an action using epsilon-greedy based on current Q-table.
    '''

    return super().ChooseAction(state)

  def UpdateParameters(self, state, nextState, reward, action, nextAction=None):
    r'''
    Process a real experience (state, action, reward, nextState):
      - Update Q with a standard Q-learning step (real experience).
      - Update internal one-step model and predecessors.
      - Compute priority for the experienced state and push it if large.
      - Run up to planningSteps prioritized planning updates.

    Parameters:
      state (int): Current state index.
      nextState (int): Next state index observed.
      reward (float): Observed reward.
      action (int): Action taken in ``state``.
    '''

    # Real Q-learning update.
    old = self.qTable[state, action]
    target = reward + self.gamma * np.max(self.qTable[nextState, :])
    self.qTable[state, action] = old + self.alpha * (target - old)

    # Update model: store last seen reward/next for this (s,a).
    self.modelReward[state, action] = reward
    self.modelNext[state, action] = int(nextState)
    just_new = not self.observed[state, action]
    self.observed[state, action] = True

    # Update predecessors map: predecessor of nextState includes (state, action).
    self.predecessors[int(nextState)].add((int(state), int(action)))

    # Compute priority for the state-action pair and push state.
    priority = abs(target - old)
    self._push_state(state, priority)

    # Planning loop: pop highest-priority state and update predecessors.
    for _ in range(self.planningSteps):
      popped = self._pop_state()
      if (popped is None):
        break
      sPopped, pVal = popped
      # For each predecessor (sPrev, aPrev) of s_popped, perform update.
      for (sPrev, aPrev) in list(self.predecessors.get(sPopped, [])):
        rP = float(self.modelReward[sPrev, aPrev])
        sPNext = int(self.modelNext[sPrev, aPrev])
        oldPrev = self.qTable[sPrev, aPrev]
        targetPrev = rP + self.gamma * np.max(self.qTable[sPNext, :])
        # Q update for predecessor
        self.qTable[sPrev, aPrev] = oldPrev + self.alpha * (targetPrev - oldPrev)
        # Compute priority for predecessor and push if large.
        pr = abs(targetPrev - oldPrev)
        self._push_state(sPrev, pr)


# Self-checking test harness
if __name__ == "__main__":
  '''
  Run deterministic tests for all agents and compare to expected values.

  The tests use a tiny deterministic scenario (3 states x 2 actions)
  and perform the same sequence of calls used previously. Expected
  results are computed analytically and compared with a small numeric
  tolerance. The harness prints observed vs expected values and an
  overall PASS/FAIL summary.
  '''

  import math

  np.random.seed(0)  # Make sampling deterministic for reproducibility.

  print("Running deterministic agent tests...")

  # Environment and sampler.
  noS = 3
  noA = 2


  def sampler():
    # Deterministic pseudo-random sampler (depends on np.random seed).
    return int(np.random.randint(0, noA))


  results = {}

  # 1) SarsaLambdaAgent: single SARSA(lambda) update.
  sarsaL = SARSALambdaAgent(sampler, 0.5, 0.9, noS, noA, lambd=0.8, epsilon=0.2)
  results["SARSALambda_Q_sum"] = float(np.sum(sarsaL.qTable))
  sarsaL.UpdateParameters(prevState=0, nextState=1, reward=1.0, prevAction=0, nextAction=1)
  results["SARSALambda_Q_sum"] = float(np.sum(sarsaL.qTable))

  # 2) MonteCarloAgent: two-step episode (0,0,1.0), (1,1,2.0).
  mc = MonteCarloAgent(sampler, 0.9, noS, noA, epsilon=0.2, useFirstVisit=True)
  results["MonteCarlo_Q_00"] = float(mc.qTable[0, 0])
  mc.StoreTransition(0, 0, 1.0)
  mc.StoreTransition(1, 1, 2.0)
  mc.EndEpisodeAndUpdate()
  results["MonteCarlo_Q_00"] = float(mc.qTable[0, 0])

  # 3) DynaQAgent: one real update then 3 planning steps.
  dyna = DynaQAgent(sampler, 0.5, 0.9, noS, noA, epsilon=0.2, planningSteps=3)
  results["DynaQ_Q_sum"] = float(np.sum(dyna.qTable))
  dyna.UpdateParameters(state=0, nextState=1, reward=1.0, action=0)
  results["DynaQ_Q_sum"] = float(np.sum(dyna.qTable))

  # 4) UCB1Agent: choose untried arm then update its count.
  ucb = UCB1Agent(sampler, noA, c=1.0)
  results["UCB1_counts"] = np.array(ucb.counts, copy=True)
  chosen = ucb.ChooseAction()
  ucb.UpdateParameters(chosen, 1.0)
  results["UCB1_counts"] = np.array(ucb.counts, copy=True)
  cb = CountBonusQLAgent(sampler, 0.5, 0.9, noS, noA, epsilon=0.1, bonus_coef=1.0, countType="state")
  # 5) CountBonusQLAgent: state-count bonus update.
  results["CountBonusQL_Q_sum"] = float(np.sum(cb.qTable))
  cb.UpdateParameters(state=0, nextState=1, reward=0.0, action=1)
  results["CountBonusQL_Q_sum"] = float(np.sum(cb.qTable))

  # 6) NStepTDAgent: two-step episode flushed at end.
  ns = NStepTDAgent(sampler, 0.5, 0.9, noS, noA, n=2, epsilon=0.1)
  ns.UpdateParameters(state=0, nextState=1, reward=1.0, action=0, nextAction=1, done=False)
  ns.UpdateParameters(state=1, nextState=2, reward=2.0, action=1, nextAction=0, done=True)
  results["NStep_Q_sum"] = float(np.sum(ns.qTable))

  # 7) PrioritizedSweepingAgent complex test: two sequential updates to build predecessors
  # First update: (1,0) -> nextState=2 with reward=1.0
  # Second update: (0,1) -> nextState=1 with reward=2.0
  # We expect real updates only (no additional predecessor-triggered changes) for this small scenario
  psa = PrioritizedSweepingAgent(sampler, 0.5, 0.9, noS, noA, epsilon=0.1, planningSteps=2, theta=0.0)
  psa.UpdateParameters(state=1, nextState=2, reward=1.0, action=0)
  psa.UpdateParameters(state=0, nextState=1, reward=2.0, action=1)
  # Analytical expectations:
  # Q[1,0] = 0.5 (first real update with alpha=0.5)
  # Q[0,1] = 0.5 * (2.0 + 0.9 * 0.5) = 0.5 * 2.45 = 1.225
  # Sum = 0.5 + 1.225 = 1.725
  results["PrioritizedSweeping_Q_sum"] = float(np.sum(psa.qTable))

  # Analytically computed expected values for the above deterministic sequence.
  expected = {
    "SARSALambda_Q_sum"        : 0.5,  # 0.5 after single SARSA(lambda) accumulating update.
    "MonteCarlo_Q_00"          : 2.8,  # G for state 0: 1 + 0.9*2 = 2.8.
    "DynaQ_Q_sum"              : 0.9375,  # 0.5 -> 0.75 -> 0.875 -> 0.9375 after 3 planning steps.
    "UCB1_counts"              : np.array([1, 0]),  # first untried arm chosen then incremented.
    "CountBonusQL_Q_sum"       : 0.5,  # Bonus=1->aug_reward=1->Q update yields 0.5.
    "NStep_Q_sum"              : 2.4,  # 2-step returns lead to Q[0,0]=1.4 and Q[1,1]=1.0 -> sum=2.4.
    "PrioritizedSweeping_Q_sum": 1.725  # See analytical expectations above.
  }

  print("\nTest results:")
  tol = 1e-8
  allPass = True

  for k in expected.keys():
    expVal = expected[k]
    obsVal = results[k]
    if (isinstance(expVal, np.ndarray)):
      testPass = np.allclose(obsVal, expVal, atol=tol)
    else:
      testPass = math.isclose(obsVal, expVal, abs_tol=tol)
    allPass = allPass and testPass
    status = "PASS" if testPass else "FAIL"
    print(f"  {k:25s}: observed={obsVal} expected={expVal} [{status}]")

  print("\nOverall test result: " + ("PASS" if allPass else "FAIL"))

  # Another advanced test.
  print("\nRunning advanced test...")

  # Advanced deterministic scenario: two sequential Q-updates with gamma=0 simplify bootstrap.
  np.random.seed(123)  # Reseed for reproducibility in advanced test.

  advResults = {}

  # QLearningAgent: two updates on (state=0, action=0) with gamma=0, alpha=0.5.
  ql = QLearningAgent(sampler, 0.5, 0.0, noS, noA, epsilon=0.0)
  ql.UpdateParameters(state=0, nextState=0, reward=1.0, action=0, nextAction=None)
  ql.UpdateParameters(state=0, nextState=0, reward=3.0, action=0, nextAction=None)
  advResults["QLearning_Q00"] = float(ql.qTable[0, 0])

  # DynaQAgent with planningSteps=0 should behave like Q-learning here.
  dyna0 = DynaQAgent(sampler, 0.5, 0.0, noS, noA, epsilon=0.0, planningSteps=0)
  dyna0.UpdateParameters(state=0, nextState=0, reward=1.0, action=0)
  dyna0.UpdateParameters(state=0, nextState=0, reward=3.0, action=0)
  advResults["DynaQ_Q00"] = float(dyna0.qTable[0, 0])

  # PrioritizedSweepingAgent with planningSteps=0 should also reduce to a Q-like update.
  psa0 = PrioritizedSweepingAgent(sampler, 0.5, 0.0, noS, noA, epsilon=0.0, planningSteps=0, theta=0.0)
  psa0.UpdateParameters(state=0, nextState=0, reward=1.0, action=0)
  psa0.UpdateParameters(state=0, nextState=0, reward=3.0, action=0)
  advResults["PSA_Q00"] = float(psa0.qTable[0, 0])

  # Analytical expected Q after two sequential updates with alpha=0.5, gamma=0:
  # Q1 = 0 + 0.5*(1 - 0) = 0.5
  # Q2 = 0.5 + 0.5*(3 - 0.5) = 1.75
  advExpected = {"QLearning_Q00": 1.75, "DynaQ_Q00": 1.75, "PSA_Q00": 1.75}

  print("\nAdvanced test results:")
  advAllPass = True
  for k, obs in advResults.items():
    exp = advExpected[k]
    ok = bool(np.isclose(obs, exp, rtol=1e-6, atol=1e-8))
    advAllPass = advAllPass and ok
    print(f"  {k:15s}: observed={obs:.6f} expected={exp:.6f} [{'PASS' if ok else 'FAIL'}]")

  print("\nAdvanced overall: " + ("PASS" if advAllPass else "FAIL"))

  # End of advanced test.
  print("\nAll deterministic tests completed.")
