import numpy as np


def MantaRayForagingOptimizer(
  X,  # Current population of candidate solutions.
  Fs,  # Fitness values for each candidate in X.
  Ps,  # Population size.
  D,  # Number of dimensions.
  lb,  # Lower bounds for each dimension.
  ub,  # Upper bounds for each dimension.
  t,  # Current iteration number.
  T,  # Total number of iterations.
  fitnessFunction=None,  # Function to evaluate fitness of a candidate.
  spiral=True,  # Whether to use spiral foraging phase.
  spiralConstant=2.0,  # Spiral constant for spiral foraging phase.
  explorationProb=0.5,  # Probability of exploration phase.
  customUpdate=None,  # Optional custom update function for candidates.
):
  r'''
  Perform one iteration of the Manta Ray Foraging Optimization (MRFO) algorithm with dynamic options.

  Parameters:
    X (numpy.ndarray): Current population of candidate solutions (shape: [Ps, D]).
    Fs (list or numpy.ndarray): Fitness values for each candidate in X.
    Ps (int): Population size.
    D (int): Number of dimensions.
    lb (numpy.ndarray): Lower bounds for each dimension.
    ub (numpy.ndarray): Upper bounds for each dimension.
    t (int): Current iteration number.
    T (int): Total number of iterations.
    fitnessFunction (callable): Function to evaluate fitness of a candidate.
    spiral (bool, optional): Whether to use spiral foraging phase. Default is True.
    spiralConstant (float, optional): Spiral constant for spiral foraging phase. Default is 2.0.
    explorationProb (float, optional): Probability of exploration phase. Default is 0.5.
    customUpdate (callable, optional): Custom update function with signature (i, X, newX, bestSolution, r, alpha, beta, coef, lb, ub, D, t, T) -> np.ndarray.

  Returns:
    tuple: (newX, bestSolution, bestFitness)
      - newX (numpy.ndarray): Updated population after this iteration.
      - bestSolution (numpy.ndarray): Best solution found so far.
      - bestFitness (float): Fitness value of the best solution.
  '''

  # Set initial best solution, fitness, and index.
  bestSolution, bestFitness, bestIndex = X[0], Fs[0], 0
  # Copy current population.
  newX = np.copy(X)
  # Calculate coefficient for current iteration.
  coef = t / float(T)
  # Iterate over each candidate in the population.
  for i in range(0, Ps):
    # Generate random number for exploration/exploitation.
    r = np.random.random(1)
    # Calculate alpha parameter.
    alpha = 2.0 * r * np.sqrt(np.abs(np.log(r)))
    # Generate another random number for beta calculation.
    r1 = np.random.random(1)
    # Calculate factor for beta.
    factor = (T - t + 1.0) / (T * 1.0)
    # Calculate beta parameter.
    beta = 2.0 * np.exp(r1 * factor) * np.sin(2.0 * np.pi * r1)
    # Use custom update function if provided.
    if (customUpdate is not None):
      # Call the custom update function for candidate i.
      newX[i, :] = customUpdate(
        i, X, newX, bestSolution, r, alpha, beta, coef, lb, ub, D, t, T
      )
    else:
      # Decide update strategy based on exploration probability.
      if (np.random.random(1) < explorationProb):
        # Exploration phase.
        if (coef < np.random.random(1)):
          # Generate random solution within bounds.
          low, high = 0, 1
          s = np.subtract(ub, lb)
          u = np.random.uniform(low=low, high=high, size=D)
          m = np.multiply(u, s)
          xRand = np.clip(np.add(lb, m), lb, ub)
          # Update candidate based on random solution.
          if (i == 0):
            newX[i, :] = xRand + r * (xRand - X[i, :]) + beta * (xRand - X[i, :])
          else:
            newX[i, :] = xRand + r * (X[i - 1, :] - X[i, :]) + beta * (xRand - X[i, :])
        else:
          # Update candidate based on best solution.
          if (i == 0):
            newX[i, :] = bestSolution + r * (bestSolution - X[i, :]) + beta * (bestSolution - X[i, :])
          else:
            newX[i, :] = bestSolution + r * (X[i - 1, :] - X[i, :]) + beta * (bestSolution - X[i, :])
      else:
        # Exploitation phase.
        if (i == 0):
          newX[i, :] = X[i, :] + r * (bestSolution - X[i, :]) + alpha * (bestSolution - X[i, :])
        else:
          newX[i, :] = X[i, :] + r * (X[i - 1, :] - X[i, :]) + alpha * (bestSolution - X[i, :])
    # Ensure candidate is within bounds.
    newX[i, :] = np.clip(newX[i, :], lb, ub)
    # Evaluate fitness of updated candidate.
    currentScore = fitnessFunction(newX[i, :])
    # Update the best solution if improved.
    if (currentScore < bestFitness):
      bestSolution, bestFitness = newX[i, :].copy(), currentScore
    # Spiral foraging phase if enabled.
    if (spiral):
      r2, r3 = np.random.random(1), np.random.random(1)
      newX[i, :] = X[i, :] + spiralConstant * (r2 * bestSolution - r3 * X[i, :])
      # Ensure candidate is within bounds after spiral update.
      newX[i, :] = np.clip(newX[i, :], lb, ub)
      # Evaluate fitness after spiral update.
      currentScore = fitnessFunction(newX[i, :])
      # Update the best solution if improved.
      if (currentScore < bestFitness):
        bestSolution, bestFitness = newX[i, :].copy(), currentScore
  # Ensure all candidates are within bounds.
  newX = np.clip(newX, lb, ub)
  # Return updated population, the best solution, and best fitness.
  return newX, bestSolution, bestFitness


if __name__ == "__main__":
  import tqdm


  # Define a simple fitness function (sum of elements).
  def FitnessFunction(X):
    # Calculate sum of elements as fitness.
    result = np.sum(X)
    return result


  # Set population size.
  Ps = 100
  # Set number of dimensions.
  D = 10
  # Set lower bounds.
  LB = np.ones(D) * 0
  # Set upper bounds.
  UB = np.ones(D) * 20
  # Initialize population randomly within bounds.
  X = LB + (UB - LB) * np.random.random((Ps, D))
  X = np.clip(X, LB, UB)
  # Set number of iterations.
  T = 100
  # Initialize the best fitness.
  bestFitness = np.inf
  # Initialize the best solution.
  bestSolution = np.zeros(D)

  # Run optimization for T iterations.
  for t in tqdm.tqdm(range(1, T + 1)):
    # Evaluate fitness for all candidates.
    Fs = [FitnessFunction(fs) for fs in X]
    # Sort population by fitness.
    X, Fs = zip(*sorted(zip(X, Fs), key=lambda x: x[1]))
    X = np.array(X)
    # Run one iteration of MRFO.
    X, bestSolution, bestFitness = MantaRayForagingOptimizer(
      X, Fs, Ps, D, LB, UB, t, T, FitnessFunction
    )
  # Print final population shape and best fitness.
  print("The shape of X is:", X.shape)
  print("The best fitness is:", bestFitness)
  print("The best solution is:", bestSolution)
