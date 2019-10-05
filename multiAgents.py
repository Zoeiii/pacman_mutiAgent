# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        # number of moves before ghost is not being scared
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # using the current states and the action to evaluate this action
        food = currentGameState.getFood().asList()
        # list of successor states
        currentPos = list(successorGameState.getPacmanPosition())
        distance = float("-Inf")  # initialize it to -Inf, not a good move

        # not moving is a bad move
        if action == 'Stop':
            return float("-Inf")

        for state in newGhostStates:
            # if agent are too close to ghost, and the ghost is not scared, bad move
            if state.getPosition() == tuple(currentPos) and (state.scaredTimer == 0):
                return float("-Inf")

        for x in food:
            # find the best move using manhattan distance between agent and ghost
            distance = max(distance, (-1 * (manhattanDistance(currentPos, x))))

        # the larger the number we returned the better the move/action is
        return distance


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        # taking the agentIndex, depth and gameState
        def minimax(agentIndex, depth, gameState):
            # if state is terminal state, return state's utility
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)
            # if next agent is MAX/pacmam, return max-value
            if agentIndex == 0:
                v = float("-Inf")
                for newState in gameState.getLegalActions(agentIndex):
                    # calculate the max value for next agent
                    v = max(v, (minimax(1, depth, gameState.generateSuccessor(agentIndex, newState))))
                return v

            # minimize ghosts
            else:
                nextAgent = agentIndex + 1
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                    depth += 1
                v = float("+Inf")
                for newState in gameState.getLegalActions(agentIndex):
                    v = min(v, (minimax(nextAgent, depth, gameState.generateSuccessor(agentIndex, newState))))
                return v

        # the root of the tree
        maximum = float("-inf")
        action = Directions.WEST
        # for every legalAction, find the best move
        for agentState in gameState.getLegalActions(0):
            # root of the tree
            utility = minimax(1, 0, gameState.generateSuccessor(0, agentState))
            # compare each move's utility and mark down the best move as action
            if utility > maximum or maximum == float("-inf"):
                maximum = utility
                action = agentState
        # return the best action
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # question, why i cannot write >= and <= instead of > and <, equality
        # maximizer function
        def maxVal(agentIndex, depth, gameState, alpha, beta):
            v = float("-inf")
            #for each succsor of the state
            for newAction in gameState.getLegalActions(agentIndex):
                v = max(v, value(1, depth, gameState.generateSuccessor(agentIndex, newAction), alpha, beta))
                if v > beta: # if val is greater than beta, we can prune
                    return v
                alpha = max(alpha, v)
            return v

        def minVal(agentIndex, depth, gameState, alpha, beta):
            v = float("inf")
            nextAgent = agentIndex + 1
            # when it calculates the value of all the agents of the game, we can start a new action/ increase depth
            if nextAgent == gameState.getNumAgents():
                nextAgent = 0
                depth += 1

            for newAction in gameState.getLegalActions(agentIndex):
                v = min(v, value(nextAgent, depth, gameState.generateSuccessor(agentIndex, newAction), alpha,beta))
                if v < alpha: # if val is less than alpha, we can prune
                    return v
                beta = min(beta, v)

            return v
        # alpha beta puring function with alpha and beta value init to -inf and +inf
        def value(agentIndex, depth, gameState, alpha, beta):
            if gameState.isWin() or gameState.isLose() or self.depth == depth:
                return self.evaluationFunction(gameState)
            if agentIndex == 0:  # the pacman
                return maxVal(agentIndex, depth, gameState, alpha, beta)
            else:
                return minVal(agentIndex, depth, gameState, alpha, beta)

        # calling the AlphaBeta value function from the root of the tree
        alpha = float("-inf")
        beta = float("inf")
        utility = float("-inf")
        action = Directions.EAST
        for newAction in gameState.getLegalActions(0):
            ghostVal = value(1, 0, gameState.generateSuccessor(0, newAction), alpha, beta)
            if ghostVal > utility:
                utility = ghostVal
                action = newAction
            if utility > beta:
                return utility
            alpha = max(alpha, utility)

        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        # maximizer function
        def maxVal(agentIndex, depth, gameState):
            v = float("-inf")
            # for each succsor of the state
            for newAction in gameState.getLegalActions(agentIndex):
                v = max(v, expectimax(1, depth, gameState.generateSuccessor(agentIndex, newAction)))
            return v

        def expVal(agentIndex, depth, gameState):
            v = 0
            nextAgent = agentIndex + 1
            # when it calculates the value of all the agents of the game, we can start a new action/ increase depth
            if nextAgent == gameState.getNumAgents():
                nextAgent = 0
                depth += 1
            # return the avg of all the ghost value
            for newAction in gameState.getLegalActions(agentIndex):
                v += expectimax(nextAgent, depth, gameState.generateSuccessor(agentIndex, newAction))
            return v / float(len(gameState.getLegalActions(agentIndex)))

        # alpha beta puring function with alpha and beta value init to -inf and +inf
        def expectimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or self.depth == depth:
                return self.evaluationFunction(gameState)
            if agentIndex == 0:  # the pacman
                return maxVal(agentIndex, depth, gameState)
            else:
                return expVal(agentIndex, depth, gameState)

        # calling the root node
        maximum = float("-inf")
        action = Directions.WEST
        for agentState in gameState.getLegalActions(0):
            utility = expectimax(1, 0, gameState.generateSuccessor(0, agentState))
            if utility > maximum or maximum == float("-inf"):
                maximum = utility
                action = agentState

        return action


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
