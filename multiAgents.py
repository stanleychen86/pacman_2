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
        legMoves = gameState.getLegalActions()
        scores = [self.evaluationFunction(gameState, action) for action in legMoves]
        chosenIndex = random.choice([index for index in range(len(scores)) if scores[index] == max(scores)]) 
        return legMoves[chosenIndex]

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
        sgs = currentGameState.generatePacmanSuccessor(action)
        ngs = sgs.getGhostStates()
        newFood = sgs.getFood()
        newPos = sgs.getPacmanPosition()
        nst = [ghostState.scaredTimer for ghostState in ngs]

        "*** YOUR CODE HERE ***"
        if sgs.isWin():
          return 9999

        foodList = newFood.asList()
        foodDistance = [0]
        gps = []
        gdis = []
        cgp = []
        gdc = []
        score = 0

        for a in foodList:  
            foodDistance.append( manhattanDistance(newPos,a) )
        
        for a in ngs:
            gps.append(a.getPosition())
   
        for a in gps:
            gdis.append(manhattanDistance(newPos,a))

        for a in currentGameState.getGhostStates():
            cgp.append(a.getPosition())

        for a in cgp:
            gdc.append(manhattanDistance(newPos,a))

        score += sgs.getScore() - currentGameState.getScore()
        if action == Directions.STOP:
            score -= 12
        if newPos in currentGameState.getCapsules():
            score += 143 * len(sgs.getCapsules())
        if len(foodList) < len(currentGameState.getFood().asList()):
            score += 200

        score -= 12 * len(foodList)
        
        score = ((score - 120 if (min(gdc) > min(gdis)) else score + 200) if (sum(nst) > 0) else (score + 200 if (min(gdc) > min(gdis)) else score - 120))
        
        return score


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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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
        def minLevel(gameState, depth, agentIndex):
            minvalue = 9999
            actions = gameState.getLegalActions(agentIndex)
            if gameState.isWin() or gameState.isLose():  
                return self.evaluationFunction(gameState)
            
            for action in actions:
                if agentIndex == (gameState.getNumAgents() - 1):
                    minvalue = min (minvalue, maxLevel(gameState.generateSuccessor(agentIndex,action),depth)) 
                else:
                    minvalue = min(minvalue, minLevel(gameState.generateSuccessor(agentIndex,action),depth,agentIndex+1))  
            return minvalue
        
        def maxLevel(gameState, depth):
            cur = depth + 1
            maxvalue = -9999
            actions = gameState.getLegalActions(0)
            if cur == self.depth or gameState.isWin() or gameState.isLose(): 
                return self.evaluationFunction(gameState)
            for action in actions:
                maxvalue = max (maxvalue,minLevel(gameState.generateSuccessor(0,action),cur,1))
            return maxvalue



        currentScore = -9999
        returnAction = ''
        actions = gameState.getLegalActions(0)
        for action in actions:
            score = minLevel(gameState.generateSuccessor(0,action),0,1)
            if score >= currentScore:
                currentScore = score
                returnAction = action
        return returnAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxLevel(gameState,depth,alpha, beta):
            maxvalue = -9999
            actions = gameState.getLegalActions(0)
            temp = alpha
            currDepth = depth + 1
            if gameState.isWin() or gameState.isLose() or currDepth == self.depth:  
                return self.evaluationFunction(gameState)
            
            for action in actions:
                maxvalue = max (maxvalue,minLevel(gameState.generateSuccessor(0,action),currDepth,1,temp,beta))
                if maxvalue > beta:
                    return maxvalue
                temp = max(temp,maxvalue)
            return maxvalue

        def minLevel(gameState,depth,agentIndex,alpha,beta):
            minvalue = 9999
            actions = gameState.getLegalActions(agentIndex)
            temp = beta
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            for action in actions:
                minvalue = (min(minvalue,maxLevel(gameState.generateSuccessor(agentIndex,action),depth,alpha,temp)) if (agentIndex == (gameState.getNumAgents()-1)) else min(minvalue,minLevel(gameState.generateSuccessor(agentIndex,action),depth,agentIndex+1,alpha,temp))) 
                if minvalue < alpha:
                     return minvalue
                temp = min(temp,minvalue)
            return minvalue


        actions = gameState.getLegalActions(0)
        curr = -9999
        ans = ''
        alpha = -9999
        beta = 9999
        for action in actions:
            score = minLevel(gameState.generateSuccessor(0,action),0,1,alpha,beta)
            if score > beta:
                return ans
            if score > curr:
                ans = action
                curr = score
            alpha = max(alpha, score)
        return ans  

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
        def maxLevel(gameState,depth):
            maxvalue = -9999
            actions = gameState.getLegalActions(0)
            currDepth = depth + 1
            if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
                return self.evaluationFunction(gameState)
            
            for action in actions:
                maxvalue = max (maxvalue,expectLevel(gameState.generateSuccessor(0,action),currDepth,1))
            return maxvalue
        
        
        def expectLevel(gameState,depth, agentIndex):
            actions = gameState.getLegalActions(agentIndex)
            tev = 0
            numberofactions = len(actions)

            if gameState.isWin() or gameState.isLose():  
                return self.evaluationFunction(gameState)
            for action in actions:
                ev = (maxLevel(gameState.generateSuccessor(agentIndex,action),depth) if (agentIndex == (gameState.getNumAgents() - 1)) else expectLevel(gameState.generateSuccessor(agentIndex,action),depth,agentIndex+1))   
                tev += ev
            if numberofactions == 0:
                return  0
            else:
                return float(tev / numberofactions)


        actions = gameState.getLegalActions(0)
        cur = -9999
        ra = ''
        for action in actions:
            score = expectLevel(gameState.generateSuccessor(0,action),0,1)
            if score > cur:
                ra = action
                cur = score
        return ra

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    ngs = currentGameState.getGhostStates()
    foodList = newFood.asList()
    foodDistance = [0]
    ghostPos = []
    ghostDistance = [0]
    score = 0
    rfd = 0
    
    for a in foodList:
        foodDistance.append(manhattanDistance(newPos,a))
    
    for a in ngs:
        ghostPos.append(a.getPosition())
    
    for a in ghostPos:
        ghostDistance.append(manhattanDistance(newPos,a))

    if sum(foodDistance) > 0:
        rfd = 1.0 / sum(foodDistance)  

    score += currentGameState.getScore()  + rfd + len(newFood.asList(False))   
    score += sum([ghostState.scaredTimer for ghostState in ngs]) + (-1 * len(currentGameState.getCapsules())) + (-1 * sum (ghostDistance)) if (sum([ghostState.scaredTimer for ghostState in ngs]) > 0) else sum (ghostDistance) + len(currentGameState.getCapsules())
    
    return score
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

