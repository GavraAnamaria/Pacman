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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        dist = []
        for food in currentGameState.getFood().asList():
            dist.append(manhattanDistance(food, newPos)) #calculam distanta manhattan de la pozitia copilului la fiecare bucata de mancare

        if action == 'Stop': #oprire
            return -9999999

        for ghost in newGhostStates: #daca intalnim vreo fantoma
            if ghost.getPosition() == newPos:
                return -9999999  #returnam cel mai mic nr negativ

        return -min(dist) #returnam distanta catre cea mai apropiata bucata de mancare cu semn schimbat

        return childGameState.getScore()

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
        val_min = -999999
        for act in gameState.getLegalActions(0): #pentru fiecare actiune
            minim = self.minim(gameState.getNextState(0, act), 0, gameState.getNumAgents() - 1) #apelam functia de minim
            if minim > val_min:
                val_min, action = minim, act #salvam valoarea minima si actiunea

        return action

    def maxim(self, state, depth):

        depth += 1 #creste adancimea

        if self.depth == depth or state.isWin() or state.isLose(): #executie incheiata
            return self.evaluationFunction(state)

        maxi = -999999

        for action in state.getLegalActions(0): #pentru orice actiune legala
            children = state.getNextState(0, action)
            maxi = max(maxi, self.minim(children, depth, state.getNumAgents() - 1)) #calculam maximul dintre vechiul maxim si val returnata de functia minim
        return maxi


    def minim(self, state, depth, ghosts):

        if self.depth == depth or state.isWin() or state.isLose(): #executie incheiata
            return self.evaluationFunction(state)

        ghostNr = state.getNumAgents() - ghosts  #nr de identificare a fantomei
        mini = 9999999
        for action in state.getLegalActions(ghostNr):#pentru fiecare mutare care nu este in zid
            children = state.getNextState(ghostNr, action) #trecem in starea urmatoare
            if ghosts == 1: #daca a mai ramas o singura fantoma
                mini = min(mini, self.maxim(children, depth)) # calculam minimul dintre vechiul minim si val returnata de maxim
            if ghosts > 1: #daca numarul de fantome este mai mare decat 1
                mini = min(mini, self.minim(children, depth, ghosts - 1)) # salvam minimul dintre vechiul minim si functia apelata recursiv cu un numar mai mic de fantome
        return mini


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):

        mini = -999999

        a = -999999

        for action in gameState.getLegalActions(0):
            new_state = gameState.getNextState(0, action)
            m = self.minim(new_state, 0, gameState.getNumAgents()-1, a, 99999)
            if m > mini:
                mini, actMin = m, action

            a = max(a, mini)
        return actMin

    def maxim(self, state, depth, a, b):
        depth += 1
        maxi = -99999999
        if self.depth == depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        else:
            for action in state.getLegalActions(0):
                children = state.getNextState(0, action)
                maxi = max(maxi, self.minim(children, depth, state.getNumAgents()-1, a, b))

                if maxi > b:
                    return maxi
                a = max(a, maxi)
        return maxi

    def minim(self, state, depth, ghosts_left, a, b):

        if self.depth == depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        ghost_id = state.getNumAgents() - ghosts_left
        mini = 9999999
        for action in state.getLegalActions(ghost_id):
            children= state.getNextState(ghost_id, action)
            if ghosts_left == 1:
                mini = min(mini, self.maxim(children, depth, a, b))
            if ghosts_left > 1:
                mini = min(mini, self.minim(children, depth, ghosts_left-1, a, b))

            if mini < a:
                return mini
            b = min(b, mini)
        return mini

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
        util.raiseNotDefined()

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
