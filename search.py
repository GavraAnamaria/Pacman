# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def expand(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (child,
        action, stepCost), where 'child' is a child to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that child.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
          state: Search state

        For a given state, this should return a list of possible actions.
        """
        util.raiseNotDefined()

    def getActionCost(self, state, action, next_state):
        """
          state: Search state
          action: action taken at state.
          next_state: next Search state after taking action.

        For a given state, this should return the cost of the (s, a, s') transition.
        """
        util.raiseNotDefined()

    def getNextState(self, state, action):
        """
          state: Search state
          action: action taken at state

        For a given state, this should return the next state after taking action from state.
        """
        util.raiseNotDefined()

    def getCostOfActionSequence(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):

    expanded = []
    stack = util.Stack()
    stack.push((problem.getStartState(), [])) #adaugam pe lista prima stare

    while not stack.isEmpty(): #cattimp nu am explorattoate starile
        (parent, path) = stack.pop() #scoatem ultima stare adaugata din stiva
        if problem.isGoalState(parent): #verificam daca e scop
            return path
        expanded = expanded + [parent] #il adaugam la lista nodurilor expandate

        for children, action, cost in problem.expand(parent): #ii parcurgem succesorii
            if children not in expanded: #daca nu au fost expandati
                stack.push((children, path + [action])) #ii punem pe stiva

def breadthFirstSearch(problem):

    queue = util.Queue()
    visited = []

    queue.push((problem.getStartState(), [])) #adaugam in coada primul nod
    visited = visited + [problem.getStartState()] #il vizitam

    while not queue.isEmpty():
        (parent, path) = queue.pop() #extragem primul nod adaugat din lista
        if problem.isGoalState(parent): #verificam daca e scop
            return path #in caz afirmativ, returnam drumul spre acel nod

        for children, action, cost in problem.expand(parent): #expandam nodul
            if children not in visited:
                queue.push((children, path + [action])) #adaugam copiii care nu au fost vizitati in coada
            visited = visited + [children] #ii marcam ca vizitati




def UCS(problem):
    queue = util.PriorityQueue()
    visited = []

    queue.push((problem.getStartState(), []), 0) #adaugam in coada de prioritati prima stare si costul=0
    visited = visited + [problem.getStartState()] #il marcam ca vizitat

    while not queue.isEmpty():
        parent, path = queue.pop() #extragem din coada nodul care are costul minim
        visited = visited + [parent] #marcam nodul ca vizitat
        if problem.isGoalState(parent):#daca este scop
            return path #returnam calea

        for children, action, cost in problem.expand(parent): #expandam nodul
            if children not in visited:
                queue.push((children, path + [action]), problem.getCostOfActionSequence(path + [action])) #adaugam in coada copiii care nu au fost vizitati
            visited = visited + [children] #ii marcam ca vizitati
            if problem.isGoalState(children):  #verificam daca copilul este scop
                queue.push((children, path + [action]), problem.getCostOfActionSequence(path + [action]))

    return None
    # util.raiseNotDefined()




def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):

    queue = util.PriorityQueue()
    visited = []

    queue.push((problem.getStartState(), []), heuristic(problem.getStartState(),problem)) #adaugam in coada de prioritati prima stare si valoarea euristicii(val cost=0)
    visited = visited + [problem.getStartState()] #il marcam ca vizitat

    while not queue.isEmpty():
        parent, path = queue.pop() #extragem din coada nodul care are suma dintre cost si euristica minima
        visited = visited + [parent] #marcam nodul ca vizitat
        if problem.isGoalState(parent):#daca este scop
            return path #returnam calea

        for children, action, cost in problem.expand(parent): #expandam nodul
            if children not in visited:
                queue.push((children, path + [action]), problem.getCostOfActionSequence(path + [action]) + heuristic(children,problem)) #adaugam in coada copiii care nu au fost vizitati
            visited = visited + [children] #ii marcam ca vizitati
            if problem.isGoalState(children):  #verificam daca copilul este scop
                queue.push((children, path + [action]), problem.getCostOfActionSequence(path + [action]) + heuristic(children,problem))

    return None
    # util.raiseNotDefined()



bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = UCS