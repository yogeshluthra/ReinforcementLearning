
from settings import *

def initializeTransitionProbTable(withActions=['random']):
    global TransitionProbTable; global NumberOfStates
    for action in withActions:
        TransitionProbTable[action] = \
            np.zeros((NumberOfStates, NumberOfStates),
                     dtype=float)

def setTransition(fromState, withAction, toState, theProb):
    """Sets transition probability fromState toState withProb"""
    global TransitionProbTable
    TransitionProbTable[withAction][fromState, toState]=theProb

def sampleState(fromStateN, withAction):
    """Sample a state withAction fromState using Transition Probability table"""
    global NumberOfStates; global TransitionProbTable; global INVALIDSTATE
    sumProb=0.0; sampledState=INVALIDSTATE;
    randomRoll=np.random.uniform(low=0.0001, high=1.0) # random roll
    for toStateN in range(NumberOfStates):
        sumProb+=TransitionProbTable[withAction][fromStateN, toStateN]
        if randomRoll < sumProb:
            sampledState=toStateN
            break
    if sampledState==INVALIDSTATE: raise AssertionError("No state found")
    return sampledState

def isTerminal(stateN):
    if stateN in terminalStates:
        return True
    return False