
from modelDefinition import *
from settings import *

def initializeWeights(initVal=0.0, random=False):
    global lenObsVec
    if random:
        weightVec=np.random.rand(lenObsVec)
    else:
        weightVec=np.ones((lenObsVec,), dtype=float)*initVal
    return weightVec


def stateVector(stateN):
    """
    Convert state number to state Vector.\n
    :param stateN: state number
    :return: state vector (np 1-D array of a basis vector)
    """
    stateV=np.zeros(lenObsVec, dtype=float)
    stateV[stateN-1]=1.0
    return stateV

def checkPredictionsOf(States, weightVec):
    for stateN in States:
        print "{0}: {1} ".format(stateNum_to_Name[stateN], Predict(stateN, weightVec)),
    print "\n"

def Predict(stateN, weightVec):
    """
    Values of states: {a:0, b:1, c:2, d:3, e:4, f:5, g:6}.\n
    Prediction is simple = weightVec.T * ObservationVector\n
    where ObservatioVector = stateVector\n
    :param stateN: number of current state
    :param weightVec: current weight vector
    :return: prediction of final outcome as seen from current state
    """
    global TerminalReward
    if isTerminal(stateN): return TerminalReward[stateN]
    return np.dot(weightVec.T, stateVector(stateN))

def Dw_pred(stateN, weightVec):
    """
    current Prediction function is simply\n
    Pred(stateN, weightVec) = weightVec.T * stateV\n
    So Dw_pred(stateN, weightVec) = stateV\n
    :param stateN: state number of state
    :param weightVec: Current weight vector
    :return:
    """
    return stateVector(stateN)

def WUpdateForSequence(sequence, weightVec, lam=0.0, alp=0.001):
    """
    Values of states: {a:0, b:1, c:2, d:3, e:4, f:5, g:6}.\n
    We know all sequences start from state D (value 3).

    Args:
        sequence: sequence vector, where each entry is an observation vector (basically state vector)
        lam: lambda value
        alp: alpha value
    Returns:
        WUpdate
    """
    dW=np.zeros(lenObsVec, dtype=float) # initialize update vector for the sequence.
    e_t=np.zeros(lenObsVec, dtype=float) # reset eligibility. start at t=0

    for t in range(len(sequence)):
        stateN=sequence[t] # state number at time=t in this episode of sequence
        if isTerminal(stateN): break # Terminal state seen in sequence

        stateNp1=sequence[t+1] # Number of next state in sequence
        e_t=Dw_pred(stateN, weightVec) + lam*e_t # increment eligibility (see page 16 Sutton 1988)

        dW=dW + alp*(Predict(stateNp1, weightVec)-Predict(stateN, weightVec))*e_t
    return dW

def costFunction(sequence, weightVec):
    """Cost function defined over a sequence"""
    error=0.0
    sequenceOutcome=sequence[-1] # outcome of this sequence
    for stateN in sequence:
        error+=(1.0/2)*(TerminalReward(sequenceOutcome) - Predict(stateN))**2
    return error
def rmsError(weightVec, nonTerminalStates):
    error=0.0
    global idealPrections
    # print 'ideal weights: ', idealPrections,' \npredicted weights: ',
    for stateN in nonTerminalStates:
        error+=(idealPrections[stateN]-Predict(stateN, weightVec))**2
        # print Predict(stateN),' ',
    # print "\n"
    return np.sqrt(error/len(nonTerminalStates))

def isConverged(dW):
    if np.max(np.absolute(dW))<=0.001:
        return True
    return False