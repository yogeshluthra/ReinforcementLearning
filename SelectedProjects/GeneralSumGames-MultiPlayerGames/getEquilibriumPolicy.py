import numpy as np
import random
from SoccerMDP import *
from MultiQ_agent import *
from helpers import *
from get_uCorrelatedEquilibrium import get_uCE
from get_AdversarialEquilibrium import get_AdversarialEquilibrium
from get_CoordinatedEquilibrium import get_CoordinatedEquilibrium


# uCEQ_table=np.fromfile('uCE_Q.txt', sep=",")
uCEQ_table=np.fromfile('Foe_Q.txt', sep=",")
uCEQ_table=uCEQ_table.reshape(2,32,25)
print "PLAYER 1 final Q table"
print uCEQ_table[0]
print
print
print "PLAYER 2 final Q table"
print uCEQ_table[1]
print
print
print "SUMMING two tables"
print np.add(uCEQ_table[0], uCEQ_table[1])
print
print

def get_dS(state, stateSpace):
    posA, posB, playerWithBall = state
    dS = encode((posA[0], posA[1], posB[0], posB[1], playerWithBall), stateSpace)  # discretize state
    return dS

Nplayers=2  # TODO: currently hard coded in Soccer environment as well. In future, try to generalize it.
Nrows=2
Ncols=2
Duals=2 # number of states possible for each position of a players
stateSpace=(Nrows, Ncols, Nrows, Ncols, Duals)
Nstates=np.cumprod(stateSpace)[-1]

env=Soccer(rows=Nrows, cols=Ncols,
            goalposA=np.asarray([(i, -1)    for i in range(0, Nrows)], dtype=int),
            goalposB=np.asarray([(i, Ncols) for i in range(0, Nrows)], dtype=int),
            defaultPosA=[0, 1], defaultPosB=[0, 0], defaultBallWith='B')

envActionMap=env.getActionMap()                    # get action map
actionSpace=(len(envActionMap), len(envActionMap))    # create action space. Both players have same number of actions.

#---Initialize
s=env.reset()               # reset environment
ds=get_dS(s, stateSpace)    # discretize state
playerSlicings=createSlicingsForFoeQ(actionSpace=actionSpace)

# get converged policies
V, AllPlayerProbs=get_AdversarialEquilibrium(Q=uCEQ_table[:, ds, :], actionSpace=actionSpace, playerSlicings=playerSlicings)
print "Adversarial Equilibrium"
print 'value of players'
print V
print
print 'Action probabilities'
print AllPlayerProbs
print
print envActionMap
print
print
print "Player's Q value"
print uCEQ_table[:, ds, :]
print "sum along axis=1"
print uCEQ_table[:, ds, :].sum(axis=1)
print
print
V, jointActionProbs=get_uCE(Q=uCEQ_table[:, ds, :], actionSpace=actionSpace)
print "Correlated equilibrium"
print "value of players"
print V
print
print 'Joint Action probabilities'
print jointActionProbs.reshape(actionSpace)
print
print envActionMap
print
print
V, AllPlayerProbs=get_CoordinatedEquilibrium(Q=uCEQ_table[:, ds, :], actionSpace=actionSpace)
print "Coordinated Equilibrium"
print 'value of players'
print V
print
print 'Action probabilities'
print AllPlayerProbs
print envActionMap


