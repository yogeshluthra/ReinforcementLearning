"""Q learning Algo for 2 player Soccer game"""
import numpy as np
import random
from SoccerMDP import *
from MultiQ_agent import *
from helpers import *
from get_CoordinatedEquilibrium import get_CoordinatedEquilibrium
import time

starttime=time.time()
def get_2DiscreteStates(state, stateSpace):
    posA, posB, playerWithBall = state
    dSA = encode((posA[0], posA[1], playerWithBall), stateSpace)  # discretize state
    dSB = encode((posB[0], posB[1], playerWithBall), stateSpace)  # discretize state
    return dSA, dSB

Nplayers=2  # TODO: currently hard coded in Soccer environment as well. In future, try to generalize it.
Nrows=2
Ncols=2
Duals=2 # number of states possible for each position of a players
stateSpace=(Nrows, Ncols, Duals)    # each player has his individual state space
Nstates=np.cumprod(stateSpace)[-1]

AlgorithmName="Q_learning"           # TODO: Make sure name is correct
is_ActionDistr_independent=True       # TODO: make sure setting correct adversarial flag. Correlated Q has joint action distribution.
End=1000000
decay=0.01**(1/(End*1.0))  # alpha decay (Greenwald uses alpha --> 0.001)
gamma=0.9
actionExplorationParams={'eps': 0.2, 'End': End,
                        't1': 300.0, 'tEnd': 0.1,
                        'eps1': 1.0, 'epsEnd': 0.01}


env=Soccer(rows=Nrows, cols=Ncols,
            goalposA=np.asarray([(i, -1)    for i in range(0, Nrows)], dtype=int),
            goalposB=np.asarray([(i, Ncols) for i in range(0, Nrows)], dtype=int),
            defaultPosA=[0, 1], defaultPosB=[0, 0], defaultBallWith='B')

envActionMap=env.getActionMap()                                     # get action map
actionSpace=(len(envActionMap),)                                     # create action space. Each player has his own action space (doesn't bother about what other is doing).
playerSlicings=createSlicingsForFoeQ(actionSpace=actionSpace)       # create slicings for FoeQ. This is done just done once and passed as kwarg to get_AdversarialEquilibrium
Nactions=np.cumprod(actionSpace)[-1]                                # get number of joint actions (exponential in numner of players)
monitoractA_name='S'; monitoractB_name='stand'                      # get joint action to be monitored for Q convergence

agentA=MultiQ_agent(Nstates=Nstates, actionSpace=actionSpace, actionExplorationParams=actionExplorationParams,
                        is_ActionDistr_independent=is_ActionDistr_independent)
agentB=MultiQ_agent(Nstates=Nstates, actionSpace=actionSpace, actionExplorationParams=actionExplorationParams,
                        is_ActionDistr_independent=is_ActionDistr_independent)

actionSelectionA=agentA.eGreedyAction
actionSelectionB=agentB.eGreedyAction

#---Initialize
s=env.reset()               # reset environment
dSA, dSB=get_2DiscreteStates(s, stateSpace)    # discretize state

#---Set monitors
monitorState=dSA             # state to be monitored for Q convergence
for actNum, actName in envActionMap.iteritems():
    if monitoractA_name==actName:   monitoractA_num=actNum
monitorAction=monitoractA_num
Qmonitor=[]
#-----------

agentA.makeUpdates()  # initialize agent params
agentB.makeUpdates()  # initialize agent params
_, actionDistributionA=agentA.get_V_and_ActionDistribution(dSA, get_CoordinatedEquilibrium)
_, actionDistributionB=agentB.get_V_and_ActionDistribution(dSB, get_CoordinatedEquilibrium)
actA, = actionSelectionA(actionDistributionA)
actB, = actionSelectionB(actionDistributionB)
alpha=1.0
#---------

for iteration in range(End):
    Qmonitor.append(agentA.Q[0, monitorState, monitorAction])       # if state being monitored is visited, record the value
    # print "{0}\n{1} {2}\n".format(s, actionMap[actA], actionMap[actB])
    sprime, rewards, done=env.step((envActionMap[actA], envActionMap[actB]))        # take step in environment
    dSAprime, dSBprime = get_2DiscreteStates(sprime, stateSpace)                    # discretize state
    VSAprime, actionDistributionA=agentA.get_V_and_ActionDistribution(dSAprime, get_CoordinatedEquilibrium) # get next state value and action distribution based on current Q table
    VSBprime, actionDistributionB=agentB.get_V_and_ActionDistribution(dSBprime, get_CoordinatedEquilibrium) # get next state value and action distribution based on current Q table

    if done:
        agentA.Q[0, dSA, actA]=\
            (1 - alpha) * agentA.Q[0, dSA, actA] + alpha * (rewards[0])
        agentB.Q[0, dSB, actB] = \
            (1 - alpha) * agentB.Q[0, dSB, actB] + alpha * (rewards[1])
        s=env.reset()               # reset environment
        dSA, dSB=get_2DiscreteStates(s, stateSpace)    # discretize state
        _, actionDistributionA = agentA.get_V_and_ActionDistribution(dSA, get_CoordinatedEquilibrium)
        _, actionDistributionB = agentB.get_V_and_ActionDistribution(dSB, get_CoordinatedEquilibrium)

    else:
        agentA.Q[0, dSA, actA] = \
            (1 - alpha) * agentA.Q[0, dSA, actA] + alpha * (rewards[0] + gamma * VSAprime[0])
        agentB.Q[0, dSB, actB] = \
            (1 - alpha) * agentB.Q[0, dSB, actB] + alpha * (rewards[1] + gamma * VSBprime[0])
        s=sprime
        dSA=dSAprime; dSB=dSBprime

    agentA.makeUpdates()  # prepare agent for next episode
    agentB.makeUpdates()
    alpha *= decay
    actA, = actionSelectionA(actionDistributionA)
    actB, = actionSelectionB(actionDistributionB)

Qmonitor=np.asarray(Qmonitor, dtype=float)
Qdiff=np.absolute(Qmonitor[1:] - Qmonitor[0:-1])
#---Forward filling
IndexProxy=np.arange(Qdiff.shape[0], dtype=int)
IndexProxy[Qdiff<=0.005] = 0
IndexProxy=np.maximum.accumulate(IndexProxy)
Qdiff=Qdiff[IndexProxy]

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
plt.plot(Qdiff)
plt.ylim(0.0, 0.5)
plt.yticks(np.linspace(0,0.5, num=11))
plt.savefig('{0}.png'.format(AlgorithmName))

endtime=time.time()
print '\n\t --> total run time={0}s <--'.format(endtime-starttime)
print
saveQ_infile="{0}.txt".format(AlgorithmName)
print "saving Q table in "+saveQ_infile
agentA.Q.tofile(saveQ_infile, sep=",", format='%10.5f')
print
print """To recover the Q table into an array 'x', use x.reshape(2,32,25) ...
                (as this array is 2 players, 32 states per player, 25 joint actions per state)
        Then we will have our converged Q table back """


