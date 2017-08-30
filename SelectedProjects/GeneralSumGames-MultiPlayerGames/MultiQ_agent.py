import numpy as np
import random
from helpers import *

class MultiQ_agent(object):
    def __init__(self, Nstates=800, actionSpace=(5, 5),
                 actionExplorationParams={'eps': 0.1, 'End': 1e5,
                                          't1': 300.0, 'tEnd': 0.1,
                                          'eps1': 1.0, 'epsEnd': 0.01}, is_ActionDistr_independent=False):
        self.Nplayers=len(actionSpace)              # number of players
        self.Nstates = Nstates              # Nstates
        self.NjointActions=np.cumprod(actionSpace)[-1]  # number of joint actions (based on actionShape)
        self.actionSpace = actionSpace      # shape of actions
        self.iterations=0.0
        # ---Action exploration params
        self.eps = actionExplorationParams['eps']
        self.End = actionExplorationParams['End']
        self.t1 = actionExplorationParams['t1']
        self.tEnd = actionExplorationParams['tEnd']
        self.eps1 = actionExplorationParams['eps1']
        self.epsEnd = actionExplorationParams['epsEnd']
        self.jointActions=np.arange(0, self.NjointActions, dtype=int)
        #---Main multi agent Q table
        self.Q=np.zeros(shape=(self.Nplayers, self.Nstates, self.NjointActions), dtype=float)
        #-----------
        self.is_ActionDistr_independent=is_ActionDistr_independent    # to use the code interchangably between algos which work on joint as well as separate action prob distributions

    def makeUpdates(self):
        self.iterations += 1.0
        # ---Gibbs sampling params
        mGibbs = (self.t1 - self.tEnd) / ((1 - self.End) * 1.0)  # temperature slope
        self.tN = np.max((self.t1 + (self.iterations - 1) * mGibbs, self.tEnd))  # find temperature at iteration N
        # ---EPS decay params
        mEDecay = (self.eps1 - self.epsEnd) / ((1 - self.End) * 1.0)  # temperature slope
        self.epsDecayed = np.max((self.eps1 + (self.iterations - 1) * mEDecay, self.epsEnd))  # find epsilon at iteration N
        # ---EPS first
        self.epsFirst = self.eps1 if self.iterations <= self.End else self.epsEnd

    def get_V_and_ActionDistribution(self, dS, func, kwargs={}):
        """get action distribution using function func for state dS (the discretized state)"""
        if self.iterations == 1.0:
            if not self.is_ActionDistr_independent:
                return np.zeros(self.Nplayers), np.ones(self.NjointActions, dtype=float) * 1.0 / (self.NjointActions * 1.0)  # initial action distribution is set uniform
            else:
                return np.zeros(self.Nplayers),\
                       np.asarray([np.ones(self.actionSpace[player], dtype=float) * 1.0 / (self.actionSpace[player] * 1.0)
                                   for player in range(self.Nplayers)], dtype=float)

        return func(self.Q[:,dS,:], self.actionSpace, **kwargs)

    # ---State space exploration strategies------
    def eGreedyAction(self, distribution):
        """sample epsilon-greedy action from current state. Every agent selects to behave individually"""
        finalizedActions = []
        if not self.is_ActionDistr_independent: # we are looking at joint distributions
            num_JointAction = np.where(np.random.rand() < distribution.cumsum())[0][0]
            OptimalJointActions=decode(num_JointAction, self.actionSpace)        # Optimal Joint Actions as per strategy (carried in distribution)
            for player in range(self.Nplayers):
                if np.random.rand() > self.eps:
                    finalizedActions.append(OptimalJointActions[player])                        # choose optimal action for that player
                else:
                    finalizedActions.append(np.random.randint(0, self.actionSpace[player]))     # else choose an action uniformly at random for that player
        else:               # we are looking at individual distributions
            for player in range(self.Nplayers):
                if np.random.rand() > self.eps:
                    finalizedActions.append(np.where(np.random.rand() < distribution[player].cumsum())[0][0])   # choose optimal action for that player
                else:
                    finalizedActions.append(np.random.randint(0, self.actionSpace[player]))                     # else choose an action uniformly at random for that player

        return tuple(finalizedActions)

    def epsilonDecay(self, distribution):
        """sample action from current state according with decaying epsilon. Every agent selects to behave individually"""
        finalizedActions = []
        if not self.is_ActionDistr_independent:  # we are looking at joint distributions
            num_JointAction = np.where(np.random.rand() < distribution.cumsum())[0][0]
            OptimalJointActions = decode(num_JointAction,
                                         self.actionSpace)  # Optimal Joint Actions as per strategy (carried in distribution)
            for player in range(self.Nplayers):
                if np.random.rand() > self.epsDecayed:
                    finalizedActions.append(OptimalJointActions[player])  # choose optimal action for that player
                else:
                    finalizedActions.append(np.random.randint(0, self.actionSpace[player]))  # else choose an action uniformly at random for that player
        else:  # we are looking at individual distributions
            for player in range(self.Nplayers):
                if np.random.rand() > self.epsDecayed:
                    finalizedActions.append(np.where(np.random.rand() < distribution[player].cumsum())[0][0])  # choose optimal action for that player
                else:
                    finalizedActions.append(np.random.randint(0, self.actionSpace[player]))  # else choose an action uniformly at random for that player

        return tuple(finalizedActions)

    def epsilonFirst(self, distribution):
        """sample action from current state with high randomness upto 'End' number of runs, and with low prob afterwards. Every agent selects to behave individually"""
        finalizedActions = []
        if not self.is_ActionDistr_independent:  # we are looking at joint distributions
            num_JointAction = np.where(np.random.rand() < distribution.cumsum())[0][0]
            OptimalJointActions = decode(num_JointAction,
                                         self.actionSpace)  # Optimal Joint Actions as per strategy (carried in distribution)
            for player in range(self.Nplayers):
                if np.random.rand() > self.epsFirst:
                    finalizedActions.append(OptimalJointActions[player])  # choose optimal action for that player
                else:
                    finalizedActions.append(np.random.randint(0, self.actionSpace[player]))  # else choose an action uniformly at random for that player
        else:  # we are looking at individual distributions
            for player in range(self.Nplayers):
                if np.random.rand() > self.epsFirst:
                    finalizedActions.append(np.where(np.random.rand() < distribution[player].cumsum())[0][0])  # choose optimal action for that player
                else:
                    finalizedActions.append(np.random.randint(0, self.actionSpace[player]))  # else choose an action uniformly at random for that player

        return tuple(finalizedActions)

if __name__=="__main__":
    for i in range(100):
        print decode(i, (4,5,5))
