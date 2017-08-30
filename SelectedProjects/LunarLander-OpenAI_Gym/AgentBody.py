import numpy as np


class LunarAgent(object):
    def __init__(self, initObs, nObsForState, nActions, ReplayMemMaxSize=1000,
                 actionExplorationParams={'eps': 0.1, 'End': 300,
                                          't1': 300.0, 'tEnd': 0.1,
                                          'eps1': 1.0, 'epsEnd': 0.01}):
        """
        initObs: initial observed state. helps to start the process.
        nObsForState: number of observations from the environment that make up the state.
        nActions: number of available actions
        """
        self.MAXRMEM = ReplayMemMaxSize  # max replay memory size
        self.ObsVecSize = initObs.shape[0]  # size of observation vector returned by environment
        self.nObsForState = nObsForState
        self.stateVecSize = self.ObsVecSize * self.nObsForState
        self.episodeN = 0.0
        self.nA = nActions
        # ---Action exploration params
        self.eps = actionExplorationParams['eps']
        self.End = actionExplorationParams['End']
        self.t1 = actionExplorationParams['t1']
        self.tEnd = actionExplorationParams['tEnd']
        self.eps1 = actionExplorationParams['eps1']
        self.epsEnd = actionExplorationParams['epsEnd']
        self.start_episode(initObs)

    def start_episode(self, initObs):
        """book keeping inside agent at start of an episode"""
        self.episodeN += 1.0
        self.phi = np.asarray([initObs] * self.nObsForState).flatten()
        # ---Gibbs sampling params
        mGibbs = (self.t1 - self.tEnd) / ((1 - self.End) * 1.0)  # temperature slope
        self.tN = np.max((self.t1 + (self.episodeN - 1) * mGibbs, self.tEnd))  # find temperature at episode N
        # ---EPS decay params
        mEDecay = (self.eps1 - self.epsEnd) / ((1 - self.End) * 1.0)  # temperature slope
        self.epsDecayed = np.max((self.eps1 + (self.episodeN - 1) * mEDecay, self.epsEnd))  # find epsilon at episode N
        # ---EPS first
        self.epsFirst = self.eps1 if self.episodeN <= self.End else self.epsEnd

    def getState(self):
        """return copy of current state"""
        return self.phi.copy()

    def updateState(self, envObs):
        """
        update agent's state sequence with currently observed env reaction
        add new observation to head of state vector.
        Return copy of state after update
        """
        self.phi = np.append(envObs, self.phi[:(self.nObsForState - 1) * self.ObsVecSize])
        return self.phi.copy()

    # ---State space exploration strategies------
    def eGreedyAction(self, actionVals):
        """sample epsilon-greedy action from current state"""
        if np.random.rand() > self.eps: return np.argmax(actionVals)
        return np.random.randint(self.nA)

    def gibbsAction(self, actionVals):
        """End: number of episodes for temperature to go from t1 to tEnd"""
        gibbsDist = np.exp(actionVals / self.tN) / np.sum(np.exp(actionVals / self.tN))
        cumProb = np.cumsum(gibbsDist)
        return np.where(cumProb >= np.random.rand())[0][0]

    def epsilonDecay(self, actionVals):
        if np.random.rand() > self.epsDecayed: return np.argmax(actionVals)
        return np.random.randint(self.nA)

    def epsilonFirst(self, actionVals):
        if np.random.rand() > self.epsFirst: return np.argmax(actionVals)
        return np.random.randint(self.nA)
        # -----------------------------------------
