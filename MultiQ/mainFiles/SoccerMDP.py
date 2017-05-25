import numpy as np
import random

class Soccer(object):
    def __init__(self, Nplayers=2, rows=4, cols=5,                              # defaults as per Littman-1994
                 goalposA=np.asarray([(i, -1) for i in range(1, 3)], dtype=int),
                 goalposB=np.asarray([(i, 5) for i in range(1, 3)], dtype=int),
                 defaultPosA=[1,3], defaultPosB=[2,1], defaultBallWith='B'):
        self.Nplayers=2 # TODO: Try removing this hardcoded number of players
        self.playersDict={'A':0,
                          'B':1}
        self.rows=rows
        self.cols=cols
        print "Implementation as per Littman-1994 on Markov games"
        print "This game is hard coded to 2-players for now!"
        print "Note: Rewards used are +-10 instead of +-100 as Greenwald et al used a factor (1-gamma) for reward during Q updates " \
              "(this is equivalent to smaller reward)"

        self.goalposA=goalposA    # goal location for player A
        self.goalposB=goalposB    # goal location for player B
        self.go={'N':       np.array([-1, 0], dtype=int),
                 'S':       np.array([ 1, 0], dtype=int),
                 'W':       np.array([ 0,-1], dtype=int),
                 'E':       np.array([ 0, 1], dtype=int),
                 'stand':   np.array([ 0, 0], dtype=int),
                 }
        self.bogusPos=np.array([ -1, -1], dtype=int)
        self.done = True
        self.defaultPosA=defaultPosA
        self.defaultPosB=defaultPosB
        self.defaultBallWith=defaultBallWith

    def reset(self):
        self.posA=np.array(self.defaultPosA, dtype=int)
        self.posB=np.array(self.defaultPosB, dtype=int)
        self.ballWith=self.defaultBallWith
        self.done=False
        return self.getCurrentState()

    def step(self, actionVec):
        if self.done:   raise Exception("You must reset environment as game has either ended or never started")
        actA=actionVec[0]
        actB=actionVec[1]

        # Evaluate new positions
        newposA=self.posA +self.go[actA]
        newposB=self.posB +self.go[actB]

        # check termination conditions
        if (self.isInGoal(newposA, self.goalposA) and self.ballWith=='A') or (self.isInGoal(newposB, self.goalposA) and self.ballWith=='B'):
            state=(self.bogusPos, self.bogusPos, self.playersDict[self.ballWith])
            rewards=(+10, -10)  # smaller reward than Greenwald et al due to difference in Q equations ( the reward part only)
            self.done=True
            return state, rewards, self.done
        if (self.isInGoal(newposB, self.goalposB) and self.ballWith=='B') or (self.isInGoal(newposA, self.goalposB) and self.ballWith=='A'):
            state=(self.bogusPos, self.bogusPos, self.playersDict[self.ballWith])
            rewards=(-10, +10) # smaller reward than Greenwald et al due to difference in Q equations ( the reward part only)
            self.done=True
            return state, rewards, self.done

        # if not at a goal position, do the usual.
        newposA, validA=self.validPos(newposA)
        if not validA: actA='stand' # if invalid move, stand where you are

        newposB, validB=self.validPos(newposB)
        if not validB: actB='stand'

        if np.all(newposA==self.posB) and np.all(newposB==self.posA):    # a collision scenario where both players try to exchange positions at same time
            actA = 'stand'; actB='stand'                                    # stand where you are
            newposA=self.posA
            newposB=self.posB

        # collision scenarios (include ball exchanges). 2 players can't occupy same spot.
        if np.all(newposA==newposB):                # both try to move to same location
            if actA!='stand' and actB!='stand':             # if both moved, choose randomly among players how occupies empty spot
                if random.choice(['A','B'])=='A':   self.posA=newposA
                else:                               self.posB=newposB

            if actA!='stand' and actB=='stand':             # if A tried to move but B stood
                if self.ballWith=='A':  self.ballWith='B'       # if A carried the ball, B now steals it

            if actA=='stand' and actB!='stand':             # if A stood but B tried to move
                if self.ballWith=='B':  self.ballWith='A'       # if B carried the ball, A now steals it

        else:   # no collison scenario
            self.posA=newposA
            self.posB=newposB

        state=(self.posA, self.posB, self.playersDict[self.ballWith])
        rewards=(0, 0)
        return state, rewards, self.done

    def validPos(self, pos):
        """Assumption. Agent will either violate a row boundary or a column boundary. NOT BOTH"""
        valid=True
        if pos[0]<0:                pos[0]=0; valid=False
        elif pos[0]>=self.rows:     pos[0]=self.rows-1; valid=False

        elif pos[1]<0:              pos[1]=0; valid=False
        elif pos[1]>=self.cols:     pos[1]=self.cols-1; valid=False

        return pos, valid

    def getCurrentState(self):
        if self.done:   return (self.bogusPos, self.bogusPos, self.playersDict[self.ballWith])
        return (self.posA, self.posB, self.playersDict[self.ballWith])

    def getActionMap(self):
        return {i: key for i, key in enumerate(self.go.keys())}

    def isInGoal(self, pos, goalpositions):
        for goalposition in goalpositions:
            if pos[0]==goalposition[0] and pos[1]==goalposition[1]: return True
        return False


if __name__=="__main__":
    env=Soccer()
    env.reset()
    actionMap=env.getActionMap()
    _,actions=zip(*actionMap.iteritems())
    for i in range(100):
        actA=random.choice(actions); actB=random.choice(actions)
        print actA, actB
        state, rewards, done=env.step((actA, actB))
        print state
        print rewards
        if done: print 'done'
        print





