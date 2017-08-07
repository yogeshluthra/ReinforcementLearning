import numpy as np

gamma=0.7
directionDist={'N':[[(-1,0), 0.7],
                    [(1,0),0.3]],
               'E':[[(0,1),0.7],
                    [(0,-1),0.3]],
               'S':[[(1,0),0.7],
                    [(-1,0),0.3]],
               'W':[[(0,-1),0.7],
                    [(0,1),0.3]]
                   }
AllActions=['N', 'E', 'S', 'W']
gridShape=(3,4)
Nrows, Ncols=gridShape
AllStates=[(r,c) for r in range(Nrows) for c in range(Ncols)]

rDict={ # D
        (0,3):100.,
        # Z
        (0,0):-100.,
        # W
       (0,1):10.,
       (0,2):10.,
       (1,1):10.,
       (1,3):10.,
       (2,2):10.,
        # P
        (2,1):-30.,
        (2,3):-30.,
       }
Reward=lambda (r,c): rDict.get((r,c), -5.)
terminal_states=set([(0,0),(0,3)])

def getLoc(dr, dc, rinit, cinit, gridShape=gridShape):
    """get correct location for the intended movement"""
    Nrows, Ncols=gridShape
    rmin, rmax, cmin, cmax = 0, Nrows-1, 0, Ncols-1
    r_intended= rinit + dr
    c_intended= cinit + dc
    r_final = r_intended if (r_intended >= rmin and r_intended <= rmax) else rinit
    c_final = c_intended if (c_intended >= cmin and c_intended <= cmax) else cinit
    return r_final, c_final

def getTranDist(r, c, a, gridShape=gridShape):
    """Note x is actually vertical direction. y is horizontal"""
    dist=np.zeros(shape=gridShape, dtype=float)
    dMove_n_dist=directionDist[a]
    for (dr, dc), p in dMove_n_dist:
        r_final, c_final=getLoc(dr, dc, r, c)
        dist[r_final, c_final] += p
    return dist

def getFullTransitionTable():
    tranTable={}
    for r in range(Nrows):
        for c in range(Ncols):
            tranTable[(r, c)]={}
            for a in AllActions:
                tranTable[(r, c)][a]=getTranDist(r, c, a)
    return tranTable

tranP=getFullTransitionTable()
print
print 'Rewards\n........'
for r in range(Nrows):
    for c in range(Ncols):
        state=r,c
        print '{2}'.format(r,c,Reward(state)),'\t',
    print
print

def BellmanUpdate(s, a, Vtable):
    newVal=0.0
    newVal +=Reward(s)
    if s not in terminal_states:
        for sprime in AllStates:
            newVal += gamma*tranP[s][a][sprime]*Vtable[sprime]
    return newVal

def VI():
    # Value iteration
    # - initialize Value table
    V=np.zeros(shape=gridShape)
    Pi=np.array([['N']*Ncols]*Nrows)

    print 'Value Iteration'
    print 'iter ',0
    print V
    print Pi
    print

    # - run for N times
    N=3
    for i in range(N):
        # - create a new table, where updated values will be recorded
        Vnew=np.zeros(shape=gridShape)
        # - loop over all states
        for s in AllStates:
            bestAction, bestVal=None, -1.*np.inf
            # - loop over all actions and find best val and action
            for a in AllActions:
                val_sa = BellmanUpdate(s, a, V)
                if val_sa > bestVal:
                    bestVal=val_sa
                    Vnew[s]=bestVal
                    Pi[s]=a
        V=Vnew
        print 'iter ',i+1
        print np.round(V,2)
        print Pi
        print

def ModifiedPI():
    # Modified Policy iteration
    k=1; N=3
    # - initialize policy and Value table
    Pi=np.array([['N']*Ncols]*Nrows)
    V=np.zeros(shape=gridShape)
    print 'Modified Policy Iteration\n......'
    # - repeat N times (or until convergence)
    for i in range(N):
        # - Policy Evaluation: perform k Bellman updates with last Policy
        for j in range(k):
            Vnew=np.zeros(shape=gridShape)
            for s in AllStates:
                Vnew[s]=BellmanUpdate(s, Pi[s], V)
            V=Vnew
        print 'Policy Eval with last Policy'
        print Pi
        print np.round(V,2)
        print
        # - Improve Policy
        for s in AllStates:
            bestAction, bestVal=None, -1.*np.inf
            for a in AllActions:
                val_sa = BellmanUpdate(s, a, V)
                if val_sa > bestVal:
                    bestVal = val_sa
                    Pi[s]=a
        print 'Policy after Improvement'
        print Pi
        print

VI()
print '\n\n\n..........\n\n'
ModifiedPI()





