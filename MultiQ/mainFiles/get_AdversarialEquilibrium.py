import numpy as np
from cvxopt import matrix, solvers
from helpers import *
def LP(c=[2, 1],
        Gx_gt_h=[[[-1,1],'le',1],
                [[1,1],'ge',2],
                [[0,1],'ge',0],
                [[1,-2],'le',4]],
        Ax_eq_b=[[[0,0],'eq',0]],
        verbose=False):

    G=[]
    h=[]
    A=[]
    b=[]
    c=matrix(np.asarray(c) * 1.0)
    for inequality in Gx_gt_h:
        LHS=np.asarray(inequality[0])*1.0
        RHS=inequality[-1]*1.0
        if inequality[1]=='ge' or inequality[1]=='gt':
            LHS=LHS*-1.0
            RHS=RHS*-1.0
        G.append(LHS)
        h.append(RHS)
    for equality in Ax_eq_b:
        LHS = np.asarray(equality[0])*1.0
        RHS = equality[-1]*1.0
        A.append(LHS)
        b.append(RHS)
    G=matrix(np.asarray(G))
    h=matrix(np.asarray(h))
    A=np.asarray(A)
    b=np.asarray(b)
    if np.all(A==0.0):  A=None; b=None
    else:               A=matrix(A); b=matrix(b)
    sol=solvers.lp(c, G=G, h=h, A=A, b=b, options={'show_progress':verbose,
                                                   # 'abstol':1e-12,
                                                   # 'feastol':1e-12,
                                                   # 'reltol':1e-12,
                                                   'maxiters': 1000})
    return np.array(sol['x']).ravel()

def get_AdversarialEquilibrium(Q=np.asarray([[[0.0, 1.0],
                                                [1.0, 0.5]],
                                            [[0.0, -1.0],
                                                [-1.0, -0.5]]]),
                            actionSpace=(2,2),
                            playerSlicings=np.asarray([[[slice(None)]*2 for i in range(2)] for j in range(2)])):
    """Get player values and joint action distribution"""
    # Q=np.asarray([np.random.rand(12), np.random.rand(12), np.random.rand(12)])
    # Np=3
    # actionShape=(2,3,2) # shape of variables (basically ndimensional matrix, each dimension corresponds of an agent with length=N actions of that agent
    V=[]; AllPlayerProbs=[]
    Nplayers=len(actionSpace)
    for player in range(Nplayers):
        Nprobs = actionSpace[player]        # number of probability values to deal with for player
        Nvars=Nprobs+1                      # total variables = prob + 1. Extra variable is the optimization objective
        Gx_gt_h = []                        # inequality constraints
        #---Impose valid prob distribution
        for i in range(Nprobs):  # each prob >= 0.0
            vars = np.zeros(Nvars)
            vars[i] = 1.0
            Gx_gt_h.append([vars.ravel(), 'ge', 0.0])
        Ax_eq_b = [[np.append(np.ones(Nprobs), 0.0), 'eq', 1.0]]  # equality constraints: sum probs = 1. Extra 0 append to prob vector to adjust for total number of variables
        #-------------

        Qp_rshp=Q[player].reshape(actionSpace)  # reshaped to match variable locations
        slicing=playerSlicings[player]          # get slicing matrix for player
        for aSlice in slicing:
            vars=np.zeros(Nvars); vars[-1] = -1.0 # last element in each LHS of inequality is optimization objective parameter
            vars[0:Nprobs]=Qp_rshp[list(aSlice)]
            Gx_gt_h.append([vars.ravel(), 'ge', 0.0])

        c=np.zeros(Nvars); c[-1] = -1.0 # maximization objective
        solvedVars=LP(c=c, Gx_gt_h=Gx_gt_h, Ax_eq_b=Ax_eq_b) # SOLVE LP
        playerProbs=solvedVars[:-1]
        valueOfGameForPlayer=solvedVars[-1]
        if playerProbs.shape[0]==0:
            print 'WARNING: no solution found'
            playerProbs=np.ones(Nprobs)*1.0/(Nprobs*1.0) # default uniform disribution in case of no solution
        AllPlayerProbs.append(playerProbs)
        V.append(valueOfGameForPlayer)

    # V=[]
    # for player in range(Nplayers):
    #     Vplayer=np.dot(Q[player].ravel(), ActionProbDist)
    #     V.append(Vplayer)
    return np.asarray(V, dtype=float), np.asarray(AllPlayerProbs, dtype=float)

if __name__=="__main__":
    # Q=np.asarray([[[20, -10, 5],
    #                 [5, 10, -10],
    #                     [-5, 0, 10]],
    #              [[-20, 10, -5],
    #                 [-5, -10, 10],
    #                     [5, 0, -10]]])
    # actionSpace = (3,3)

    #---for Shaun
    Q1_player = np.array(
        [[0.82563298, 0.74575045, -4.44152839, 1, 1.],
         [1, 0.9419736, -4.29208447, 0.79823645, 1.],
         [1, 1, -6.74334617, 1, 0.441383],
         [1, 4.16519806, -8.98898896, 1, 1.],
         [1, 0.81913688, -4.11509104, 1, 1.]])
    Q2_player = Q1_player*-1.0
    Q = np.zeros(shape=(2,5,5))
    Q[0]=Q1_player
    Q[1]=Q2_player
    actionSpace=(5,5)
    #---------------------
    playerSlicings=createSlicingsForFoeQ(actionSpace=actionSpace)
    V, AllPlayerProbs=get_AdversarialEquilibrium(Q=Q, actionSpace=actionSpace, playerSlicings=playerSlicings)
    print
    print 'Values of players'
    print V
    print
    print 'Individual policy using Adversarial Equilibrium'
    print AllPlayerProbs
    print
