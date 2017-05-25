import numpy as np
from cvxopt import matrix, solvers
def LP(c=[2, 1],
       Gx_iq_h=[[[-1, 1], 'le', 1],
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
    for inequality in Gx_iq_h:
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

def get_uCE(Q=np.asarray([[[6,2],
                                [7,0]],
                            [[6,7],
                                [2,0]]]),
            actionSpace=(2, 2)):
    """Get player values and joint action distribution"""
    # Q=np.asarray([np.random.rand(12), np.random.rand(12), np.random.rand(12)])
    # Np=3
    # actionShape=(2,3,2) # shape of variables (basically ndimensional matrix, each dimension corresponds of an agent with length=N actions of that agent

    Nplayers=len(actionSpace)
    Nvars = np.cumprod(actionSpace)[-1] # number of variables to deal with
    Gx_iq_h=[]  # inequality constraints
    for player in range(Nplayers):
        Qp_rshp=Q[player].reshape(actionSpace) # reshaped to match variable locations
        for conditionedAction in range(actionSpace[player]): # for every player n*(n-1) number of inequalities
            cAction = [slice(None)] * Nplayers   # create slicing pattern of :,:,:
            cAction[player]=conditionedAction            # set index of conditioned action for selected player
            for unconditionedAction in range(actionSpace[player]):
                if unconditionedAction==conditionedAction: continue
                vars = np.zeros(shape=actionSpace)  # action shape same as distribution shape over actions
                ucAction = [slice(None)] * Nplayers  # create slicing pattern of :,:,:
                ucAction[player]=unconditionedAction         # set index of un-conditioned action for selected player
                vars[cAction]=Qp_rshp[cAction] - Qp_rshp[ucAction]    # create inequality
                Gx_iq_h.append([vars.ravel(), 'ge', 0.0])

    # Impose valid prob distribution
    for i in range(Nvars):  # each prob >= 0.0
        vars=np.zeros(Nvars)
        vars[i]=1.0
        Gx_iq_h.append([vars, 'ge', 0.0])

    Ax_eq_b=[[np.ones(Nvars), 'eq', 1.0]] # equality constraints: sum probs = 1

    # Optimization Objective    # uCE-Q
    c=np.zeros(Nvars)
    for Qp in Q:    c +=Qp.ravel()*-1.0 # maximization

    ActionProbDist=LP(c=c, Gx_iq_h=Gx_iq_h, Ax_eq_b=Ax_eq_b) # SOLVE LP
    if ActionProbDist.shape[0]==0:
        print 'WARNING: no solution found'
        ActionProbDist=np.ones(Nvars)*1.0/(Nvars*1.0) # default uniform disribution in case of no solution

    V=[]
    for player in range(Nplayers):
        Vplayer=np.dot(Q[player].ravel(), ActionProbDist)
        V.append(Vplayer)
    return np.asarray(V, dtype=float), ActionProbDist

if __name__=="__main__":
    print get_uCE()
    print
