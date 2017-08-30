import numpy as np
from helpers import *


def get_CoordinatedEquilibrium(Q=np.asarray([[[0.0, 1.0],
                                                [1.0, 0.5]],
                                            [[0.0, -1.0],
                                                [-1.0, -0.5]]]),
                                actionSpace=(2,2)):
    """Get player values and joint action distribution"""
    # Q=np.asarray([np.random.rand(12), np.random.rand(12), np.random.rand(12)])
    # Np=3
    # actionShape=(2,3,2) # shape of variables (basically ndimensional matrix, each dimension corresponds of an agent with length=N actions of that agent
    V=[]; AllPlayerProbs=[]
    Nplayers=len(actionSpace)
    for player in range(Nplayers):
        Q_player=Q[player].ravel()
        bestJointAction=np.argmax(Q_player)                        # best joint action that this player anticipates
        bestActions=decode(bestJointAction, actionSpace)
        playerActionDistribution = np.zeros(actionSpace[player])
        playerActionDistribution[bestActions[player]] = 1.0         # Place all probability mass on best action for this player. Can't force the other player to do the same
                                                                    # if he is a friend, he will do what is also best of his friend.
        AllPlayerProbs.append(playerActionDistribution)
        V.append(Q_player[bestJointAction])                        # best value that this player can get 'if' this joint action is executed

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
    V, AllPlayerProbs=get_CoordinatedEquilibrium(Q=Q, actionSpace=actionSpace)
    print
    print 'Values of players'
    print V
    print
    print 'Individual policy using Coordinated Equilibrium'
    print AllPlayerProbs
    print
    print
    Q=np.random.rand(1,4,5)
    actionSpace=(5,)
    V, AllPlayerProbs = get_CoordinatedEquilibrium(Q=Q, actionSpace=actionSpace)
    print
    print 'Values of players'
    print V
    print
    print 'Individual policy using Coordinated Equilibrium'
    print AllPlayerProbs