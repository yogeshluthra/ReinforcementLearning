import numpy as np

def encode(state, shape):
    """state vector is expanded from right to left (right=MSB)"""
    encodedState=0
    mult=1
    for dim in np.linspace(len(shape)-1,0,len(shape), dtype=int):
        encodedState += state[dim]*mult
        mult *= shape[dim]
    return encodedState

def decode(discreteState, shape):
    """decodes based on shape from right to left (right=MSB)"""
    decodedState=[]
    mult=1
    for dim in np.linspace(len(shape)-1,0,len(shape), dtype=int):
        remainder=discreteState % (mult*shape[dim])
        bit=remainder//mult
        decodedState.insert(0, bit)
        discreteState -= bit*mult
        mult *= shape[dim]

    return tuple(decodedState)

def createSlicingsForFoeQ(actionSpace=(2,3,4)):
    Nactions=np.cumprod(actionSpace)[-1]
    Nplayers=len(actionSpace)
    slicingAllPlayers=[]
    for player in range(Nplayers):
        Repeats=Nactions/actionSpace[player]
        slicing=[[slice(None)]*Nplayers for i in range(Repeats)]
        slicing=np.asarray(slicing)
        mult=1
        for otherPlayer in range(Nplayers):
            if otherPlayer==player: continue
            sliceIndex = 0
            for i in range(Repeats/(actionSpace[otherPlayer]*mult)):
                for action in range(actionSpace[otherPlayer]):
                    for rep in range(mult):
                        slicing[sliceIndex, otherPlayer]=action
                        sliceIndex+=1
            mult *= actionSpace[otherPlayer]
        slicingAllPlayers.append(slicing)
    slicingAllPlayers=np.asarray(slicingAllPlayers)
    return slicingAllPlayers

if __name__=="__main__":
    for i in range(800):
        decodedState= decode(i, (4,5,4,5,2))
        encodedState=encode(decodedState, (4,5,4,5,2))
        print '{0}\t{1}'.format(decodedState, encodedState)
    print
    print
    print "Slicings for FoeQ"
    print createSlicingsForFoeQ(actionSpace=(3,4,5))
    decodedState = decode(4, (5,))
    encodedState = encode(decodedState, (5,))
    print '{0}\t{1}'.format(decodedState, encodedState)



