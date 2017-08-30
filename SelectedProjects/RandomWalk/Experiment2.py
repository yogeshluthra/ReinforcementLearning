
import matplotlib.pyplot as plt
import numpy as np
import os

import RandomWalk
from DynamicProgrammingFuncs import *
from modelDefinition import *
from settings import *

def main():
    # np.random.seed() # set seed
    initializeTransitionProbTable(withActions=['random']) # this code only implements random actions
    setTransition(1, 'random', 0, 0.5); setTransition(1, 'random', 2, 0.5)
    setTransition(2, 'random', 1, 0.5); setTransition(2, 'random', 3, 0.5)
    setTransition(3, 'random', 2, 0.5); setTransition(3, 'random', 4, 0.5)
    setTransition(4, 'random', 3, 0.5); setTransition(4, 'random', 5, 0.5)
    setTransition(5, 'random', 4, 0.5); setTransition(5, 'random', 6, 0.5)

    NTrainingSets=100; NSequencesPerTrainSet=10
    RunDir='./'
    RandomWalk.checkNcreateTrainSets(NTrainingSets, NSequencesPerTrainSet, RunDir)

    # Second experiment. Initialize weigths at start of presenting every training set. Basically training is on a training Set.
    lamTrials=np.linspace(0.0,1.0,11)
    aplhaTrials=np.linspace(0.0,0.5,11)
    averageRMSError={}; SME={}

    for lam in lamTrials:
        averageRMSError[lam]=[]
        SME[lam]=[]
        for alp in aplhaTrials:
            RMSError_2ndExp=np.zeros(NTrainingSets, dtype=float)
            for trainSetIndex in range(NTrainingSets):
                weightVec=initializeWeights(initVal=0.5, random=False)  # initialize weigths for every new training set
                with open(RunDir + 'trainSet_' + str(trainSetIndex) + '.txt') as currTrainSet:
                    for line in currTrainSet:
                        sequence=map(int, line.strip().split(' '))
                        if len(sequence)==0: raise IndexError("length of sequence cant be 0")
                        dW=WUpdateForSequence(sequence, weightVec, lam=lam, alp=alp)
                        weightVec+= dW # update weight Vector after each sequence
                RMSError_2ndExp[trainSetIndex]=rmsError(weightVec, [1,2,3,4,5])
            averageRMSError[lam].append(np.mean(RMSError_2ndExp)) # average out the error over training sets
            SME[lam].append(np.std(RMSError_2ndExp)/np.sqrt(NTrainingSets))
    for lam in lamTrials:
        plt.plot(aplhaTrials, averageRMSError[lam], label=str(lam))
    plt.legend(loc='Upper Left')
    plt.title('Average Error')
    plt.grid()
    plt.show()
    for lam in lamTrials:
        plt.plot(aplhaTrials, SME[lam], label=str(lam))
    plt.legend(loc='Upper Left')
    plt.title('Std Error of Sampled Mean')
    plt.grid()
    plt.show()
    bestVals=[]
    for lam in lamTrials:
        bestVals.append(np.min(averageRMSError[lam]))
    minVal=np.min(bestVals)
    atLambdaInd=np.argmin(bestVals)
    plt.text(0.1,0.13, "min val={0} \nat lambda={1}".format(minVal, lamTrials[atLambdaInd]))
    plt.plot(lamTrials, bestVals)
    plt.title('Average Error with Best learning rates')
    plt.grid()
    plt.show()

if __name__=='__main__':
    main()
