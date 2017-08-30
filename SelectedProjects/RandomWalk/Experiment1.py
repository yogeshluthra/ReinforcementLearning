
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

    # First experiment. Initialize weigths at start of presenting every training set. Basically training is on a training Set.
    lamTrials=np.linspace(0.0,1.0,11)
    averageRMSErrorVSLambda_1stExp=[]; SME_VSLamda_1stExp=[]
    Repeats_1stExp = 1000

    for lam in lamTrials:
        RMSError_1stExp=np.zeros(NTrainingSets, dtype=float)
        for trainSetIndex in range(NTrainingSets):
            weightVec=initializeWeights(initVal=0.5, random=True)  # initialize weigths for every new training set
            for aRepeat in range(Repeats_1stExp):
                with open(RunDir + 'trainSet_' + str(trainSetIndex) + '.txt') as currTrainSet:
                    dW = np.zeros(lenObsVec, dtype=float) # reset weigth updates
                    for line in currTrainSet:
                        sequence=map(int, line.strip().split(' '))
                        if len(sequence)==0: raise IndexError("length of sequence cant be 0")
                        dW+=WUpdateForSequence(sequence, weightVec, lam=lam, alp=0.01)
                weightVec+= dW # update weight Vector after shown a training set
                if isConverged(dW): break
            RMSError_1stExp[trainSetIndex]=rmsError(weightVec, [1,2,3,4,5])
        averageRMSErrorVSLambda_1stExp.append(np.mean(RMSError_1stExp)) # average out the error over training sets
        SME_VSLamda_1stExp.append(np.std(RMSError_1stExp)/np.sqrt(NTrainingSets))
    plt.plot(lamTrials, averageRMSErrorVSLambda_1stExp)
    plt.title('Average Error')
    plt.show()
    plt.plot(lamTrials, SME_VSLamda_1stExp)
    plt.title('Std Error of Sampled Mean')
    plt.show()

if __name__=='__main__':
    main()
