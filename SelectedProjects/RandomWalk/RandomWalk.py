import os

from modelDefinition import *
from settings import *

def RandomWalkSequence(fromState, ofNSequences, writeToFile):
    for i in range(ofNSequences):
        currState = fromState
        theSequence=[]
        while not isTerminal(currState):
            theSequence.append(currState)
            currState=sampleState(currState, withAction='random')
        theSequence.append(currState)
        for aState in theSequence:
            writeToFile.write("{0} ".format(aState))
        writeToFile.write("\n")

def checkNcreateTrainSets(NTrainingSets, NSequencesPerTrainSet, RunDir):
    TrainingSetCreatingStamp = RunDir + '__TrainingSetsExist__'
    if not os.path.exists(TrainingSetCreatingStamp):
        print "\nCreating Training Sets\n\n"
        with open(TrainingSetCreatingStamp, 'a'):
            os.utime(TrainingSetCreatingStamp, None)
        for i in range(NTrainingSets):
            trainSetPath = RunDir + 'trainSet_' + str(i) + '.txt'
            with open(trainSetPath, 'w') as trainSetFile:
                RandomWalkSequence(3, 10, trainSetFile)
    else:
        print "\nTraining Sets exist.\nIf you would like to generate Training Sets again, remove following file\n" + TrainingSetCreatingStamp + "\n\n"