import numpy as np

global lenObsVec # length of observation vector
global NumberOfStates
global terminalStates
global TerminalReward
global stateNum_to_Name
global TransitionProbTable
global INVALIDSTATE
global idealPrections

INVALIDSTATE=-1
idealPrections=np.array([0.0, 1.0/6, 1.0/3, 1.0/2, 2.0/3, 5.0/6, 1.0]) # Sutton 1988 page 20

lenObsVec=5; NumberOfStates=lenObsVec+2;
terminalStates=[0,6]
TerminalReward={0: 0.0,
                6: 1.0}
stateNum_to_Name={0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g'}

TransitionProbTable={}