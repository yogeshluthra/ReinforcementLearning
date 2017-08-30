import logging
import random
from collections import deque

import numpy as np
import tensorflow as tf
from time import time

import gym
from AgentBody import LunarAgent
from AgentBrain import MultiLayerNN
from gym import wrappers
from datetime import datetime
import os

def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)
experimentList=[
    # (eps1, learning_rate)
    # (1.0, 2.5e-4),
    # (1.0, 2.5e-5),
    # (1.0, 2.5e-3),
    (0.7, 2.5e-4),
    # (0.7, 2.5e-5),
    # (0.7, 2.5e-3),
    # (0.4, 2.5e-4),
    # (0.4, 2.5e-5),
    # (0.4, 2.5e-3),

    # (eps1, End, Niterations_to_targetNetwork_update)
    # (0.8, 300, 100),
    # (0.7, 200, 100),
    # (0.7, 300, 100),
    # (0.7, 200, 300),
    # (0.5, 200, 100),
    # (0.5, 200, 300),
    # (0.3, 300, 100)
]
# for ExplorationRate in [1.0, 0.7, 0.4]:
#     for NetworkLearnRate in [2.5e-4, 2.5e-5, 2.5e-3]:
for ExplorationRate, LearningRate in experimentList:
    now = datetime.now()
    summaries_dir = "./summary/" + now.strftime("%Y%m%d-%H%M%S") + "/"
    outdir = "/tmp/lunarlander2-agent-results/" + now.strftime("%Y%m%d-%H%M%S")

    key = 'eps{0}_alpha{1}'.format(ExplorationRate, LearningRate)

    rootLogger=logging.getLogger('DoubleDQN'); rootLogger.setLevel(logging.DEBUG)
    logfile = os.getcwd() + '/' + key + ".log"
    logFileHandler=logging.FileHandler(logfile, mode='w'); logFileHandler.setLevel(logging.DEBUG)
    rootLogger.addHandler(logFileHandler)
    rootLogger.debug("For gym upload, results in {}".format(outdir))

    #---Hyper params---------------
    gamma = 0.99  # earlier 0.75.. earlier 0.5.. earlier 0.99. behaved wildly.. earlier 0.9
    actionExplorationParams = {'eps': 0.1,
                               'End': 600,
                               't1': 300.0, 'tEnd': 0.1,
                               'eps1': ExplorationRate, 'epsEnd': 0.00}
    learning_rate = LearningRate
    beta = 0.0000

    nObsForState = 1
    ReplayMemMaxSize = 1000000  # replay memory max size.
    batch_size = min(200, int(ReplayMemMaxSize / 100))
    target_network_update_every = 500  # iterations. chance of selecting a sample 1-((ReplayMemMaxSize-batch_size)/ReplayMemMaxSize)^iterations
    nEpisodes = 1000
    #-------------------------------
    rootLogger.debug("gamma={0}\nexploration={1}\nlearning_rate={2}\nbeta={3}\nnObsForState={4}\nReplayMemMaxSize={5}\nbatch_size={6}\ntarget_network_update_every={7}\n".format(
                gamma, actionExplorationParams, learning_rate, beta, nObsForState, ReplayMemMaxSize, batch_size, target_network_update_every))

    replayMemory = deque()

    env = gym.make('LunarLander-v2')
    env = wrappers.Monitor(env, outdir, force=True)#, video_callable=True)

    nActions = env.action_space.n

    currObs = env.reset()
    agent = LunarAgent(currObs, nObsForState, nActions, ReplayMemMaxSize=ReplayMemMaxSize,
                       actionExplorationParams=actionExplorationParams)
    obsVecSize = agent.ObsVecSize
    estimator = MultiLayerNN(agent.stateVecSize, nActions, scope="q",
                             learning_rate=learning_rate, beta=beta, summaries_dir=summaries_dir)
    target_network = MultiLayerNN(agent.stateVecSize, nActions, scope="target",
                                  learning_rate=learning_rate, beta=beta, summaries_dir=summaries_dir)
    # ---get initial observation vectors for statistics in observation vector--------
    obsVectors=[]
    for i in range(3000):
        action = np.random.randint(nActions)
        currObs, reward, done, _ = env.step(action); obsVectors.append(currObs)
        if done:
            currObs = env.reset()
    # ----------------------------------
    obsVectors=np.asarray(obsVectors)
    meanObs=obsVectors.mean(axis=0); stdDev=obsVectors.std(axis=0)
    meanObs[-2:]=0.0; stdDev[-2:]=1.0
    obsVectors = (obsVectors - meanObs) / stdDev

    # ---Reset some of agent's variables
    agent.episodeN = 0.0
    currObs = (currObs - meanObs) / stdDev
    agent.start_episode(currObs)
    currState = agent.getState()

    rootLogger.info("episodeN,runTime,episodeIterations,reward")
    episodeReward = 0.0; episodeIterations = 0
    episodeRewardTracker = []
    iterations = 0
    prevTime=time()
    # ---Main algorithm-----------------
    with tf.Session() as sess:
        #     with tf.device('/gpu:0'):
        sess.run(tf.global_variables_initializer())
        while True:
            episodeIterations += 1
            # ---check if target estimator needs to be updated
            if iterations == target_network_update_every:
                copy_model_parameters(sess, estimator, target_network)
                print("\nCopied model parameters to target network.")
                iterations = 0
            iterations += 1

            if len(replayMemory) > batch_size: # checking if agent is ready to be trained
                minibatch = random.sample(replayMemory, batch_size)
                s = np.asarray([m[0] for m in minibatch], dtype=float)
                selOutputs = np.asarray([m[1] for m in minibatch], dtype=float)
                r = np.asarray([m[2] for m in minibatch], dtype=float)
                sprime = np.asarray([m[3] for m in minibatch], dtype=float)
                nonTerminal = np.asarray([m[4] for m in minibatch], dtype=float)

                sprimeVals = target_network.getEstimatesFor(sess, sprime)
                # #---DQN implementation
                # sprimeVals=np.multiply( np.max(sprimeVals, axis=1), nonTerminal) # get valid next state values

                # ---Double DQN implementation
                ArgMaxSprimeAction = np.argmax(estimator.getEstimatesFor(sess, sprime), axis=1)
                sprimeVals = np.multiply(
                    sprimeVals[np.arange(sprimeVals.shape[0]), ArgMaxSprimeAction],
                    nonTerminal)

                targetVals = r + gamma * sprimeVals

                estimator.train(sess, targetVals, selOutputs, s)

                # ---For current state, get corresponding action values and select exporatory action
                currActionVals = estimator.getEstimatesFor(sess, currState)

                # ---Action Selection strategy
        #        action = agent.eGreedyAction(currActionVals)
        #        action = agent.gibbsAction(currActionVals)
                action = agent.epsilonDecay(currActionVals)  # best so far
                # action = agent.epsilonFirst(currActionVals)
            else: # Means agent is not yet trained. Take a random action.
                action=np.random.randint(nActions)

            # ---take next step. Get observations, reward.
            observation, reward, done, _ = env.step(action)

            env.render()
            episodeReward += reward

            # ---update state,as seen by agent, and get new state vector
            observation = (observation - meanObs) / stdDev
            agent.updateState(observation)
            newState = agent.getState()

            # ---create action vector based on most recent action selected
            actionVec = np.zeros(nActions, dtype=float);
            actionVec[action] = 1.0

            # ---remove oldest s,a,r,s' from replay memory and insert latest observations.
            replayMemory.appendleft((currState, actionVec, reward, newState, float(not done)))
            if len(replayMemory) > ReplayMemMaxSize: replayMemory.pop()

            currState = newState
            if done:
                rootLogger.info("%d,%f,%d,%f", int(agent.episodeN), time()-prevTime, episodeIterations, episodeReward)
                prevTime=time()
                currObs = env.reset()
                currObs = (currObs - meanObs) / stdDev
                agent.start_episode(currObs)
                currState = agent.getState()
                episodeRewardTracker.append(episodeReward)
                episodeReward = 0.0; episodeIterations = 0

            if agent.episodeN > nEpisodes:
                break
            # ------------------------------------
    env.close()
    resultMapFileName='./resultMap.txt'
    with open(resultMapFileName, 'a') as resultMap:
        print 'writing to result map file at ' + resultMapFileName
        resultMap.write('{0}\t {1}\n'.format(key, logfile))

    logFileHandler.close()
    rootLogger.removeHandler(logFileHandler)
