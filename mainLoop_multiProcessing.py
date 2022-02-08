import pygame
import random
import numpy as np
import fullTetris_Sarsa
import pickle
import pandas
import time
import concurrent.futures

#-------------------------------------------------------------------------------------------

EPOCHS=50
EPISODES=3000
ALPHA=0.1
GAMMA=0.9
INITIALEPSILON=0.01
DECAYEPSILON=1
INITIALWEIGHTS=1
REWARDSIGNAL=[10,-0.1, -2, 0, 1, 0, 2, 0.1]
STATEVECTORROWS=2
HEIGHTDIFBOUND=[-2,2]
MAXCOARSEHEIGHT=4
COARSERESOLUTION=2
MAXLINESCLEARED=100
parameterValues=[0]

#use run for epoch
#average over multiple runs
#plot every n data points
#+- show 100 data points
#correlation between hyperparameters
#time stamp each episode?? - plot per episode and say something about time
#Plot distribution of episode lengths - histogram
#Add more features - more training time, large weight vector can prohibit learning

#tell agent where there is an overhang - in state vector - 1 hot encode for each column
#only tell agent that there is an overhang, not where it is

#bottom row + overhang


comment = ""

iteratedParameter = "columnHeightsWithHoles"
if iteratedParameter != "": 
    fileName = iteratedParameter
else:
    fileName = ""

#--------------------------------------------------------------------------------------------

gameInstance = fullTetris_Sarsa.tetris()
agentInstance = fullTetris_Sarsa.trainAgent()

def helper(args):
    return trainEpoch(args[0], args[1])


def trainEpoch(x, epochNumber):

    a = fullTetris_Sarsa.trainAgent(episodes=EPISODES, alpha=ALPHA, 
    gamma=GAMMA, initialEpsilon=INITIALEPSILON, decayEpsilon=DECAYEPSILON, 
    initialWeights=INITIALWEIGHTS, rewardSignal=REWARDSIGNAL, stateVectorRows=STATEVECTORROWS,
    heightDifBound=HEIGHTDIFBOUND, coarseResolution=COARSERESOLUTION, 
    maxCoarseHeight=MAXCOARSEHEIGHT, maxLinesCleared=100,
    verbose=False, epoch=epochNumber, parameter=x)
    
    start=time.time()
    a.train()
    end=time.time()
    epochDuration = round(end-start, 2)
    string =f"Parameter = {x}"
    string1=f"Epoch {epochNumber} complete"
    string2=f"Time Taken = {epochDuration}"
    
    print(f"-------------------------------------------------------------")
    if x==0:
        print(f"{string1:<20}{string2:<20}")
    else:
        print(f"{string:<20}{string1:<20}{string2:<20}")
    print(f"-------------------------------------------------------------")

    outfile=open("agentWeights", "wb")
    pickle.dump(a.player.weights, outfile)
    outfile.close()

    return  a.linesCleared, a.episodeReward, epochDuration, a.player.featureVectorLength, a.episodeTimes #a.moves

def helper(args):
    return trainEpoch(args[0], args[1])

episodeReward = np.zeros([EPOCHS,EPISODES])
episodeLinesCleared = np.zeros([EPOCHS, EPISODES])
episodeTimes = np.zeros([EPOCHS, EPISODES])
epochDurations = np.zeros(EPOCHS)
moves = np.zeros([EPOCHS, EPISODES])

featureVectorLength = np.zeros(len(parameterValues))

episodeAverageReward = np.zeros(EPISODES)
episodeAverageLinesCleared = np.zeros(EPISODES)
episodeLineDeviation = np.zeros(EPISODES)
episode75Percentile = np.zeros(EPISODES)
episode25Percentile = np.zeros(EPISODES)

averageReward = []
averageLinesCleared = []
lineDeviation = []
line75Percentile = []
line25Percentile = []
totalTime = []
totalTimeDeviation = []
duration = []
aveMoves = []

totStart = time.time()
results=[]

args = []
for x in parameterValues:
    for y in range(EPOCHS):
        args.append((x,y))

with concurrent.futures.ProcessPoolExecutor() as executor:
    for result in executor.map(helper, args):
        results.append(result)

for i in range(len(parameterValues)):
    for j in range(EPOCHS):
        episodeLinesCleared[j,:] = results[i*EPOCHS+j][0]
        episodeReward[j,:] = results[i*EPOCHS+j][1]
        episodeTimes[j,:] = results[i*EPOCHS+j][4]
        #moves[j,:] = results[i*EPOCHS+j][5]
        epochDurations[j] = results[i*EPOCHS+j][2]
        
    featureVectorLength[i] = results[i*EPOCHS+j][3]

    episodeAverageReward = np.average(episodeReward, axis=0)
    episodeAverageLinesCleared = np.average(episodeLinesCleared, axis=0)
    episodeAverageTime = np.average(episodeTimes, axis=0)
    episodeLineDeviation = np.std(episodeLinesCleared, axis=0)
    episode75Percentile = np.percentile(episodeLinesCleared, 75, axis=0)
    episode25Percentile = np.percentile(episodeLinesCleared, 25, axis=0)
    episodeTimeDeviation = np.std(episodeTimes, axis=0)


    averageReward.append([episodeAverageReward])
    averageLinesCleared.append([episodeAverageLinesCleared])
    lineDeviation.append([episodeLineDeviation])
    line75Percentile.append([episode75Percentile])
    line25Percentile.append([episode25Percentile])
    duration.append([np.average(epochDurations)])
    totalTime.append([episodeAverageTime])
    totalTimeDeviation.append([episodeTimeDeviation])
    #aveMovesBeforeDrop.append([np.average(movesBeforeDrop, axis=0)])
    #aveMovesPerLine.append([np.average(movesPerLine, axis=0)])
    #aveMoves.append([np.average(moves, axis=0)])

    episodeReward = np.zeros([EPOCHS,EPISODES])
    episodeLinesCleared = np.zeros([EPOCHS,EPISODES])
    episodeAverageReward = np.zeros(EPISODES)
    episodeAverageLinesCleared = np.zeros(EPISODES)
    epochDurations = np.zeros(EPOCHS)
    episodeTimes = np.zeros([EPOCHS, EPISODES])
    #movesPerLine = np.zeros([EPOCHS, EPISODES])
    episode75Percentile = np.zeros(EPISODES)
    episode25Percentile = np.zeros(EPISODES)


#Output file

outfile=open(fileName, "wb")
pickle.dump(iteratedParameter, outfile)
pickle.dump(parameterValues, outfile)
pickle.dump(averageReward, outfile)
pickle.dump(averageLinesCleared, outfile)
pickle.dump(EPOCHS, outfile)
pickle.dump(EPISODES, outfile)
pickle.dump(ALPHA, outfile)
pickle.dump(GAMMA, outfile)
pickle.dump(INITIALEPSILON, outfile)
pickle.dump(DECAYEPSILON, outfile)
pickle.dump(gameInstance.width, outfile)
pickle.dump(gameInstance.height, outfile)
pickle.dump(REWARDSIGNAL, outfile)
pickle.dump(gameInstance.block.names, outfile)
pickle.dump(INITIALWEIGHTS, outfile)
pickle.dump(lineDeviation, outfile)
pickle.dump(agentInstance.maxLinesCleared, outfile)
pickle.dump(comment, outfile)
pickle.dump(duration, outfile)
pickle.dump(HEIGHTDIFBOUND, outfile)
pickle.dump(featureVectorLength, outfile)
pickle.dump(totalTime, outfile)
pickle.dump(totalTimeDeviation, outfile)
#pickle.dump(aveMoves, outfile)
pickle.dump(line75Percentile, outfile)
pickle.dump(line25Percentile, outfile)
outfile.close()

totEnd = time.time()
print(f"Total time taken: {round(totEnd-totStart, 2)} seconds")