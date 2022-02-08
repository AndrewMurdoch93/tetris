import pickle
import numpy as np
import operator
import matplotlib.pyplot as plt
import itertools

class plotTools:
    
    def getParameters(self, fileName):
        infile = open(fileName,'rb')
        self.iteratedParameter = pickle.load(infile)
        self.parameterValues = pickle.load(infile)
        self.averageReward = pickle.load(infile)
        self.averageLinesCleared = pickle.load(infile)
        self.epochs = pickle.load(infile)
        self.episodes = pickle.load(infile)
        self.alpha = pickle.load(infile)
        self.gamma = pickle.load(infile)
        self.initialEpsilon = pickle.load(infile)
        self.decayEpsilon = pickle.load(infile)
        self.width = pickle.load(infile)
        self.height = pickle.load(infile)
        self.rewardSignal = pickle.load(infile)
        self.tetrominos = pickle.load(infile)
        self.initialWeights = pickle.load(infile)
        self.lineDeviation = pickle.load(infile)
        self.maxLinesCleared = pickle.load(infile)
        self.comment = pickle.load(infile)
        self.epochDuration = pickle.load(infile)
        self.heightDifBound = pickle.load(infile)
        self.featureVectorLength = pickle.load(infile)
        self.totalTime = pickle.load(infile)
        self.totalTimeDeviation = pickle.load(infile)
        #self.aveMoves = pickle.load(infile)

        infile.close()

    def displayParameters(self, fileName):
        self.getParameters(fileName)
        print(f"iteratedParameter = {self.iteratedParameter}")
        print(f"parameterValues = {self.parameterValues}")
        print(f"epochs = {self.epochs}")
        print(f"episodes = {self.episodes}")
        print(f"alpha = {self.alpha}")
        print(f"gamma = {self.gamma}")
        print(f"initialEpsilon = {self.initialEpsilon}")
        print(f"decayEpsilon = {self.decayEpsilon}")
        print(f"width = {self.width}")
        print(f"height = {self.height}")
        print(f"rewardSignal = {self.rewardSignal}")
        print(f"tetrominos = {self.tetrominos}")
        print(f"initialWeights = {self.initialWeights}")
        print(f"maxLinesCleared = {self.maxLinesCleared}")
        print(f"comment = {self.comment}")
        print(f"heightDifBound = {self.heightDifBound}")
        print(f"featureVectorLength = {self.featureVectorLength}")

    def plotParameters(self, fileName):
        points=100
        self.getParameters(fileName)
        legendEntry=[]
        for i in self.parameterValues:
            legendEntry.append([self.iteratedParameter + " = " + str(i)])
            #legendEntry.append([str(i)])
            #legendEntry.append(["n" + " = " + str(i)])
            #legendEntry.append(["line clear reward = " + str(i[0]) + ", step reward = " + str(i[1])])
        legendEntries=list(itertools.chain(*legendEntry))
        #legendEntries=["Game over penalty = -0.5", "Game over penalty = -1", "Game over penalty = -2"]
        #legendEntries = ["Height penalty = 0.01", "Height penalty = 0.1", "Height penalty = 0.5", "Height penalty = 1"]
        #legendEntries = ["Hole penalty = 0.1", "Hole penalty = 0.5", "Hole penalty = 1", "Hole penalty = 2"]
        #legendEntries = ["Unevenness penalty = 0.1", "Unevenness  = 0.5", "Unevenness  = 1"]
        #legendEntries = ["Lines cleared multiplier = 1", "Lines cleared multiplier = 5", "Lines cleared multiplier = 10"]
        #legendEntries = ["n = 3", "n = 5", "n = 10", "n = 15"]
        
        for i in range(len(self.parameterValues)):
            plt.plot( np.linspace(0,self.episodes,points) , np.take(self.averageLinesCleared[i][0], np.floor(np.linspace(0,self.episodes-1,points)).tolist()) )
        plt.xlabel("Number of episodes")
        plt.ylabel("Average Lines Cleared")
        #plt.ylim([0, 50])
        #plt.xlim(0,600)
        plt.title("Learning curves")
        plt.legend(legendEntries, loc="lower right")
        plt.show()

    def compareAgents(self, fileNames):
        points=100
        for i in fileNames:
            self.getParameters(i)

            for i in range(len(self.parameterValues)):
                plt.plot( np.linspace(0,self.episodes,points) , np.take(self.averageLinesCleared[i][0], np.floor(np.linspace(0,self.episodes-1,points)).tolist()) )
        
        #legendEntries=[""]
        #plt.xlim([0,500])
        plt.xlabel("Number of episodes")
        plt.ylabel("Average Lines Cleared")
        plt.title("Comparison of agents")
        plt.legend(legendEntries, loc = "lower right")
        plt.show()

    def agentConfidenceBound(self, fileName, i):
        points=100
        self.getParameters(fileName)

        averageLinesCleared = np.take(self.averageLinesCleared[i][0], np.floor(np.linspace(0,self.episodes-1,points)).tolist())
        #lineDeviation = np.take(self.lineDeviation[i][0], np.floor(np.linspace(0,self.episodes-1,points)).tolist())
        upperPercentiles = np.take(self.line75Percentile[i][0], np.floor(np.linspace(0,self.episodes-1,points)).tolist())
        lowerPercentiles = np.take(self.line25Percentile[i][0], np.floor(np.linspace(0,self.episodes-1,points)).tolist())

        episodes = np.linspace(0,self.episodes,points)

        #upperBound = averageLinesCleared+lineDeviation
        #lowerBound = averageLinesCleared-lineDeviation

        plt.plot(episodes, averageLinesCleared, color="black")
        #plt.plot(range(self.episodes), upperBound)
        #plt.plot(range(self.episodes), lowerBound)
        plt.fill_between(episodes, upperPercentiles, lowerPercentiles, color="lightBlue")
        #plt.hlines(0, 0, self.episodes)
        plt.hlines(self.maxLinesCleared, 0, self.episodes, linestyles="dashed")

        #plt.xlim([0,500])
        #plt.ylim([])
        plt.xlabel("Number of episodes")
        plt.ylabel("Average Lines Cleared")
        plt.title("Learning Curve")
        plt.legend(["Average", "Standard deviation", "Lines cleared limit"], loc="lower right")
        plt.show()

    def plotReward(self, fileName):
        self.getParameters(fileName)
        legendEntry=[]
        for i in self.parameterValues:
            legendEntry.append([self.iteratedParameter + " = " + str(i)])
            #legendEntry.append(["line clear reward = " + str(i[0]) + ", step reward = " + str(i[1])])
        legendEntries=list(itertools.chain(*legendEntry))

        for i in range(len(self.parameterValues)):
            plt.plot(range(self.episodes), self.averageReward[i][0])
        plt.xlabel("Number of episodes")
        plt.ylabel("Average Lines Cleared")
        #plt.ylim([0, 50])
        #plt.xlim(0,600)
        plt.title("Learning Curve")
        plt.legend(legendEntries)
        plt.show()
    
    def plotTime(self, fileName):
        self.getParameters(fileName)
        legendEntry=[]
        for i in self.parameterValues:
            legendEntry.append([self.iteratedParameter + " = " + str(i)])
            #legendEntry.append(["line clear reward = " + str(i[0]) + ", step reward = " + str(i[1])])
        legendEntries=list(itertools.chain(*legendEntry))

        for i in range(len(self.parameterValues)):
            plt.plot(range(self.episodes), self.totalTime[i][0])
            print(f"{self.iteratedParameter} = {i} = {a.totalTime[i][0][-1]}")
        
        plt.xlabel("Number of episodes")
        plt.ylabel("Cumulative Time Taken")
        #plt.ylim([0, 50])
        #plt.xlim(0,600)
        plt.title("Total Time Taken")
        plt.legend(legendEntries)
        plt.show()
        
    
    def plotMoves(self, fileName):
        self.getParameters(fileName)
        legendEntry=[]
        for i in self.parameterValues:
            legendEntry.append([self.iteratedParameter + " = " + str(i)])
            #legendEntry.append(["line clear reward = " + str(i[0]) + ", step reward = " + str(i[1])])
        legendEntries=list(itertools.chain(*legendEntry))

        for i in range(len(self.parameterValues)):
            plt.plot(range(self.episodes), self.averageLinesCleared[i][0]/self.aveMoves[i][0])
        
        plt.xlabel("Number of episodes")
        plt.ylabel("Lines per move")
        #plt.ylim([0, 50])
        #plt.xlim(0,600)
        plt.title("Actions taken per per Episode")
        plt.legend(legendEntries)
        plt.show()




#f = "stateVectorBottomRows"
#f = "stateVectorBottomRowHoles"
#f = "stateVectorRows"
#f = "stateVectorHoles"
#f = "heightDif"
#f = "heightDifHoles"
#f = "heightDifContReward"
#f = "heightDifHolesContReward"
#f = "severeRewardFunction"
#f = "rewardSignalValues"
#f = "2rowComplexTetris"
#f = "Qlearn"
#f = "alpha_Qlearn"
#f = "gamma_Qlearn"
#f = "epsilon_Qlearn"
#f = "columnCoarseEncode"
#f=["epsilon_Qlearn", "gamma_Qlearn"]

#f = "FinalResults/rewardLinesCleared"
#f = "FinalResults/rewardGameOverPenalty"
#f = "FinalResults/rewardGameOverPenalty1"
#f = "FinalResults/rewardGameIncreaseHeight"
#f = "FinalResults/penaliseColumnHoles"
#f = "FinalResults/penaliseHoles"
#f = "FinalResults/penaliseUneveness"
#f = ["FinalResults/rewardGameOverPenalty","FinalResults/rewardGameOverPenalty1"]

#f = "gamma"
#f = "finalRewardSignal"
#f = "nStepSarsa"
#f = "FinalResults/increaseLinesClearedReward"
#f = "FinalResults/alpha"
#f = "FinalResults/epsilon"
#f = "FinalResults/nStepSarsa"
#f = "FinalResults/initialWeights"
#f = "FinalResults/Qlearn"

a = plotTools()
#a.displayParameters(fileName=f)
#a.plotReward(fileName=f)
#a.plotParameters(fileName=f)
#a.compareAgents(f)
#a.agentConfidenceBound(fileName=f, i=1)
a.plotTime(fileName=f)
#a.plotMoves(fileName=f)


points=100

'''
#Reward signal 1
a.getParameters("FinalResults/rewardLinesCleared")
plt.plot( np.linspace(0,a.episodes,points) , np.take(a.averageLinesCleared[0][0], np.floor(np.linspace(0,a.episodes-1,points)).tolist()) )
a.getParameters("FinalResults/rewardGameOverPenalty")
plt.plot( np.linspace(0,a.episodes,points) , np.take(a.averageLinesCleared[2][0], np.floor(np.linspace(0,a.episodes-1,points)).tolist()) )
a.getParameters("FinalResults/rewardGameIncreaseHeight")
plt.plot( np.linspace(0,a.episodes,points) , np.take(a.averageLinesCleared[3][0], np.floor(np.linspace(0,a.episodes-1,points)).tolist()) )
a.getParameters("FinalResults/penaliseHoles")
plt.plot( np.linspace(0,a.episodes,points) , np.take(a.averageLinesCleared[3][0], np.floor(np.linspace(0,a.episodes-1,points)).tolist()) )
a.getParameters("FinalResults/penaliseUneveness")
plt.plot( np.linspace(0,a.episodes,points) , np.take(a.averageLinesCleared[1][0], np.floor(np.linspace(0,a.episodes-1,points)).tolist()) )
legendEntries=["Original", "Game over penalty", "Penalise height increase", "Penalise Holes", "Penalise unevennes"]
plt.xlabel("Number of episodes")
plt.ylabel("Average Lines Cleared")
#plt.ylim([0, 50])
#plt.xlim(0,600)
plt.title("Learning Curve")
plt.legend(legendEntries)
plt.show()
'''
'''
#Reward Signal 2
a.getParameters("FinalResults/increaseLinesClearedReward")
plt.plot( np.linspace(0,a.episodes,points) , np.take(a.averageLinesCleared[0][0], np.floor(np.linspace(0,a.episodes-1,points)).tolist()) )
plt.plot( np.linspace(0,a.episodes,points) , np.take(a.averageLinesCleared[1][0], np.floor(np.linspace(0,a.episodes-1,points)).tolist()) )
plt.plot( np.linspace(0,a.episodes,points) , np.take(a.averageLinesCleared[2][0], np.floor(np.linspace(0,a.episodes-1,points)).tolist()) )
#a.getParameters("FinalResults/gamma")
#plt.plot( np.linspace(0,a.episodes,points) , np.take(a.averageLinesCleared[1][0], np.floor(np.linspace(0,a.episodes-1,points)).tolist()) )
plt.xlabel("Number of episodes")
plt.ylabel("Average Lines Cleared")
#plt.ylim([0, 50])
#plt.xlim(0,600)
plt.title("Learning Curve")
legendEntries = ["line cleared reward = 1", "ine cleared reward = 5", "line cleared reward = 10"]
plt.legend(legendEntries)
plt.show()
'''
'''
#Learning methods
a.getParameters("FinalResults/Qlearn")
plt.plot( np.linspace(0,a.episodes,points) , np.take(a.averageLinesCleared[0][0], np.floor(np.linspace(0,a.episodes-1,points)).tolist()) )
a.getParameters("FinalResults/gamma")
plt.plot( np.linspace(0,a.episodes,points) , np.take(a.averageLinesCleared[1][0], np.floor(np.linspace(0,a.episodes-1,points)).tolist()) )
a.getParameters("FinalResults/nStepSarsa")
plt.plot( np.linspace(0,a.episodes,points) , np.take(a.averageLinesCleared[0][0], np.floor(np.linspace(0,a.episodes-1,points)).tolist()) )
plt.xlabel("Number of episodes")
plt.ylabel("Average Lines Cleared")
#plt.ylim([0, 50])
plt.xlim(0,3000)
plt.title("Learning Curve")
legendEntries = ["Q-learning", "Sarsa", "n-step Sarsa"]
plt.legend(legendEntries)
plt.show()
'''

'''
#Feature vector representation
a.getParameters("FinalResults/rowsMaxColumnHeight")
plt.plot( np.linspace(0,a.episodes,points) , np.take(a.averageLinesCleared[0][0], np.floor(np.linspace(0,a.episodes-1,points)).tolist()) )
a.getParameters("FinalResults/columnHeightFeature")
plt.plot( np.linspace(0,a.episodes,points) , np.take(a.averageLinesCleared[0][0], np.floor(np.linspace(0,a.episodes-1,points)).tolist()) )
a.getParameters("FinalResults/columnHeightHolesFeature")
plt.plot( np.linspace(0,a.episodes,points) , np.take(a.averageLinesCleared[0][0], np.floor(np.linspace(0,a.episodes-1,points)).tolist()) )
a.getParameters("FinalResults/gamma")
plt.plot( np.linspace(0,a.episodes,points) , np.take(a.averageLinesCleared[1][0], np.floor(np.linspace(0,a.episodes-1,points)).tolist()) )
plt.xlabel("Number of episodes")
plt.ylabel("Average Lines Cleared")
#plt.ylim([0, 50])
plt.xlim(0,3000)
plt.title("Learning Curve")
legendEntries = ["Matrix represntation", "Column heights", "Column heights with holes", "Height differences"]
plt.legend(legendEntries)
plt.show()
'''

#fileName = "sarsaHistogram"
#infile = open(fileName,'rb')
#lines = pickle.load(infile) 
#infile.close()

#print(np.average(lines))
#rint(np.std(lines))

#density = gaussian_kde(lines)

#plt.hist(lines,bins=20,density=True)
#plt.xlabel("Lines cleared per episode")
#plt.ylabel("number of agents")
#plt.title("Distribution of agent scores")
#plt.show()