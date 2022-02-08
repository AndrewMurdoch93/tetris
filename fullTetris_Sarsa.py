import pygame
import random
import numpy as np
import sys
import time
import pickle

#Expand agent by changing feature vector
#Feature vector now includes wether a row has an overhang

O = np.array([[(0,0),(0,1),(1,0), (1,1)]])
I = np.array([[(1,0), (1,1), (1,2), (1,3)], [(0,2), (1,2), (2,2), (3,2)]])
S = np.array([[(0,1), (0,2), (1,0), (1,1)], [(0,0), (1,0), (1,1), (2,1)]])
Z = np.array([[(1,0), (1,1), (2,1), (2,2)], [(0,1), (1,0), (1,1), (2,0)]])
L = np.array([[(1,0), (1,1), (1,2), (2,0)], [(0,0), (1,0), (2,0), (2,1)], [(1,0), (1,1), (1,2), (0,2)], [(0,0), (0,1), (1,1), (2,1)]])
J = np.array([[(1,0), (1,1), (1,2), (2,2)], [(0,0), (0,1), (1,0), (2,0)], [(0,0), (1,0), (1,1), (1,2)], [(0,1), (1,1), (2,1), (2,0)]])
T = np.array([[(1,0), (1,1), (1,2), (2,1)], [(0,0), (1,0), (1,1), (2,0)], [(1,0), (1,1), (1,2), (0,1)], [(0,1), (1,0), (1,1), (2,1)]])

o=np.array([[(0,0)]])
O=np.array([[(0,0),(0,1),(1,0), (1,1)]])
i=np.array([[(0,0), (1,0)], [(0,0), (0,1)]])
l=np.array([[(0,0), (1,0), (1,1)], [(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,1)], [(0,0),(0,1),(1,0)]])
s=np.array([[(0,0),(1,1)],[(1,0),(0,1)]])

class tetromino:
    def __init__(self, x=4, y=0):
        self.x = x  #Position of tetromino block (defualt spawn position is x=3, y=0)
        self.y = y
        #self.types = ["O", "I", "S", "Z", "L", "J", "T"]
        self.tetrominos = [O, I, L, J, T, S, Z]
        #self.tetrominos = [o,O,i,l]
        self.rotation = 0
        self.index = random.choice(range(len(self.tetrominos))) #Randomly select tetromino to spawn
        #self.index=1 #Tetris easy mode
        #self.name = self.types[self.index]  #Name of spawned tetromino block
        self.shape = self.tetrominos[self.index][self.rotation] #Tetrmonio field positions
        self.names = ["o", "O", "i","l"]

    def rotate(self, direction):
        if direction=="clockwise":
            self.rotation = (self.rotation + 1) % len(self.tetrominos[self.index])
        elif direction=="antiClockwise":
            self.rotation = (self.rotation - 1) % len(self.tetrominos[self.index])
        
        self.shape = self.tetrominos[self.index][self.rotation]

class tetris:
    def __init__(self, height=20, width=10, rewardSignal=[10, -0.1, -10, 0.2, 0, 0, 0.5, 0.1], coarseResolution=2,maxCoarseHeight=4):
        
        self.height=height  # Dimensions of the game field
        self.width=width
        self.block=tetromino()  # Spawns a tetromino in default position
        self.field=np.zeros((self.height,self.width)) # Initialise empty game field
        self.score=0
        self.gameOver = False
        self.linesCleared=0
        self.droppedBlocks=0
        self.moves=0
        self.columnHeights=np.zeros(self.width, dtype=int)
        self.columnHoles=np.zeros(self.width, dtype=bool)
        self.reward=0
        self.rewardSignal=rewardSignal
        self.heightDif=np.zeros(self.width-1, dtype=int)
        self.totHoles=0
        self.totColHoles=0
        self.aveHeight=0
        self.maxHeight=0
        self.unevenness=0
        self.coarseResolution=coarseResolution
        self.maxCoarseHeight=maxCoarseHeight
        self.coarseColumnHeight=np.zeros(self.width, dtype=float)

    def intersection(self):
        # Checks to see if a move is valid
        # Valid moves do not intersect occupied spaces in the field or move the tetromino beyond the field
        # intersection is called to freeze the tetromino in place
        y = self.block.y+self.block.shape[:,0]
        x = self.block.x+self.block.shape[:,1]

        if np.any(y>=self.height) or np.any(y<0):
            return True
        if np.any(x>=self.width) or np.any(x<0):
            return True
        if np.any(self.field[y,x]==1):
            return True
        return False

    def clearLines(self):
        # If a line is completely filled, it disappears (is cleared)
        # The lines above the cleared line drop down 1 y value
        # Score is proportional to the square of the number of lines cleared at onve
    
        lines=0
        linesToClear = np.where(np.all(self.field==np.ones(self.width), axis=1))[0]

        for index in linesToClear:
                lines+=1
                self.linesCleared+=1
                #print("Line cleared")
                self.field[np.arange(index,0,-1)]=self.field[np.arange(index,0,-1)-1]
                self.field[0]=0
        self.score+=(lines**2)    # Add to score counter
        
        return lines

    def takeAction(self, command):
        # Makes a change to the game field based on given command (either from player or agent)
        # Checks to see if move is valid before making change

        self.moves+=1
        if command=="right":
            self.block.x+=1
            if self.intersection():
                self.block.x-=1
            self.giveReward(lines=0)
        
        if command=="left":
            self.block.x-=1
            if self.intersection():
                self.block.x+=1
            self.giveReward(lines=0)
                
        if command=="rotate":
            self.block.rotate(direction="clockwise")
            if self.intersection():
                self.block.rotate(direction="antiClockwise")
            self.giveReward(lines=0)
       
        if command=="drop":
            self.droppedBlocks=0
            self.totHoles,self.totColHoles,self.aveHeight,self.maxHeight,self.unevenness=self.getRewardFeatures()
            while not self.intersection():
                self.block.y += 1
            self.block.y -= 1
            self.freeze()

        if command=="down":
            self.block.y += 1
            if self.intersection():
                self.block.y -= 1
                self.freeze()
    
    def freeze(self):
    # Freezes tetrominos in place once they fall onto an occupied field space

        y = self.block.y+self.block.shape[:,0]
        x = self.block.x+self.block.shape[:,1]
        self.field[y,x]=1
        #if self.intersection():
        #    self.gameOver = True
        lines = self.clearLines()
        self.block=tetromino()

        heights=np.argmax(self.field,axis=0)
        heights[heights==0]=self.height
        heights=self.height-heights
        self.columnHeights=heights

        #Holes in each column
        firstHole=np.argmin(np.flip(self.field,axis=0), axis=0)
        self.columnHoles=np.any([firstHole<heights], axis=0)    
        #print(self.columnHoles)

        #Column height differences
        self.heightDif=np.subtract(self.columnHeights[1:len(self.columnHeights)], self.columnHeights[0:-1])
    
        #coarseEncodeColumns
        self.coarseColumnHeight = np.floor(np.divide(heights, self.coarseResolution)+1)      
        self.coarseColumnHeight[heights==0]=0
        self.coarseColumnHeight = np.clip(self.coarseColumnHeight, 0, self.maxCoarseHeight)
        
        if np.any(self.columnHeights>=self.height-2):
            self.gameOver = True

        self.giveRewardDrop(lines)
    
    def getRewardFeatures(self):
        holes=np.zeros(self.width)
        #print(f"columnHeights: {self.columnHeights}")
        for i in range(self.width):
            if self.columnHeights[i] != 0:
                index = int(self.height-self.columnHeights[i])
                holes[i] = np.sum(self.field[index:self.height,i]==0)
            else:
                holes[i] = np.array(0)
        #print(holes)
        holes= np.sum([holes])
        colHoles = np.sum(self.columnHoles)
        aveHeight = np.average(self.columnHeights)
        maxHeight = np.amax(self.columnHeights)
        unevenness = np.sum(np.square(self.heightDif))
        #print(unevenness)
        return holes, colHoles, aveHeight, maxHeight, unevenness
        
    
    def giveRewardDrop(self, lines):
        holes,colHoles,aveHeight,maxHeight,unevenness = self.getRewardFeatures()

        if lines == 0:
            self.reward=0
        if lines == 1:
            self.reward=1
        if lines == 2:
            self.reward=3
        if lines == 3:
            self.reward=5
        if lines >= 4:
            self.reward=8

        self.reward *= self.rewardSignal[0]
        if not lines:
            self.reward += self.rewardSignal[1]
            self.reward += self.rewardSignal[3]*(self.aveHeight-aveHeight)
            self.reward += self.rewardSignal[4]*(self.maxHeight-maxHeight)
            self.reward += self.rewardSignal[5]*(self.totColHoles-colHoles) 
            self.reward += self.rewardSignal[6]*(self.totHoles-holes) 
            self.reward += self.rewardSignal[7]*(self.unevenness-unevenness) #Quadratic unevenness

        #print(f"Reward signal: {self.rewardSignal[7]}")
        #print(f"before: {self.unevenness}")
        #print(f"after: {unevenness}")
        #print(f"lines cleared: {self.rewardSignal[0]}   aveHeight: {self.rewardSignal[3]*(self.aveHeight-aveHeight)}, totHoles: {self.rewardSignal[6]*(self.totHoles-holes)}, uneveness: {self.rewardSignal[7]*(self.unevenness-unevenness)}")

        if self.gameOver==True:
            self.reward=self.rewardSignal[2]
        
        #print(self.reward)

    def giveReward(self, lines):
        self.reward = self.rewardSignal[1]
        if self.gameOver==True:
            self.reward=self.rewardSignal[2]
     

class playGame:
    # A tetris implementation that is playable by humans 

    def __init__(self, selfPlay, player):
        self.game = tetris()
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.gray = (128, 128, 128)
        self.fps = 10
        self.counter = 0
        self.size = (400, 500)
        self.done = False
        self.zoom = 20
        self.x=100
        self.y=60
        self.level=4
        self.selfPlay=selfPlay
        self.player=player
    
    def chooseAction(self):
        # Interprets user input to select an action
        # OR allows the agent to select an action, if agent==True
        # Does not perform selected action 
        
        action=""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.game=tetris()
                if self.selfPlay==False:
                    if event.key == pygame.K_UP:
                        action="rotate"
                    if event.key == pygame.K_LEFT:
                        action="left"
                    if event.key == pygame.K_RIGHT:
                        action="right"
                    if event.key == pygame.K_SPACE:
                        action="drop"
    
        if self.selfPlay==True:
            
            act = self.player.chooseAction(self.game)
            action = self.player.actionNames[act]

        return action

    def displayScreen(self, screen):
        # Uses game state to output images on the screen

        screen.fill(self.black) #Display background
        
        # Display field
        for i in range(self.game.height):
            for j in range(self.game.width):
                pygame.draw.rect(screen, self.gray, [self.x+self.zoom*j, self.y+self.zoom*i, self.zoom, self.zoom], 1)
                if self.game.field[i][j] == 1:
                    pygame.draw.rect(screen, self.white, [self.x + self.zoom*j+1, self.y+self.zoom*i+1, self.zoom-2, self.zoom-1])
        
        # Display current tetromino block
        for i in self.game.block.shape:
                pygame.draw.rect(screen, self.white, [self.x+self.zoom*(i[1]+self.game.block.x)+1, self.y+self.zoom*(i[0]+self.game.block.y)+1, self.zoom-2, self.zoom-2])
        
        # Display text
        font = pygame.font.SysFont('Calibri', 25, True, False)
        font1 = pygame.font.SysFont('Calibri', 65, True, False)
        text = font.render("Score: " + str(self.game.score), True, self.white)
        text_game_over = font1.render("Game Over", True, (255, 125, 0))
        text_game_over1 = font1.render("Press ESC", True, (255, 215, 0))
        screen.blit(text, [0, 0])
        if self.game.gameOver == True:
            screen.blit(text_game_over, [20, 200])
            screen.blit(text_game_over1, [25, 265])
        
        # Display output
        pygame.display.flip()

    def play(self):
        # Game logic function - handles timing, when to take user inputs, etc

        pygame.init()
        screen = pygame.display.set_mode(self.size)
        clock = pygame.time.Clock()
        movesBeforeDrop=0

        while self.done==False:
            
            # Counter to move tetromino down 1 vertical position
            self.counter += 1
            if self.counter > 1000:
                self.counter = 0
            if self.counter % (self.fps//self.level) == 0: # and :self.game.state="
                self.game.takeAction("down")
            
            action = self.chooseAction()
            
            if self.selfPlay==True:
                if action!="drop":
                    movesBeforeDrop+=1
                if movesBeforeDrop>=self.game.width:
                    action="drop"
                if action=="drop":
                    movesBeforeDrop=0

            if self.game.gameOver==False: #Player can no longer take actions when state=="gameOver"
                self.game.takeAction(action)
                #print(self.game.block.x)
            
            self.displayScreen(screen)
            clock.tick(self.fps)    #Limit game to specified fps
        
        pygame.quit()


class trainAgent:
    # An implementation of tetris that isnt playable by humans
    # Intended use is to train agents

    def __init__(self, episodes=50, alpha=0.3, gamma=0.9, 
    initialEpsilon=0.1, decayEpsilon=1, initialWeights=1, 
    rewardSignal=[10, -0.1, -10, 0.2, 0.2, 0.5, 0.5, 0.1], 
    stateVectorRows=2, heightDifBound=[-2,2], coarseResolution=2, testEpisodes=100,
    maxCoarseHeight=4, maxLinesCleared=100, verbose=True, epoch=0, parameter=0):
        
        self.rewardSignal = rewardSignal
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.initialEpsilon = initialEpsilon
        self.initialWeights = initialWeights
        self.decayEpsilon = decayEpsilon
        self.rewardSignal = rewardSignal
        self.stateVectorRows=stateVectorRows
        self.heightDifBound = heightDifBound
        self.coarseResolution = coarseResolution
        self.maxCoarseHeight = maxCoarseHeight
        self.maxLinesCleared = maxLinesCleared
        self.episodeTimes = np.zeros(self.episodes)
        self.cumTime = np.zeros(self.episodes)
        self.moves= np.zeros(self.episodes)
        self.testEpisodes = testEpisodes

        self.game = tetris(rewardSignal=self.rewardSignal, coarseResolution=self.coarseResolution, 
        maxCoarseHeight=self.maxCoarseHeight)

        self.player = agent(width=self.game.width, height=self.game.height, initialEpsilon=self.initialEpsilon, 
        decayEpsilon=self.decayEpsilon, initialWeights=self.initialWeights, stateVectorRows=self.stateVectorRows, 
        heightDifBound=self.heightDifBound, coarseResolution=self.coarseResolution, maxCoarseHeight=self.maxCoarseHeight)

        self.numTetrominos = len(self.game.block.tetrominos)
        self.episodeReward = np.zeros(episodes)
        self.linesCleared = np.zeros(episodes)
        self.verbose=verbose
        self.epoch=epoch
        self.parameter=parameter
        self.testLinesCleared = np.zeros(testEpisodes)
    
    def train(self):
        #Controls game logic
        
        #Initialise state and actions
        action = self.player.chooseAction(self.game)
        state = self.player.getFeatureVector(action, self.game)

        movesBeforeDrop=0
        drops=0
        i=0
        start=time.time()

        while i<self.episodes and (time.time()-start)<3600: #and np.amax(self.episodeTimes)<661:
        

            self.game.takeAction(self.player.actionNames[action])
            reward=self.game.reward
            self.episodeReward[i] += reward
            
            #Meant to observe next state here - but dont have state action pair yet

            if(self.game.gameOver == True) or (time.time()-start)>3600: #(self.game.linesCleared>=self.maxLinesCleared):
                end=time.time()
                #(time.time()-start)>900:
                self.episodeTimes[i] = end-start
                #print( self.episodeTimes[i])
                self.linesCleared[i] = self.game.linesCleared
                self.moves[i] = self.game.moves
                if self.verbose==True:
                    if i%1==0:       
                        string1 = f"Parameter = {self.parameter}"
                        string2 = f"Epoch = {self.epoch}"
                        string3 = f"Episode = {i}. "
                        string4 = f"Lines Cleared: {self.linesCleared[i]}"
                        print(f"{string1:<20}{string2:<12}{string3:<17}{string4:<12}")
                        #timeToGo = 661-np.amax((self.episodeTimes))
                        #print(f"timeToGo:{timeToGo}")
            
                self.player.weights[state]+=self.alpha*(reward - self.player.getApproxQvalue(state))
                i+=1
                
                self.player.decreaseEpsilon()

                self.game = tetris(rewardSignal=self.rewardSignal, maxCoarseHeight=self.maxCoarseHeight, coarseResolution=self.coarseResolution)
                action=self.player.chooseAction(self.game)
                state=self.player.getFeatureVector(action, self.game)

            else:    
                nextAction = self.player.chooseAction(self.game)
                nextState = self.player.getFeatureVector(nextAction, self.game)
                
                #Iterate w
                #self.player.weights[state] += self.alpha*(reward + self.gamma*self.player.getApproxQvalue(self.player.greedyAction(self.game)) - self.player.getApproxQvalue(state))
                self.player.weights[state] += self.alpha*(reward + self.gamma*self.player.getApproxQvalue(nextState) - self.player.getApproxQvalue(state))

                state=nextState
                action=nextAction
        #print(self.cumTime[-1])

    def testAgent(self):
        
        start=time.time()
        for i in range(self.testEpisodes):
            self.game = tetris() 
            movesBeforeDrop=0
            
            while self.game.gameOver==False and (time.time()-start)<3600:
                
                action = self.player.chooseAction(self.game)
                if action!=3:
                    movesBeforeDrop+=1
                if movesBeforeDrop>=2*self.game.width:
                    action=3
                if action==3:
                    movesBeforeDrop=0

                self.game.takeAction(self.player.actionNames[action])
            
            self.testLinesCleared[i]=self.game.linesCleared

            if self.verbose==True:
                if i%1==0:       
                    string1 = f"Episode = {i}. "
                    string2 = f"Lines Cleared: {self.testLinesCleared[i]}"
                    print(f"{string1:<20}{string2:<12}")
        
        #ave = np.true_divide(self.testLinesCleared)
        #stdDev = np.std(self.testLinesCleared)

        #print(f"Average Lines Cleared = {ave}")
        #print(f"Standard deviation = {stdDev}")

        outfile=open("final_Sarsa_6_results", "wb")
        pickle.dump(self.testLinesCleared, outfile)
        outfile.close()

        
class agent:
    def __init__(self, width, height, initialEpsilon, decayEpsilon, initialWeights, stateVectorRows, heightDifBound, coarseResolution, maxCoarseHeight):
        self.actionNames = ["left", "right", "rotate","drop"]
        self.actions = np.arange(len(self.actionNames))
        self.width = width
        self.height = height
        self.epsilon = initialEpsilon
        self.stateVectorRows = stateVectorRows
        self.lowHeightDifBound = heightDifBound[0]
        self.upHeightDifBound = heightDifBound[1]
        self.heightRange = self.upHeightDifBound-self.lowHeightDifBound+1
        self.coarseResolution = coarseResolution
        self.maxCoarseHeight = maxCoarseHeight

        self.featureVectorLength = len(self.actions)*8*(self.width)*(self.heightRange**(self.width-1))
        #self.featureVectorLength = len(self.actions)*8*(self.width)*((2**self.stateVectorRows)**self.width)
        #self.featureVectorLength = len(self.actions)*8*(self.width)*((self.stateVectorRows+1)**self.width)*(2**self.width)


        self.initialWeights = initialWeights
        self.weights = np.ones(self.featureVectorLength)*self.initialWeights
        self.decayEpsilon=decayEpsilon
    
    def chooseAction(self, game):
            # Agent implementation goes here - epsilon greedy policy
            if random.uniform(0,1)<=self.epsilon:  #Explore exp_rate % of the time
                actionChosen=random.choice(self.actions)
            else:
                Qvalues=np.zeros(len(self.actions))
                for i, act in enumerate(self.actions):
                    featureIndex = self.getFeatureVector(act, game)
                    Qvalues[i] = self.getApproxQvalue(featureIndex)

                equivalentAction = [] #A list of actions as good as good as 'action'
                for act in self.actions:
                    if Qvalues[act] == np.amax(Qvalues):
                        equivalentAction.append(act)    #If action is as good, populate list
                actionChosen = random.choice(equivalentAction)  #Randomly select from best actions

            return actionChosen

    def greedyAction(self, game):
        Qvalues=np.zeros(len(self.actions))
        for i, act in enumerate(self.actions):
            featureIndex = self.getFeatureVector(act, game)
            Qvalues[i] = self.getApproxQvalue(featureIndex)

        equivalentAction = [] #A list of actions as good as good as 'action'
        for act in self.actions:
            if Qvalues[act] == np.amax(Qvalues):
                equivalentAction.append(act)    #If action is as good, populate list
        actionChosen = random.choice(equivalentAction)  #Randomly select from best actions
        return actionChosen
    
    def decodeColumnHeight(self, game):
        h = np.clip(game.columnHeights, a_min=0, a_max=self.stateVectorRows, dtype=int)
        #print(h)
        intRep = int(''.join(map(lambda h: str(int(h)), h)), self.stateVectorRows+1)
        return intRep

    def decodeCoarseColumnHeight(self, game):
        h = game.coarseColumnHeight
        intRep = int(''.join(map(lambda h: str(int(h)), h)), self.maxCoarseHeight+1)

        return intRep
    
    def decodeHeightDif(self, game):
        h = np.clip(game.heightDif+np.abs(self.lowHeightDifBound), a_min=0, a_max=self.upHeightDifBound+abs(self.lowHeightDifBound))
        intRep = int(''.join(map(lambda h: str(int(h)), h)), self.heightRange)
        return intRep
    
    def decodeHoles(self, game):
        binRep=game.columnHoles
        intRep = int(''.join(map(lambda binRep: str(int(binRep)), binRep)), 2)
        return intRep
    
    def decodeRows(self, game):
    
        maxHeight = np.amax(game.columnHeights)
        if maxHeight <= self.stateVectorRows:
            maxHeight=self.stateVectorRows
    
        transMaxHeight = int(game.height-maxHeight)
        binRep = np.array([])
        for i in range(transMaxHeight, transMaxHeight+self.stateVectorRows):
            binRep = np.append(binRep, [game.field[transMaxHeight,:]])
        
        intRep = int(''.join(map(lambda binRep: str(int(binRep)), binRep)), 2)
        return intRep

    
    def decodeBottomRows(self, game):
        #Converts the bottom row of the playing field to index compatible with feature vector
        bottomRow = game.field[game.height-1,:]
        decodedValue=0
        for i, n in enumerate(reversed(bottomRow)):
            decodedValue+=n*2**i
        return int(decodedValue)

    def getFeatureVector(self, action, game):
        #Constructs the 1 hot encoded feature vector
        position = game.block.x

        featureIndex = action*8*(game.width)*(self.heightRange**(self.width-1))
        #featureIndex = action*8*(game.width)*((2**self.stateVectorRows)**self.width)
        #featureIndex = action*8*(game.width)*((self.stateVectorRows+1)**self.width)
        #featureIndex = action*8*(game.width)*((self.stateVectorRows+1)**self.width)*(2**self.width)

        if game.block.index==0:
            pass
        if game.block.index==1:
            featureIndex += 1*(game.width)*(self.heightRange**(self.width-1))
            #featureIndex += 1*(game.width)*((2**self.stateVectorRows)**self.width)
            #featureIndex = 1*(game.width)*((self.stateVectorRows+1)**self.width)
            #featureIndex += 1*(game.width)*((self.stateVectorRows+1)**self.width)*(2**self.width)
        if game.block.index==2:
            featureIndex+= 2*(game.width)*(self.heightRange**(self.width-1))
            #featureIndex += 2*(game.width)*((2**self.stateVectorRows)**self.width)
            #featureIndex += 2*(game.width)*((self.stateVectorRows+1)**self.width)
            #featureIndex += 2*(game.width)*((self.stateVectorRows+1)**self.width)*(2**self.width)
        if game.block.index==3:
            featureIndex+= 4*(game.width)*(self.heightRange**(self.width-1))
            #featureIndex += 4*(game.width)*((2**self.stateVectorRows)**self.width)
            #featureIndex += 4*(game.width)*((self.stateVectorRows+1)**self.width)
            #featureIndex += 4*(game.width)*((self.stateVectorRows+1)**self.width)*(2**self.width)
        
        #if game.block.index==8:
            #featureIndex+= 8*(game.width)*(self.heightRange**(self.width-1))
            #featureIndex += 8*(game.width)*((2**self.stateVectorRows)**self.width)
            #featureIndex += 8*(game.width)*((self.stateVectorRows+1)**self.width)
       
        featureIndex += game.block.rotation*(game.width)*(self.heightRange**(self.width-1))
        featureIndex += position*(self.heightRange**(self.width-1))
        featureIndex += self.decodeHeightDif(game)
        
        #featureIndex += game.block.rotation*(game.width)*((2**self.stateVectorRows)**self.width)
        #featureIndex += position*((2**self.stateVectorRows)**self.width)
        #featureIndex += self.decodeRows(game)

        #featureIndex += game.block.rotation*(game.width)*((self.stateVectorRows+1)**self.width)
        #featureIndex += position*((self.stateVectorRows+1)**self.width)
        #featureIndex += self.decodeColumnHeight(game)

        #featureIndex += game.block.rotation*(game.width)*((self.stateVectorRows+1)**self.width)*(2**self.width)
        #featureIndex += position*((self.stateVectorRows+1)**self.width)*(2**self.width)
        #featureIndex += self.decodeColumnHeight(game)*(2**self.width)
        #featureIndex += self.decodeHoles(game)

        #featureIndex += game.block.rotation*(game.width)*((self.maxCoarseHeight+1)**self.width)
        #featureIndex += position*((self.maxCoarseHeight+1)**self.width)
        #featureIndex += self.decodeCoarseColumnHeight(game)

        return featureIndex

    def getApproxQvalue(self, featureIndex):
        #Returns the value of a state action pair
        Qvalue = self.weights[featureIndex]
        return Qvalue

    def decreaseEpsilon(self):
        self.epsilon = self.epsilon*self.decayEpsilon

a = playGame(selfPlay=False, player=0)
a.play()

#a = trainAgent(episodes=50000, decayEpsilon=1, initialEpsilon=0.01, rewardSignal=[10, -0.1, -2, 0, 1, 0, 2, 0.1], 
#verbose=True, stateVectorRows=4, heightDifBound=[-2,2], coarseResolution=2, maxLinesCleared=100, gamma=0.9, initialWeights=0)
#a.train()

#outfile=open("final_Sarsa_6", "wb")
#pickle.dump(a.player.weights, outfile)
#outfile.close()

#a.testAgent()

#b = playGame(selfPlay=True, player=a.player)
#b.play()


