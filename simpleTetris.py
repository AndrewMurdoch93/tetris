import pygame
import random
import numpy as np

#O = [[(1,1),(1,2),(2,1), (2,2)]]
#I = [[(1,0), (1,1), (1,2), (1,3)], [(0,2), (1,2), (2,2), (3,2)]]
#S = [[(1,2), (1,3), (2,1), (2,2)], [(0,2), (1,2), (1,3), (2,3)]]
#Z = [[(1,1), (1,2), (2,2), (2,3)], [(0,3), (1,2), (1,3), (2,2)]]
#L = [[(1,1), (1,2), (1,3), (2,1)], [(0,2), (1,2), (2,2), (2,3)], [(1,1), (1,2), (1,3), (0,3)], [(0,1), (0,2), (1,2), (2,2)]]
#J = [[(1,1), (1,2), (1,3), (2,3)], [(0,2), (0,3), (1,2), (2,2)], [(0,1), (1,1), (1,2), (1,3)], [(0,2), (1,2), (2,2), (2,1)]]
#T = [[(1,1), (1,2), (1,3), (2,2)], [(0,2), (1,2), (1,3), (2,2)], [(0,2), (1,1), (1,2), (1,3)], [(0,2), (1,1), (1,2), (2,2)]]

b=[[(0,0)]]


class tetromino:
    def __init__(self, x=2, y=0):
        self.x = x  #Position of tetromino block (defualt spawn position is x=3, y=0)
        self.y = y
        #self.types = ["O", "I", "S", "Z", "L", "J", "T"]
        #self.tetrominos = [O, I, S, Z, L, J, T]
        self.tetrominos = [b]
        self.rotation = 0
        self.index = random.choice(range(len(self.tetrominos))) #Randomly select tetromino to spawn
        #self.index=1 #Tetris easy mode
        #self.name = self.types[self.index]  #Name of spawned tetromino block
        self.shape = self.tetrominos[self.index][self.rotation] #Tetrmonio field positions
        
    def rotate(self, direction):
        if direction=="clockwise":
            self.rotation = (self.rotation + 1) % len(self.tetrominos[self.index])
        elif direction=="antiClockwise":
            self.rotation = (self.rotation - 1) % len(self.tetrominos[self.index])
        
        self.shape = self.tetrominos[self.index][self.rotation]

class tetris:
    def __init__(self, height=20, width=5):
        self.height=height  # Dimensions of the game field
        self.width=width
        self.block=tetromino()  # Spawns a tetromino in default position
        self.field=np.zeros((self.height,self.width)) # Initialise empty game field
        self.score=0
        self.gameOver = False
        self.linesCleared=0
        self.droppedBlocks=0
        self.reward=0

    def intersection(self):
        # Checks to see if a move is valid
        # Valid moves do not intersect occupied spaces in the field or move the tetromino beyond the field
        # intersection is called to freeze the tetromino in place

        for i in self.block.shape:  # Loops through positions of the tetromino shape 
            x=self.block.x+i[1] 
            y=self.block.y+i[0] 
            if x>=self.width or x<0 or y>=self.height or y<0:
                return True # Move is not valid (out of bounds)
            if self.field[y][x]==1:
                return True # Move is not valid (collides with occupied field)
        return False # Move is valid
      
    def clearLines(self):
        # If a line is completely filled, it disappears (is cleared)
        # The lines above the cleared line drop down 1 y value
        # Score is proportional to the square of the number of lines cleared at onve
        
        lines=0 # Set lines cleared

        for index, y in enumerate(self.field):
            if (y==np.ones(self.width)).all(): # Check to see if line is filled
                lines+=1
                self.linesCleared+=1
                for i in range(index, 0, -1):   # Drop lines above cleared line
                    self.field[i] = self.field[i-1]
                self.field[0]=0 # Generate zeros at the top of the field
        self.score+=(lines**2)    # Add to score counter
        self.giveReward(lines)

    def takeAction(self, command):
        # Makes a change to the game field based on given command (either from player or agent)
        # Checks to see if move is valid before making change
        
        if command=="right":
            self.block.x+=1
            if self.intersection():
                self.block.x-=1
        
        if command=="left":
            self.block.x-=1
            if self.intersection():
                self.block.x+=1
                
        if command=="rotate":
            self.block.rotate(direction="clockwise")
            if self.intersection():
                self.block.rotate(direction="antiClockwise")
       
        if command=="drop":
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
        
        for i in self.block.shape: 
            x = self.block.x+i[1] 
            y = self.block.y+i[0]
            self.field[y][x] = 1    # Default occupied field value is 1 
        self.clearLines()   # Clear the lines as soon as tetromino is frozen
        self.block=tetromino()  # Spawn a new tetromino
        self.droppedBlocks+=1
        if self.intersection():
            self.gameOver = True

    def getState(self):

        height=np.zeros((self.width))
        heightDif=np.zeros((self.width-1))

        for i in range(self.width):
            if (self.field[:,i]==np.zeros((self.height))).all():
                height[i]=20    
            else:
                height[i]=np.where(self.field[:,i]==1)[0][0]
        height=20-height        
        #print(height)

        for i in range(self.width-1):
            heightDif[i] = height[i+1]-height[i]
        #print(heightDif)

        x=self.block.x
        state=np.hstack(np.array([[x], list(heightDif)]))

        return state

    def giveReward(self, lines):
        self.reward=lines**2

    
class playerGame():
    # A tetris implementation that is playable by humans 

    def __init__(self, agent=False):
            self.game = tetris()
            self.black = (0, 0, 0)
            self.white = (255, 255, 255)
            self.gray = (128, 128, 128)
            self.fps = 25
            self.counter = 0
            self.size = (400, 500)
            self.done = False
            self.zoom = 20
            self.x=100
            self.y=60
            self.level=4
            self.agent=agent
    
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
                if self.agent==False:
                    if event.key == pygame.K_UP:
                        action="rotate"
                    if event.key == pygame.K_LEFT:
                        action="left"
                    if event.key == pygame.K_RIGHT:
                        action="right"
                    if event.key == pygame.K_SPACE:
                        action="drop"
    
        if self.agent==True:
            action = random.choice(["right", "left", "drop", ""]) # Moves available to agent are limited to avoid incredibly fast play
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

        while self.done==False:
            
            # Counter to move tetromino down 1 vertical position
            self.counter += 1
            if self.counter > 1000:
                self.counter = 0
            #if self.counter % (self.fps//self.level) == 0 and self.game.state=="play":
                #self.game.takeAction("down")
            
            action = self.chooseAction()
            if self.game.gameOver==True: #Player can no longer take actions when state=="gameOver"
                self.game.takeAction(action)
            
            self.displayScreen(screen)
            clock.tick(self.fps)    #Limit game to specified fps
        
        pygame.quit()


class agentGame():
    # An implementation of tetris that isnt playable by humans
    # Intended use is to train agents

    def __init__(self):
        self.game = tetris()
        self.actions = [0, 1, 2]
        self.actionNames = ["left", "right", "drop"]
        self.State = self.game.getState()
        self.nextState = self.State
        self.left1hot = np.hstack([np.ones(len(self.State)), np.zeros(len(self.State)), np.zeros(len(self.State))])
        self.right1hot = np.hstack([np.zeros(len(self.State)), np.ones(len(self.State)), np.zeros(len(self.State))])
        self.drop1hot = np.hstack([np.zeros(len(self.State)), np.zeros(len(self.State)), np.ones(len(self.State))])
        self.stateVector = np.hstack(np.array([self.State, self.State, self.State]))
        self.weights = np.random.normal(size=len(self.stateVector))
        self.approximation = np.multiply(self.weights, self.stateVector)
        self.Q_values = np.zeros(len(self.actions))

        self.rounds = 100
        self.alpha = 0.1
        self.gamma=0.9

    def chooseAction(self):

        # Agent implementation should go here
        self.Q_values[0]=sum(np.multiply(self.approximation, self.left1hot))
        self.Q_values[1]=sum(np.multiply(self.approximation, self.right1hot))
        self.Q_values[2]=sum(np.multiply(self.approximation, self.drop1hot))
        action = self.actions[np.argmax(self.Q_values)]
        
        return action

    def getq_values(self, action):
        self.approximation = np.multiply(self.weights, self.stateVector)
        q_values=np.zeros(len(self.stateVector))
        dq_values=np.zeros(len(self.stateVector))        

        if action == 1:
            q_values = np.multiply(self.approximation, self.left1hot)
            dq_values = np.multiply(self.stateVector, self.left1hot)
        elif action == 2:
            q_values = np.multiply(self.approximation, self.right1hot)
            dq_values = np.multiply(self.stateVector, self.right1hot)
        elif action == 3:
            q_values = np.multiply(self.approximation, self.drop1hot)
            dq_values = np.multiply(self.stateVector, self.right1hot)

        return q_values, dq_values
    

    def play(self):
        # Controls game logic
        self.State = self.game.getState()
        action = self.chooseAction()
        self.game.takeAction(self.actionNames[action])
        self.nextState = self.game.getState
        i=0

        while i<self.rounds:
            self.game.takeAction(self.actionNames[action])
            #R is observed by taking action

            if self.game.gameOver==True or self.game.linesCleared>100:
                self.game = tetris()
                i+=1
        
            else:
                nextAction = self.chooseAction()
                q_current,dq_current = self.getq_values(action)
                q_next,dq_next = self.getq_values(nextAction)

                self.weights +=  self.weights + self.alpha*(self.game.reward + self.gamma*q_current - q_next)*dq_current
                
                self.State = self.nextState
                action = nextAction

 

#a = playerGame(agent=True)
#a = playerGame(agent=False)
a = agentGame()

a.play()