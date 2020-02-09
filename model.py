'''
This script contains the main model namely the society class It includes:
    -class initializer
    -init_population which is responsible for creating the agents in the society
    -happiness_calc which calculates the general satisfaction with results 
        (Note: the word happiness was used instead of satisfaction)
    -moderation_calc
    -step
    -vote
    -run_model
'''


from STV import Ranked_vote
from mesa import Agent, Model
import random
from mesa.space import Grid
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
import matplotlib.pyplot as plt
import math
from mesa.space import MultiGrid
from mesa.space import SingleGrid
import numpy as np
import seaborn as sns
from tqdm import tqdm
import seaborn as sns
import time
from scipy.stats import truncnorm
from agent import Voter

class Society(Model):
    def __init__(self,number_voters=2000,number_candidates=5,number_seats=20,polarization=0.5,C=0.8, K=0.05,height=50, width=50):
        super().__init__()
        '''
        Initialising the society class with the values given in inputs: number of voters(agents) , 
        number of parties participating in the election, number of seats in the parliament that need to be filled,
        polarization of the society, confirmation bias(c), social influence strength(k) and size of the grid.
        All other parameters in the model are initialised to 0 and will be calculated later.
        '''
        self.height = height
        self.width = width
        self.number_voters = number_voters
        self.polarization = polarization
        self.number_candidates=number_candidates        
        self.number_seats = number_seats
        self.happiness_fp = 0
        self.happiness_scores = 0
        self.happiness_ranking=0
        
        # moderation index for all voting systems
        self.moderation_FP = 0    #??
        self.moderation_Scores = 0
        self.moderation_Ranking = 0
        
        # parliament distribution for all voting systems
        self.parliament_fp = 0
        self.parliament_scores = 0
        self.parliament_ranking = 0
        
        self.has_neighbours = 0
        self.move_count = 0
        self.avg_move_prob = 0
        self.K = K
        self.C=C
        
        # for simplicity we assume that all the parties are equally spaced on the view spectrum (left-right spectrum)
        parties=np.linspace(0,1,number_candidates)
        self.parties = parties
        self.schedule = RandomActivation(self) 
        
        #Initialise the DataCollector
        self.datacollector = DataCollector(
             { "Happiness FP": lambda m: self.happiness_fp, 
              "Happiness Scores": lambda m: self.happiness_scores,
              "Happiness STV": lambda m: self.happiness_ranking,
              "Moderation_FP": lambda m: self.moderation_FP,
             "Moderation_Scores": lambda m: self.moderation_Scores,
              "Moderation_STV": lambda m: self.moderation_Ranking,
             "Parliament FP": lambda m: self.parliament_fp,
             "Parliament Scores": lambda m: self.parliament_scores,
             "Parliament STV": lambda m: self.parliament_ranking,
             'Move count': lambda m: self.move_count,
             'Avg prob moving': lambda m: self.avg_move_prob})
        
        self.grid = SingleGrid(self.width, self.height, torus=1)
        
        # Initialise voter population
        self.init_population(Voter, self.number_voters)

        
        # This is required for the datacollector to work
        self.running = True
        #self.datacollector.collect(self)
        
        
    def init_population(self, agent_type, n):
        '''
        Making a fixed amount of voters
        '''
        for i in range(n):
            agent = agent_type(self.next_id(), self)
            self.grid.position_agent(agent)
        
            getattr(self, f'schedule').add(agent)
            
 
        
    def happiness_calc(self,ideol,parliament_firstpref,parliament_scores,parliament_ranking):
        '''
         calculates agent's dissatisfaction with the parliament results for all three voting systems using L2 norm of
         the difference between agent's ideology vector and parliament's vector
        '''
        
        Differences_Scores = abs(ideol - parliament_scores)
        Differences_First_Pref = abs(ideol - parliament_firstpref)
        Differences_Ranking = abs(ideol - parliament_ranking)

        Happiness_Scores =   np.sum((Differences_Scores)**2, axis = 1)**(0.5)
        Happiness_First_Pref = np.sum((Differences_First_Pref)**2, axis = 1)**(0.5)
        Happiness_Ranking = np.sum((Differences_Ranking)**2, axis = 1)**(0.5)

        self.happiness_fp = Happiness_First_Pref
        self.happiness_scores = Happiness_Scores
        self.happiness_ranking=Happiness_Ranking
        
    def Moderation_calc(self, parliament_firstpref, parliament_scores, parliament_ranking):
        '''
        Function to calculate how moderate the parliament is.
        moderation of the parliament is sum of the absolute difference between 
        each party's view and complete moderation times the number of seats they own in the parliament
        '''
            
        self.moderation_FP = np.sum(parliament_firstpref * abs(self.parties-0.5))*2
        self.moderation_Scores = np.sum(parliament_scores * abs(self.parties-0.5))*2
        self.moderation_Ranking = np.sum(parliament_ranking * abs(self.parties-0.5))*2
        
    def step(self):
        '''
        Method that calls the step method for each of the voters.
        '''
        self.schedule.step()

    def vote(self):

        '''
        This function collects the ideology vectors of all self.schedule.agents and turns them into votes and
        returns the final parlaimant distriution for all 3 voting systems
        '''
        # Collect ideologies of all agents
        Ideologies_Collection = [a.ideology for a in self.schedule.agents]
        Ideologies_Collection2 = np.stack( Ideologies_Collection, axis=0 ) # Stack into one array

        '''First past the post: assume each agent votes for their favorite party(the one with biggest value in ideology vector)'''
        First_Preference_Votes = np.argmax(Ideologies_Collection2, axis=1) # fins the index(party) of each agent's first preference
        Parlaimant_Distribution_First_Pref = []
        for candidate in range(self.number_candidates):
            #count the votes for each candidate and normalize it
            Parlaimant_Distribution_First_Pref.append(list(First_Preference_Votes).count(candidate)) 
        Parlaimant_Distribution_First_Pref = np.array(Parlaimant_Distribution_First_Pref)/self.number_voters
        a = Parlaimant_Distribution_First_Pref
        quota = 1/self.number_seats # calculate quota for election
        seats=np.zeros(self.number_candidates)
        excess=np.zeros(self.number_candidates)
        for i in range (len(a)):
            if a[i] > quota:
                '''If a party exceeds the quota, allocate the seats to them and calculate the excess votes'''
                temp= int(a[i]/quota)
                seats[i]+= temp
            excess[i]= a[i] % quota
        while (sum(seats)<self.number_seats):
            '''If there are empty seats in the parlaimant allocate them to the party with most excess votes'''
            ind=np.argmax(excess)
            seats[ind]+=1
            excess[ind]=0
        Parlaimant_Distribution_First_Pref = np.array(seats)/self.number_seats #normalize the final distribution of parliament to 1

        '''Scored voting: Each agent assigns a score to each candidate''' 
        '''calculate each party's aggregated scores '''
        Parlaimant_Distribution_Scores = np.sum(Ideologies_Collection2, axis = 0)/self.number_voters 
        a = Parlaimant_Distribution_Scores
        quota = 1/self.number_seats  # calculate quota for election
        seats=np.zeros(self.number_candidates)
        excess=np.zeros(self.number_candidates)
        for i in range (len(a)):
            if a[i] > quota:
                '''If a party exceeds the quota, allocate the seats to them and calculate the excess votes'''
                temp= int(a[i]/quota)
                seats[i]+= temp
            excess[i]= a[i] % quota
        while (sum(seats)<self.number_seats):
           '''If there are empty seats in the parlaimant allocate them to the party with most excess votes'''
           ind=np.argmax(excess)
           seats[ind]+=1
           excess[ind]=0
        Parlaimant_Distribution_Scores = np.array(seats)/self.number_seats

        Parlaimant_Distribution_Ranked=Ranked_vote(self.number_voters,self.number_candidates, self.number_seats, Ideologies_Collection2,quota)
        return(Ideologies_Collection2, Parlaimant_Distribution_First_Pref,Parlaimant_Distribution_Scores,Parlaimant_Distribution_Ranked)
  
        
    def run_model(self, step_count=20):
        
        '''
        Method that runs the model for a specific amount of steps. 
        At each step, the model calls the model step() method (movement).
        We vote every 10 timesteps (can be changed).
        '''
        
        for i in (range(step_count)):
            self.move_count = 0
            self.avg_move_prob = 0
            
            self.step()
            if step_count%10==0:
                '''
                after a pre-defined number of steps agents vote.
                the dissastisfaction with resultsand the moderation of the resulting parliament 
                is calclated in all three voting systems at the same time
                '''
                ideologies,parliament_firstpref,parliament_scores, parliament_ranking = self.vote() 
                self.parliament_fp = parliament_firstpref  
                self.parliament_scores = parliament_scores
                self.parliament_ranking = parliament_ranking

                self.happiness_calc(ideologies,parliament_firstpref,parliament_scores,parliament_ranking)
                self.Moderation_calc(parliament_firstpref,parliament_scores,parliament_ranking)
            self.avg_move_prob = self.avg_move_prob/self.number_voters #average probability of moving is updated to monitor convergence
            
            self.datacollector.collect(self) # collects data from the model