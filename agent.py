'''
This script contains the voter Class including:
      -initializer
      -ideology_calculator
      -step function
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


    
class Voter(Agent):
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        '''
        Initialising a voter object with a random view
        '''
        self.model = model
        
        self.moving = 1
        #parameters for underlying pdf ofr agents' views which is sum of two truncated normal distribution
        myclip_a = 0       #lower bound for the truncated normal distribution
        myclip_b = 1       #upper bound for the truncated normal distribution
        my_std = 0.2       #standard deviation for the truncated normal distribution
        if random.random()<0.5:   
            #agent's view is sampled from a truncated normal distribution with a mean on the right side of the spectrum
            my_mean = 0.5 + model.polarization/2
        else:
            #agent's view is sampled from a truncated normal distribution with a mean on the left side of the spectrum
            my_mean = 0.5 - model.polarization/2
        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        view = truncnorm.rvs(a, b, loc=my_mean,scale=my_std)
        
        self.view=view
        #calculating the ideology vector based on view
        self.ideology=self.ideology_calculator(model.parties,self.view)    
    
    def ideology_calculator(self,parties,view):
        '''
        This function gets the agent's view and all the parties' location on the spectrum and calculates how strongly the agent
        would support each party
        '''
        ideology=[]
        for i in range(len(parties)):
            # the farthest the view is from a certain party, the less the agent is likely to support them
            ideology.append(1-np.abs(view-parties[i]))
            # ideology vector is normalised to one
        return(ideology/sum(ideology))
        

        
    def step(self):
        '''Interacting with neighbours , schelling segregation and opinion dynamics'''
        nbh = SingleGrid.get_neighbors(self.model.grid,self.pos,moore=1)
        len_nbh = len(nbh)
        K = self.model.K #social influence
        C = self.model.C #confirmation bias
        
        if len(nbh)!=0:
            '''Loop through neighbours, calculate K_ij for each'''
            K_ij = 0
            for i in range(len_nbh):
                if (abs(self.view - nbh[i].view)) < 0.5*(1-C):
                    '''sum the effect on agent's view from all neighbors whose views are close to that of the agent'''
                    K_ij = K_ij+ K*(nbh[i].view - self.view) 
                    
            self.view = self.view + K_ij  #update agent's view
            
            '''ensuring that after interaction, attribute view stays within its bounds'''
            if self.view >1:
                self.view =1
            elif self.view < 0:
                self.view=0
            self.ideology=self.ideology_calculator(self.model.parties,self.view)
            
            '''Randomly move, with probability P_i which is proportional to mean of agent's distance fromm its neighbors'''
            P = []
            for i in range(len_nbh): 
                neighbour = nbh[i]
                P.append(abs(self.view - neighbour.view))  #collecting the difference between agents view and all its neighbors
                
            p_i = np.mean(P)-0.1
            self.model.avg_move_prob = self.model.avg_move_prob + p_i #average moving probability which is used in ensuring convergence
            
            if random.uniform(0,1) < p_i:
                self.model.grid.move_to_empty(self)  #agent moves to an empty space on the grid
                self.model.move_count += 1           #counter to track the total number of agents that move at each time step
                self.moving = abs(self.moving) + 1   #counter to keep track of how long an agent has been stationary
                
            else:
                self.moving -=1 
            

        else:
            '''If an agent does not have any neighbors, he definitely moves'''
            self.model.grid.move_to_empty(self)
            self.model.move_count += 1
            self.model.avg_move_prob = self.model.avg_move_prob + 1
            self.moving = abs(self.moving) + 1