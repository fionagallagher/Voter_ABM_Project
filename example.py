'''
An example for how to initialize the parameters and run the model
'''
from model import Society


number_voters = 2000
polarization= 0.1
C=0.3
K=0.01
num_candidates= 5
num_seats=10
height = 50
width = 50
model = Society(number_voters,num_candidates,num_seats,polarization,C,K,height,width)


model.run_model(step_count=30)
