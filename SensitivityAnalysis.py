""" Sensitivity Analysis"""

# Import necessary libraries
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np

from SALib.sample import saltelli
from mesa.batchrunner import BatchRunner
from SALib.analyze import sobol
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from model import Society

##############################################
# Define function for creating graphical output of Sensttivity Analysis
def plot_index(s, params, i, title=''):
    """
    Taken from SA notebook:
    
    Creates a plot for Sobol sensitivity analysis that shows the contributions
    of each parameter to the global sensitivity.

    Args:
        s (dict): dictionary {'S#': dict, 'S#_conf': dict} of dicts that hold
            the values for a set of parameters
        params (list): the parameters taken from s
        i (str): string that indicates what order the sensitivity is.
        title (str): title for the plot
    """

    if i == '2':
        p = len(params)
        params = list(combinations(params, 2))
        indices = s['S' + i].reshape((p ** 2))
        indices = indices[~np.isnan(indices)]
        errors = s['S' + i + '_conf'].reshape((p ** 2))
        errors = errors[~np.isnan(errors)]
    else:
        indices = s['S' + i]
        errors = s['S' + i + '_conf']
        plt.figure()

    l = len(indices)

    plt.title(title)
    plt.ylim([-0.2, len(indices) - 1 + 0.2])
    plt.yticks(range(l), params)
    plt.errorbar(indices, range(l), xerr=errors, linestyle='None', marker='o')
    plt.axvline(0, c='k')

##############################################
# Define the variables we want include in the Sensitivity analysis, and their bounds
problem = {
    'num_vars': 6,
    'names': [ 'step_count', 'num_candidates', 'num_seats','polarization','C','K'],
    'bounds': [[1,80],
               [2, 10],
               [1, 20],
               [0.01, 0.99],
               [0.01, 0.99],
               [0.001, 0.99]]
}

# Sample input values for each parameter value to use in the analyisis
N = 300
param_values = saltelli.sample(problem, N)

##############################################
# Run the model for each sampled combination of input parameters

# Define empty arrays to save model outputs for all sets of inputs used
Y_hs_2 = np.zeros([param_values.shape[0]]) # Dissatisfaction with Scored Voting 
Y_hf_2 = np.zeros([param_values.shape[0]]) # Dissatisfaction with FPTP
Y_hp_2 = np.zeros([param_values.shape[0]]) # Dissatisfaction with STV
Y_mf_2 = np.zeros([param_values.shape[0]]) # Moderation with FPTP
Y_mr_2 = np.zeros([param_values.shape[0]]) # Moderation with STV
Y_ms_2 = np.zeros([param_values.shape[0]]) # Moderation with Scored Voting

for i, X in enumerate(param_values):
    # Use the sampled input values
    model = Society(2000,int(X[1]),int(X[2]),X[3],X[4],X[5],height=50,width=50)
    model.run_model(step_count = int(X[0]))
    
    # Save all model outputs for analysis
    Y_hs_2[i] = np.mean(model.happiness_scores)
    Y_hf_2[i] = np.mean(model.happiness_fp)
    Y_hp_2[i] = np.mean(model.happiness_ranking)
    Y_mf_2[i] = model.moderation_FP
    Y_mr_2[i] = model.moderation_Ranking
    Y_ms_2[i] = model.moderation_Scores
    
    # Print the progress of the iterations (to see the progress)
    print(i)

##############################################
# Perform Sobol Analysis on model outputs:

# Repeat for each model output 
Si_hs_2 = sobol.analyze(problem, Y_hs_2) # Dissatisfaction: Scores
Si_hf_2 = sobol.analyze(problem, Y_hf_2) # Dissatisfaction: FPTP
Si_hp_2 = sobol.analyze(problem, Y_hp_2) # Dissatisfaction: STV
Si_mf_2 = sobol.analyze(problem, Y_mf_2) # Moderation: FPTP
Si_mr_2 = sobol.analyze(problem, Y_mr_2) # Moderation: STV
Si_ms_2 = sobol.analyze(problem, Y_ms_2) # Moderation: Scores


##############################################
# Plot Sensitivity Analysis results

# labels for iteratively saving plots
labels = ['hs2','hf2','hp2','mf2','mr2','ms2']

idx = 0
print("Sensitivity analysis Graphs")
for Si in (Si_hs_2, Si_hf_2,Si_hp_2,Si_mf_2,Si_mr_2,Si_ms_2):
    print("Sensitivity Analysis for one output")
    # First order
    plot_index(Si, problem['names'], '1', 'First order sensitivity')
    name = 'SA_No_Interactions '+str(labels[idx])+' Order 1 '
    plt.savefig(name,bbox_inches='tight')
    plt.show()

    # Second order
    plot_index(Si, problem['names'], '2', 'Second order sensitivity')
    name = 'SA_No_Interactions '+str(labels[idx])+' Order 2 '
    plt.savefig(name,bbox_inches='tight')
    plt.show()

    # Total order
    plot_index(Si, problem['names'], 'T', 'Total order sensitivity')
    name = 'SA_No_Interactions '+str(labels[idx])+' Order T '
    plt.savefig(name,bbox_inches='tight')
    plt.show()
    
    idx += 1