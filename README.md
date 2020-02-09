# Investigating the performance of voting systems with agent-based voter interactions

In this project we have designed and implemented an Agent Based Model (ABM) to simulate an artificial society with social influence and opinion dynamics in order to compare the performance of popular voting systems. We focus on popular proportional representation electoral voting systems used throughout the world. The performance of selected systems are compared in terms of population dissatisfaction and parliament moderation.

This repository contains all the scripts required to run our model and generate our results. 

# Structure
* model.py implements our model of society.
* agent.py describes our voter agents, their attributes and step function.
* example.py provides a script for running one instance of our society model.
* SensitivityAnalysis.py contains the code required to perform Sobol sensitivity analysis on our model.
* server.ipynb contains a Jupyter Notebook showing an example of visualising our agents on the grid.
* STV.py contains our algorithm for an election using the Single-Transferable-Vote system.
* Experiments_with_interaction.ipynb contains a notebook outlining our code for running experiments without voter interaction.
* Experiments-without_interaction.ipynb contains a notebook outlining our code for running experiments including opinion dynamics through voter interaction.

Contributors: Fiona Gallagher, Parva Alavian, Cillian Hourican, Salome Kakhaia, Namitha Tresa Joppan
