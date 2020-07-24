Readme for simulation code in "Spite is contagious on dynamic networks"

Paper by Fulker, Forber, Smead, and Riedl

This README describes how to run code used in above paper. The code simulates a Prisoner's Delieght game with 
reinforcement learning on adjacency links and imitation learning for strategies. Agents thus learn what strategy to play 
and who to play against. Agents choose their interaction partner, update who they are likely to choose on the basis of 
payoffs received, and periodically imitate others who are receiving higher average payoffs.

This project includes this README, three python files that are used to set up simulation in the Setup folder, C++ simulation code in the SimCode folder, 
an IM_Input folder to hold generated model inputs (including an example), an IM_Output_Data folder to hold simulation results (including an example), GameTheoryEvo.exe
a compiled example executable of the simulation code, and “ESD_Seeds_All_Ordered.csv” which C++ reads a list of seeds from.

###########################################

Parameters of this model are the following:
base, payoffs, pop_list_in, net_discount_list_in, init_cond_list_in, net_speed_list_in, net_tremble_list_in, strat_tremble_list_in, imitation_tremble_list_in, 
net_sym_list_in, memory_sym_list_in, t_max, imitationRate_list_in, memory_list_in, average_comp_list_in, memory_type_in, game_name, run_now, num_seeds

Base: A negligible decimal value added to the outcome of every payoff to prevent link weights from reaching 0 (default 0.0001)

Payoffs [p1_payoffs,p2_payoffs, array w/ values in [0,1]: Interaction payoffs given by visitor and host strategy played in a single game. See table in Figure 1 of paper 
for constraints.

Population [pop, integer > 1]: Number of agents (default 20)

Network Discount [net_discount, float in [0,1]]: memory factor in link updating. All current outgoing network weights are multiplied by 1 - net_discount just before 
network learning occurs. As it is a multiplication factor, 1 implies maximum discounting, 0 implies no discounting. (default 0.01)

Initial Strategy [init_strategy, integer>1]: initial strategy configuration, 50 represents an equal number of Social and Spiteful initialized agents, 
100 represents an entirely Spiteful initial population (defualt 50)

Network Learning Speed [net_learning_speed, float >= 0]: multiplier for network payoff additions (larger value corresponds to faster network learning). 
0 implies no learning. (defualt 1)

Network Tremble [net_tremble, float in [0,1]]: network error rate between 0 and 1 (default 0.01). When agents make a network error, they choose a random agent in the 
network to play against regardless of adjacency weights. 0 implies no errors, 1 means that agents make a random partner choice every round.

Strategy Tremble [strat_tremble, float in [0,1]]: strategy error rate between 0 and 1 (default 0). When agents make a strategy error they flip a coin to decide which 
strategy to play, rather than relying on strategy weights.

Imitation Tremble [imit_tremble, float in [0,1]]: imitation error rate between 0 and 1 (default 0.01). When agents make an imitation error they flip a coin to decide 
what strategy to adopt for futute interactions.

Network Symmetry [net_sym, integer 0 or 1]: When network symmetry is 0 only the (visitor) agent who selected an interaction partner updates their outgoing network weights. 
If 1 both agents update their outgoing network weights. (default 0)

Memory Symemetry [mom_sym, integer 0 or 1]: When memory symmetry is 0 only the (visitor) agent who selected an interaction partner updates their memory of payoffs.
If 1 both agents update their memory. (default 1)

T Max integer > 1: Number of timesteps to complete (default 1,000,000)

Imitation Rate [imit_rate, float in [0,1]]: imitation rate between 0 and 1 (default 0.01). The likelihood that an agent considers imitatating another agent's strategy at the end 
of a timestep. 0.01 represents the case where each agents considers imitation on average once every 100 timesteps.

Memory [mem, integer > 1]: Number of rounds or interactions that an agent tracks their history of payoffs, akin to a maximum memory. (default 1)

Average Comparison [average_comp, integer 0 or 1]: Comparison method an agent uses when considering imitation of another agent's strategy. 0 represents a comparison of total 
payoffs tracked in memory, 1 is a comparison of average payoffs tracked in memory. (default 1)

Memory Type [mem_type, str]: Unit of measurement for an agent's memory. 'round' means and agent's memory length is measured in timesteps i.e. a 5 timestep memory of payoffs, 
'interaction' means an agent's memory is measured in terms of interactions i.e. an agent remembers their payoffs from the past 5 interactions. (default 'round')  

###########################################
############## Downloads ##################
###########################################

Compile Requirements: On Windows 10.0.18363, I have downloaded boost-1.72.0 and Cygwin x86_64 version 3.1.4.

To download Cygwin, see here: https://cygwin.com/install.html

Make sure the Devel package is installed with Cygwin (Download takes ~1 hour): https://warwick.ac.uk/fac/sci/moac/people/students/peter_cock/cygwin/part2/

To download boost, see here (Extraction takes ~45 minutes): https://dl.bintray.com/boostorg/release/1.72.0/source/

Next add Cygwin to you PATH variable: https://www.eclipse.org/4diac/documentation/html/installation/cygwin.html

Python will also need to be in your PATH if it is not already: https://geek-university.com/python/add-python-to-the-windows-path/

###########################################
############### Compile ###################
###########################################

Open the command prompt and navigate to the project folder.
SEE GameTheoryEvo.exe as an example for compiled output

*Be sure to replace C:\Users\Owner\Documents with the file path to your boost download
Once you have all those, compile code is:
g++ -g -O3 -Wall SimCode/*.cpp -I C:\Users\Owner\Documents\boost_1_72_0\boost_1_72_0\stage\include -L C:\Users\Owner\Documents\boost_1_72_0\boost_1_72_0\stage\lib -std=c++11 -o GameTheoryEvo

OR, this code supports multi-threading. To compile with multi-threading, use: 
g++ -g -O3 -Wall -fopenmp SimCode/*.cpp -I C:\Users\Owner\Documents\boost_1_72_0\boost_1_72_0\stage\include -L C:\Users\Owner\Documents\boost_1_72_0\boost_1_72_0\stage\lib -std=c++11 -o GameTheoryEvo
*MAKE sure your gcc is set up to run OpenMP otherwise this code will run indefinitely and produce no output

###########################################
########## Simulation Setup ###############
###########################################

Next, open driver.py and choose the model parameters that you would like to simulate. As long as the datatypes are compatible (outlined above) and each input list has at 
least one element, it will work.

Then, open payoffs.py and enter the values of the c payoff for you desired b/c ratios into the list on line 7.

BEWARE choosing model parameters. “setup_simulation.py” will create an input parameter set for every possible combination of lists that you enter. Choosing 5 populations, 
4 payoff points, 3 imitation rates, and 2 learning speeds will result in 5*4*3*2 = 120 total simulation parameter sets (times however many seeds are specified).

ENTER to create input files with all parameter combinations: python Setup/driver.py 
*Please keep file path to folder short or files being created may exceed max file path character length (260), this limit can be removed in Registry Editor if necessary.

###########################################
############### Running ###################
###########################################

You will need to know 3 things: Number of seeds to run (num_seeds below), the name of the input folder (Input_Folder below) where the input files are stored 
(inside IM_Input after you run python driver.py, remove 'Input_IM_' and copy starting with 'Pop_...'), and the number of threads to run in parallel. 
If you compiled with no “-fopenmp” flag, the number of threads must be 1.

Compiled_Executable Input_Folder Input_File_Number Thread_Count Num_Seeds Base_Seed Start_Key
Ex: GameTheoryEvo Pop-50_Discount-0.01_Tremble-0.01_NLS-1_SymN-0_SymM-1_ImRate-0.01_Memory-1_TMAX-1.0M_Delight_InitH-50_P-0.0-0.8571428571499999-0.14285714285-1.0_AvgComp-1_MType-round 0 1 25 0 0

*BE sure to start the input file name at Pop...
*FOR driver inputs with more than 20 parameter combinations multiple configuration files are created. 
To run all configuration files you have to specify the Input_File_Numbers for each. 0 for the 1st, 1 for the 2nd etc.

EXAMPLE RUN TIME: ~2 hours for 25 seeds of the baseline parameters on 1 thread
*Run time significantly decreases for smaller populations than the baseline of 50

###########################################
######## Interpreting Results #############
###########################################

For each seed of each parameter combination, 4 output files are produced to track model behavior. 
These files are marked with a key and a seed. Each key represents a unique parameter combination and the configuration file can be used to match keys and parameters.
An example set of outputs for 1 seed and our baseline parameters can be found in the IM_Output folder.

*NOTE these files may not open manually if file path is over 260 characters

IM_Weights_Key_Seed.csv: This file documents the weight of each link at the specified timesteps to record.

IM_Strategy_Key_Seed.csv: This file documents the strategy type of each agent at the specified timesteps to record. Each agent is represented by one column for spite and one column for
social. So for example if there are 50 agents there will be 100 columns. A 1,0 pairing is a spiteful agent and a 0,1 pairing is a social agent. 

Im_EvoStats_Key_Seed.csv: This file tracks the expected proportion of each interaction type at the specified timesteps to record. (Columns in order: CC, CD, DC, DD) 

IM_AvgPayoffs_Key_Seed.csv: This file tracks the average of the payoffs recorded in memory for each agent at the specified timesteps to record.


###########################################
####### Calculating Correlation ###########
###########################################

The included file get_correlation.py calculates the degree of correlated interaction values for a given set of simulations. The code is setup so that the user enters a hard coded 
output folder and population size in the get_corrs.py function. Then on line 70 the user enters all of the keys from the given folder for which they would like to generate output.

Output: The code saves a pickled output file after it finishes running. The output is a list of lenghth 2. The first item in the list is a dictionary containing 4 lists. 
The 4 items in the dictionary are the seqeunce (averaged over all seeds) of social, spiteful, and overall degree of correlated interactions as well as the sequence if the 
proportion of spiteful agents. The second item in the outermost output list is a list of dictionaries that hold the degree of correlated interactions for each individual seed.     


###########################################
#### Replicating Supplementary Material ###
###########################################

Several results in the supplementary information used parameter combinations that can not be recreated by altering inputs in the driver.py file. Details on how to run these 
simulations are provided below.

Figure S4: This set of simulations looked at the effect of changing the payoff structure so that b+c<1. To run these simulations change 1-c on line 8 in payoff.py to be X-c
where X is your desired value for b+c. Also b+c in lines 11 and 12 should be replaced with 1, so the social-social payoff does not change with changes to b+c.

Figure S5: This set of simulations looked at the effect of replacing imatation learning with the biologically based moran process. To run simulations using the moran process 
open NetworkDriver.cpp. Comment out line 405 and uncomment lines 409-413. Then you will need to recompile SimCode to create new executable for the moran process.
