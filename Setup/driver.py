from setup_sim import setup_simulation
from payoffs import get_payoffs

"""
    Set Input Parameters
    
    """

base = 0.0001

# Population (size of network)
# CHOOSE LIST OF POPULATIONS TO SIMULATE (INTEGER ONLY)
pop_list_in = [50]

# Discounting determines memory of agents
# Acceptable values are in [0,1] range inclusive
net_discount_list_in = [0.01]

# Network and strategy tremble (0.01 is default)
# STRATEGY AND NETWORK TREMBLE EQUAL
net_tremble_list_in = [0.01]
strat_tremble_list_in = [0]
imitation_tremble_list_in = [0.01]

# Network and learning speed 
net_speed_list_in = [1]

# 0: only visitor updates network/memory matrix 1: Both players update
net_sym_list_in = [0]
memory_sym_list_in = [1]

total_weight = 2

# INITIAL CONDITIONS FOR SOCIAL STRATEGY
# DEFAULT 50, 50 (EQUAL NUMBER OF SOCIAL AND SPITEFUL AGENTS INITIALIZED)
init_cond_list_in = [50]

# number of time steps
t_max = 1000000

# how frequently a player considers imitating at end of each round 
imitationRate_list_in = [0.01] 

# 0 compare total payoff, 1 compare average payoff
average_comp_list_in = [1]

game_name = "Delight"

# how many previous payoff units a player tracks
memory_list_in = [1]

# unit of memory
memory_type_in = "round"

# Payoffs
payoffs = get_payoffs(game_name)

setup_simulation(base, payoffs, pop_list_in, net_discount_list_in, init_cond_list_in, net_speed_list_in, net_tremble_list_in, strat_tremble_list_in, imitation_tremble_list_in, 
net_sym_list_in, memory_sym_list_in, total_weight, t_max, imitationRate_list_in, memory_list_in, average_comp_list_in, memory_type_in, game_name)
