import os, sys
from itertools import product
import numpy as np
import random
import string
import pandas as pd
import re
from itertools import combinations
from subprocess import call
from pathlib import Path
import copy

# Helper function to flatten list of lists
flatten = lambda l: [item for sublist in l for item in sublist] 

#base_path = '\\\\?\\' + os.path.abspath(os.path.dirname(__file__))
base_path = os.path.abspath(os.path.dirname(__file__))


# Header of input files (these are all parameters that C++ simulation code uses)
full_input_list = ['Base','Pop','TMax','NetDiscount','NetLearningSpeed','MemSymmetric','NetSymmetric','ImTremble','NetTremble','StratTremble','ImRate','Memory','AverageComp','MemoryType','Game','OutFolder','Key']

def build_prev_inputs(full_inputpath):
    existing_inputs = []
    if os.path.exists(full_inputpath):
        existing_inputs = [a for a in os.listdir(full_inputpath) if a.endswith("conf")]
    
        all_input_dfs = []
        
        for inp_file in existing_inputs:
            full_inpfile_path = os.path.abspath(os.path.join(full_inputpath,inp_file))
            inp_df = pd.read_csv(full_inpfile_path,sep=" ",dtype=str)
            all_input_dfs.append(inp_df)
        
        all_inputs = pd.concat(all_input_dfs)
            
        all_inputs.set_index("Key",inplace=True)
                
        rest_df = pd.DataFrame()
            
        for key in list(all_inputs.index):
            full_payfile_path = os.path.abspath(os.path.join(full_inputpath,"Payoffs","Payoffs_" + key + ".csv"))
            full_stratfile_path = os.path.abspath(os.path.join(full_inputpath,"Strategy","Strategy_" + key + ".csv"))
            pay_df = pd.read_csv(full_payfile_path,header=None,sep=" ",skiprows=1,dtype=str)
            strat_df = pd.read_csv(full_stratfile_path,header=None,sep=" ",dtype=str)
            
            existing_payoffs = flatten(pay_df.values)
            existing_strats = flatten(strat_df.values)
            
            existing_rest = existing_payoffs + existing_strats
            
            rest_df[key] = existing_rest
        
        all_inputs = pd.concat([all_inputs,rest_df.T],axis=1)
        all_inputs.columns = range(all_inputs.shape[1])  
    else:
        all_inputs = pd.DataFrame()
    
    return all_inputs


def setup_simulation(base, payoffs, pop_list_in, net_discount_list_in, init_cond_hawk_list_in, net_speed_list_in, net_tremble_list_in, strat_tremble_list_in, imitation_tremble_list_in, net_sym_list_in, memory_sym_list_in, total_weight, tmax_in, imitationRate_list_in, memory_list_in, average_comp_list_in, memory_type_in, game_name):
    """ 
    THE FOLLOWING SETS ALL INPUT PARAMETERS FOR MODEL
    """

    # Each element in these lists corresponds to one parameter point.  For payoff symmetry, set dd1s = dd2s, dh1s == dh2s. (Can do manually or just add an if statement).  Must have equal number of elements in all 4 lists!
    
    # For delight and dilemma
    num_strats_p1 = 2
    num_strats_p2 = 2
    
    # FIXED JUST FOR FILE NAMING
    game = "IM"

    # FOLDER WHERE INPUT FILES ARE STORED
    input_folder = game + "_Input"
    
    # Network and strategy symmetry (0 for asymmetric, 1 for symmetric)
    unzip_payoff = copy.deepcopy(payoffs)
    unzip_payoff = *unzip_payoff,
    unzip_payoff = flatten(unzip_payoff[0][0])
	
	# Parameters to string - leave this
    pop_str = "-".join(map(str, pop_list_in))
    nd_str = "-".join(map(str, net_discount_list_in))
    nls_str = "-".join(map(str, net_speed_list_in))
    net_tremble_str = "-".join(map(str, net_tremble_list_in))
    strat_tremble_str = "-".join(map(str, strat_tremble_list_in))
    im_tremble_str = "-".join(map(str, imitation_tremble_list_in))
    n_sym_str = "-".join(map(str, net_sym_list_in))
    m_sym_str = "-".join(map(str, memory_sym_list_in))
    ir_str = "-".join(map(str, imitationRate_list_in))
    mem_str = "-".join(map(str, memory_list_in))
    payoff_str = "-".join(map(str, unzip_payoff))
    init_cond_str = "-".join(map(str, init_cond_hawk_list_in))
    average_comp_str = "-".join(map(str, average_comp_list_in))
	

    time_str = str(tmax_in/1000000) + "M"
    if tmax_in < 1000000:
        time_str = tmax_in

    # Simulation description
    description_in = "Pop-"+ pop_str + "_Discount-" + nd_str + "_Tremble-" + net_tremble_str + "_NLS-" + nls_str + "_SymN-"+ n_sym_str + "_SymM-" + m_sym_str + "_ImRate-" + ir_str + "_Memory-" + mem_str + "_TMAX-" + str(time_str) + "_" + game_name + "_InitH-" + init_cond_str + "_P-" + payoff_str + "_AvgComp-" + average_comp_str + "_MType-" + memory_type_in
                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                    
    data_description = game + "_" + description_in

    input_subfolder = "Input_" + data_description

    full_inputpath = os.path.abspath(os.path.join(base_path,"..",input_folder,input_subfolder))

    if not os.path.exists(game + "_Output_Data"):
        os.makedirs(game + "_Output_Data")
    
    print(data_description)


    # Get all combinations of parameter lists above
    param_combos = list(product(payoffs,pop_list_in, init_cond_hawk_list_in, net_discount_list_in, net_speed_list_in, net_tremble_list_in, strat_tremble_list_in, imitation_tremble_list_in, 
	net_sym_list_in, memory_sym_list_in, imitationRate_list_in, memory_list_in, average_comp_list_in))

    input_params_list = []
    for ind,i in enumerate(param_combos):
        
        # Set local parameter variables
        p1_payoffs_in = i[0][0].flatten()
        p2_payoffs_in = i[0][1].flatten()
        payoffs_in = pd.concat([pd.DataFrame(p1_payoffs_in).T,pd.DataFrame(p2_payoffs_in).T])
        pop_in = i[1]
        
        num_hawks = round(i[2]/100. * pop_in)
        
        init_strategy_fill_in = np.ones(shape=(pop_in,1))
        hawk_inds = np.random.choice(pop_in,num_hawks,replace=False)
        init_strategy_fill_in[hawk_inds] = 0
        
        net_discount_in = i[3]
        net_learningspeed_in = i[4]
        net_tremble_in = i[5]
        strat_tremble_in = i[6]
        imitation_tremble_in = i[7]
        net_sym_in = i[8]
        memory_sym_in = i[9]
        imitationRate_in = i[10]
        memory_in = i[11]
        average_comp_in = i[12]

        if pop_in <= 50:
            file_lines = 20
        elif pop_in < 500:
            file_lines = 8
        else:
            file_lines = 1
			
        key = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(8))

        input_params = [base,pop_in,tmax_in,net_discount_in,net_learningspeed_in,memory_sym_in,net_sym_in,imitation_tremble_in,net_tremble_in,strat_tremble_in,imitationRate_in,memory_in,average_comp_in,memory_type_in,game,data_description,key]

        payoff_folder = os.path.abspath(os.path.join(full_inputpath,"Payoffs"))
        initw_folder = os.path.abspath(os.path.join(full_inputpath,"Strategy"))
        initnet_folder = os.path.abspath(os.path.join(full_inputpath,"Network"))
        
        payoff_path = os.path.abspath(os.path.join(payoff_folder,"Payoffs_" + key + ".csv"))
        initw_path = os.path.abspath(os.path.join(initw_folder,"Strategy_" + key + ".csv"))

        full_row = np.array(input_params)

        rest_df = flatten(payoffs_in.values) + flatten(init_strategy_fill_in)
    
        full_row = np.array(input_params[:-1] + rest_df)

    
        if not os.path.exists(payoff_folder):
            os.makedirs(payoff_folder)
            os.makedirs(initw_folder)
            
        print(key)
        
        pd.DataFrame([num_strats_p1,num_strats_p2]).T.to_csv(payoff_path,sep = " ", header=None,index=False)
            
        with open(payoff_path, 'a') as f:
            pd.DataFrame(payoffs_in).to_csv(f,sep = " ",header=None,index=False)
        
        pd.DataFrame(init_strategy_fill_in).to_csv(initw_path,sep= " ",header=None,index=False)
            
        if input_params not in input_params_list:
            input_params_list.append(input_params)

        
			
    if len(input_params_list) > 0:
        input_param_df = pd.DataFrame(input_params_list)    
        
        input_param_df.columns = full_input_list
        
        dataframes = []
        while len(input_param_df) > file_lines:
            top = input_param_df[:file_lines]
            dataframes.append(top)
            input_param_df = input_param_df[file_lines:]
        
        dataframes.append(input_param_df)

        if not os.path.exists(full_inputpath):
            os.makedirs(full_inputpath)

        for ind, frame in enumerate(dataframes):
            input_filename = "Input_" + str(ind) + ".conf" #"Input_" + data_description + "_" + str(ind) + ".conf" 
            full_conf_path = os.path.abspath(os.path.join(full_inputpath,input_filename))
            if os.path.exists(full_conf_path):
                with open(full_conf_path,'a') as f:
                    frame.to_csv(f,index=False,sep= " ",header=None)
            else:
                frame.to_csv(full_conf_path, index=False, sep = " ")

