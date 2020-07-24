import pickle
import time
import numpy as np
from statistics import mean
import os
import pandas as pd
import multiprocessing as mp


def get_corrs(key):
    
    Pop=20
    times=list(range(0,1000)) + list(range(1000, 10000, 1000)) + list(range(10000, 1000000, 10000))
    output_folder='IM_Output_Data//Output_IM_Pop-20_Discount-0.01_Tremble-0.01_NLS-1_SymN-0_SymM-1_ImRate-0.01_Memory-1_TMAX-1.0M_Delight_InitH-50_P-0.0-0.8571428571499999-0.14285714285-1.0_AvgComp-1_MType-round'
    
    all_dfs = []
    files = [a for a in os.listdir(output_folder)if key in a]
    print(len(files))
    mat_files = [a for a in files if "Weight" in a]
    strat_files = [a for a in files if "Strategy" in a]
    for ind in range(len(mat_files)): #len(mat_files)
        mat_file = mat_files[ind]
        seed = mat_file.split(".")[-2].split("_")[-1]
        strat_file = [a for a in strat_files if seed + ".csv" in a][0]
        
        all_mats = pd.read_csv(os.path.join(output_folder,mat_file),header=None)
        try:
            strats = pd.read_csv(os.path.join(output_folder,strat_file),header=None)
        except:
            print(os.path.join(output_folder,strat_file)) 		
        dfs = []
        for time in range(len(times)):
            r_dict = {i:0 for i in range(pop)}
            try:
                this_mat = all_mats.iloc[time].values.reshape(pop,pop)
            except:
                print(mat_file)
                print(time)
            these_strats = strats.iloc[time][1::2].values
            prop_defect = 1 - these_strats.mean()
            for i in range(pop):
                for j in range(pop):
                    if i != j:
                        if these_strats[i] == these_strats[j]:
                            r_dict[i] += (this_mat[i,j] + this_mat[j,i])/(this_mat[:,i].sum() + this_mat[i,:].sum())
                r_dict[i] -= (sum(these_strats == these_strats[i]))/pop
            corrs = pd.DataFrame([these_strats,r_dict.values()])
            corrs = corrs.T.groupby(0).mean().T
            corrs["PropDef"] = [prop_defect]
            corrs["All"] = [np.mean(list(r_dict.values()))]
            dfs.append(corrs)
        
        corr_time = pd.concat(dfs, sort=True)
        corr_time.reset_index(drop=True,inplace=True)
        corr_time.columns = ["Defector","Cooperator","PropDef","ALL"]
        all_dfs.append(corr_time)
    
    concat_df = pd.concat(all_dfs)
    concat_means = concat_df.groupby(concat_df.index).mean()
    concat_means["Time"] = times

    return concat_means, all_dfs
    
    
#Inputs
num_cores = mp.cpu_count()
pool = mp.Pool(num_cores)
print(num_cores)

inputs = ['E9QGMZBX'] # list of all keys to generate correlation output for

results = pool.map(get_corrs, inputs)
# save output
with open('cor_pop_20', 'wb') as file:
    pickle.dump(results, file)

## Non parallel function call
# cor_dilemma_4 = get_corrs('TAWPNCSX')   
# #save corr output
# with open('cor_dilemma_4', 'wb') as file:
    # pickle.dump(cor_dilemma_4, file) 
    

    
	
