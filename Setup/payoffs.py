import numpy as np

def get_payoffs(game="Delight"):
    p1_payoffs_list_in = []
    p2_payoffs_list_in = []

    for c in [0.14285714285]: 
        b = 1 - c
        
        if game == "Delight":
            user_p1_payoff = [[0,b],[c,b+c]]
            user_p2_payoff = [[0,c],[b,b+c]]
        elif game == "Dilemma":
            user_p1_payoff = [[c,b+c],[0,b]]
            user_p2_payoff = [[c,0],[b+c,b]]

        this_payoff_p1 = np.array(user_p1_payoff)
        this_payoff_p2 = np.array(user_p2_payoff)

        p1_payoffs_list_in.append(this_payoff_p1)
        p2_payoffs_list_in.append(this_payoff_p2)

    return zip(p1_payoffs_list_in,p2_payoffs_list_in)

