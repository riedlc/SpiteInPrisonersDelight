/* The Agent class Implementation (Agent).cpp) */
#include "Network.h" // user-defined header in the same directory
#include <iostream>
#include <fstream>
#include <iterator>
#include <bitset>
#include <math.h>
#include <numeric>
#include <algorithm>
#include <vector>
#include <tr1/functional>

//Pure strategy vectors

// Constructor
// default values should only be specified in the declaration,
// cannot be repeated in definition
Agent::Agent(int agent_id, float network_learning_speed, float network_discount, float strategy_tremble, float network_tremble, bool network_sym, float imitationRate, int memory, float imitation_tremble, bool memory_sym, bool average_comp, std::string memory_type){
    this->agent_id = agent_id;
    this->network_learning_speed = network_learning_speed;
    this->network_discount = network_discount;
    this->strategy_tremble = strategy_tremble;
    this->network_tremble = network_tremble;
    this->network_sym = network_sym;
	this->imitationRate = imitationRate; 
	this->memory = memory;
	this->imitation_tremble = imitation_tremble;
	this->memory_sym = memory_sym;
    this->average_comp = average_comp;
    this->memory_type = memory_type;

	// This is the case of non-evolving networks
    if(network_learning_speed == 0){
        this->network_discount = 0;
    }
    	
	if (agent_id % 2 == 0){ //agent is given one of two pure strategies
		this->cur_pureCoop = false;
		this->new_pureCoop = cur_pureCoop;
	}else{
		this->cur_pureCoop = true;
		this->new_pureCoop = cur_pureCoop;
	}
}

Agent::Agent(int agent_id, std::vector<double> fill_values, float network_learning_speed, float network_discount, float strategy_tremble, float network_tremble, bool network_sym, float imitationRate, int memory, float imitation_tremble, bool memory_sym, bool average_comp, std::string memory_type){
    this->agent_id = agent_id;
    this->network_learning_speed = network_learning_speed;
    this->network_discount = network_discount;
    this->strategy_tremble = strategy_tremble;
    this->network_tremble = network_tremble;
    this->network_sym = network_sym;
    this->imitationRate = imitationRate; 
    this->memory = memory;
    this->imitation_tremble = imitation_tremble;
    this->memory_sym = memory_sym;
    this->average_comp = average_comp;
    this->memory_type = memory_type;
    
    if(this->network_learning_speed == 0){
        this->network_discount = 0;
    }
    
    this->cur_pureCoop = (bool) fill_values.at(agent_id);
	this->new_pureCoop = cur_pureCoop;
}


void Agent::discountNeighbors(){
    int pop = cur_friends.size();
    
    //Iterate over neighbors
    for(int nid = 0; nid < pop; nid++)
    {
        // Discount neighbors
        new_friends.at(nid) = new_friends.at(nid) * (1-network_discount);
    }
}

void Agent::setFriends(std::vector<double> friends){
    new_friends = friends;
}

void Agent::setCurFriends(std::vector<double> friends){
    cur_friends = friends;
}

std::vector<double> Agent::getFriends() const{
    return cur_friends;
}

void Agent::updateAgent(){
    cur_friends = new_friends;
    
    if(memory_type == "round"){
        if(!this_time_payoff.empty()){
            if(memory_p1_payoff.size() < memory){
                memory_p1_payoff.push_back(this_time_payoff);
                memory_p2_payoff.push_back(this_time_payoff);
            }else{
                memory_p1_payoff.erase(memory_p1_payoff.begin());
                memory_p2_payoff.erase(memory_p2_payoff.begin());
                memory_p1_payoff.push_back(this_time_payoff);
                memory_p2_payoff.push_back(this_time_payoff);
            }
            
            this_time_payoff.clear();
        }
    }
        
        
    memory_payoff_sum = 0;
    total_interactions = 0;
    
    for(int i = 0; i < memory_p1_payoff.size(); i++){
        for(int j = 0; j < memory_p1_payoff.at(i).size(); j++){
            memory_payoff_sum += memory_p1_payoff.at(i).at(j);
            total_interactions++;
        }
    }
    
}

int Agent::chooseFriend(UGenerator rng, std::vector<int> temp_agent_seq){
   
    int friend_ind = -1; // Flag (goes >= 0 as the index) for when the neighbor is picked
    int nid;
    float rand_tremble = rng();
    
    int pop = cur_friends.size();

    // If agent doesn't make an error
    if(rand_tremble > network_tremble){
		//creates pop sized vector to store accumulating sum
        std::vector<double> sum_vec(pop);
        //float sum_vec[pop];
        
        std::partial_sum(cur_friends.begin(), cur_friends.end(), sum_vec.begin());
        
        double interaction_random_draw = rng() * sum_vec[pop-1];
        
        //Iterate over neighbors
        for(nid = 0; nid < pop; nid++)
        {
            // Add viable neighbor
            // If it's time to copy
            
            if(friend_ind == -1 && interaction_random_draw <= sum_vec.at(nid) && nid != agent_id){
                friend_ind = nid; // This is agents partner
            }
            // Discount neighbors
            new_friends.at(nid) = new_friends.at(nid) * (1-network_discount);
        }
    }else{ // If agent makes an error
        for(nid = 0; nid < pop; nid++)
        {
            // Discount all neighbors
            new_friends.at(nid) = new_friends.at(nid) * (1-network_discount);
        }
        // Choose random neighbor
        int temp_friend_ind = (int) (rng() * (pop-1));
        friend_ind = temp_agent_seq.at(temp_friend_ind);
        
    }
    currentFriend = friend_ind;
    
    return friend_ind;
}

int Agent::chooseImitationPartner(UGenerator rng, std::vector<int> temp_agent_seq, std::vector<double> norm_adj){
	int friend_ind = -1; // Flag (goes >= 0 as the index) for when the neighbor is picked
    int nid;
    int pop = cur_friends.size();
	
	//if boolean_input else call chooseFriend(), but dont discount!!!
	int agent_id = getID();
	std::vector<double> sum_vec(pop);
	sum_vec.at(0) = norm_adj.at(agent_id*pop) + norm_adj.at(agent_id);
	for(int sum_vec_ind = 1; sum_vec_ind < pop; sum_vec_ind++){
		sum_vec.at(sum_vec_ind) = norm_adj.at(agent_id*pop + sum_vec_ind) + norm_adj.at(sum_vec_ind*pop + agent_id) + sum_vec.at(sum_vec_ind - 1);
	}
	
	double interaction_random_draw = rng() * sum_vec[pop-1];
        
	//Iterate over neighbors
	for(nid = 0; nid < pop; nid++){
		if(friend_ind == -1 && interaction_random_draw <= sum_vec.at(nid) && nid != agent_id){
			friend_ind = nid; // This is agents partner
		}
	}	
	return friend_ind;
}

void Agent::chooseStrategy(UGenerator rng, bool cur_pureCoop){
    
    int strat_draw;
    
    double tremble_strat_draw;
    double tremble_draw = rng();
	
    if(tremble_draw < strategy_tremble){
		//If user makes strategy error, choose randomly
        tremble_strat_draw = rng();
        strat_draw = (int) (tremble_strat_draw);
    }else{
		strat_draw = cur_pureCoop;
    }
    
    currentStrategy = strat_draw;
}

void Agent::setCurrentPayoff(double currentPayoff){
    this->currentPayoff = currentPayoff;
}

void Agent::recordInteraction(double myPayoff, double oppPayoff){
    past_p1_payoff = myPayoff; //Change to vector of given size, average, save to variable defined in network.h
    past_p2_payoff = oppPayoff; // what would this be used for?
    last_visit = currentFriend;
}

// Setters
void Agent::setNetworkLearning(float network_learning_speed){
    this->network_learning_speed = network_learning_speed;
}

void Agent::setNetworkDiscount(float network_discount){
    this->network_discount = network_discount;
}

void Agent::setStrategyTremble(float strategy_tremble){
    this->strategy_tremble = strategy_tremble;
}
void Agent::setNetworkTremble(float network_tremble){
    this->network_tremble = network_tremble;
}

void Agent::setNetworkSym(bool network_sym){
    this->network_sym = network_sym;
}

void Agent::addNetworkPayoff(){
    new_friends.at(currentFriend) = new_friends.at(currentFriend) + currentPayoff * network_learning_speed;
}

void Agent::setCurrentFriend(int friend_id){
    currentFriend = friend_id;
}

//Getters
int Agent::getID(){
    return agent_id;
}

float Agent::getNetworkLearning(){
    return network_learning_speed;
}

float Agent::getNetworkDiscount(){
    return network_discount;
}

float Agent::getStrategyTremble(){
    return strategy_tremble;
}

float Agent::getNetworkTremble(){
    return network_tremble;
}

bool Agent::getNetworkSym(){
    return network_sym;
}

bool Agent::getMemorySym(){
    return memory_sym;
}

int Agent::getCurrentStrategy(){
    return currentStrategy;
}

double Agent::getPastmyPayoff(){
    return past_p1_payoff;
}

double Agent::getPastoppPayoff(){
    return past_p2_payoff;
}

int Agent::getPastVisitPartner(){
    return last_visit;
}
double Agent::getCurrentPayoff(){
    return currentPayoff;
}

void Agent::chooseToImitate(UGenerator rng, Agent possibleRoleModelAgent){
    double myCompScore;
    double friendCompScore;
    if(average_comp == 1){
        myCompScore = getAveragePayoff();
        friendCompScore = possibleRoleModelAgent.getAveragePayoff();
    }else{
        myCompScore = getTotalPayoff();
        friendCompScore = possibleRoleModelAgent.getTotalPayoff();
    }
	double imitate_draw = rng(); 
	double strat_error = rng();
	if(imitate_draw < imitationRate){
		if(strat_error > imitation_tremble){
			if(friendCompScore > myCompScore){ 
				setNextStrategyProfile(possibleRoleModelAgent.getStrategyProfile());
				// reset memory
				memory_p1_payoff.clear();
				memory_p2_payoff.clear();
				//roleModel = possibleRoleModel; This could be useful to track
			}
		}else{ // randomly pick new strategy
			if(strat_error < 0.5){
				setNextStrategyProfile(false);
			}else{
				setNextStrategyProfile(true);
			}
			// reset memory
			memory_p1_payoff.clear();
			memory_p2_payoff.clear();
		}
	}
}
			
bool Agent::getStrategyProfile(){
	return cur_pureCoop;
}

bool Agent::getNextStrategyProfile(){
	return new_pureCoop;
}

void Agent::setStrategyProfile(bool newPureStrat){
	this->cur_pureCoop = newPureStrat;		
}

void Agent::setNextStrategyProfile(bool newPureStrat){
	this->new_pureCoop = newPureStrat;		
}

double Agent::getAveragePayoff(){
    return memory_payoff_sum/total_interactions;
}

double Agent::getTotalPayoff(){
    return memory_payoff_sum;
}

void Agent::recordMemory(double myPayoff){
    if(memory_type == "round"){
        roundMemory(myPayoff);
    }else if(memory_type == "interaction"){
        interactionMemory(myPayoff);
    }
}

void Agent::roundMemory(double myPayoff){
    this_time_payoff.push_back(myPayoff);
	
    last_visit = currentFriend;
}

void Agent::interactionMemory(double myPayoff){
    
    if(memory_p1_payoff.empty()){
        std::vector<double> initfiller;
        memory_p1_payoff.push_back(initfiller);
        memory_p2_payoff.push_back(initfiller);
    }
    if(memory_p1_payoff.at(0).size() > memory){
        memory_p1_payoff.at(0).erase(memory_p1_payoff.at(0).begin());
        memory_p2_payoff.at(0).erase(memory_p2_payoff.at(0).begin());

        memory_p1_payoff.at(0).push_back(myPayoff);
        memory_p2_payoff.at(0).push_back(myPayoff);
    }else{
        memory_p1_payoff.at(0).push_back(myPayoff);
        memory_p2_payoff.at(0).push_back(myPayoff);
    }
    
    last_visit = currentFriend;
}
