/* The Network class Implementation (Network).cpp) */
#include "Network.h" // user-defined header in the same directory
#include <iostream>
#include <fstream>
#include <iterator>
#include <math.h>
#include <numeric>

// Constructor
// default values shall only be specified in the declaration,
// cannot be repeated in definition
Network::Network(int pop, float network_learning_speed, float network_discount, float strategy_tremble, float network_tremble, bool network_sym, int net_fill, float imitationRate, int memory, float imitation_tremble, bool memory_sym, bool average_comp, std::string memory_type){

    this->pop = pop;
	//this->net_fill = net_fill;
	double normalized_net_fill = 9.0/(pop-1);
    for(int i = 0; i < pop; i++){
        Agent A(i, network_learning_speed, network_discount, strategy_tremble, network_tremble, network_sym, imitationRate, memory, imitation_tremble, memory_sym, average_comp, memory_type);
        
        // Create friend weights vector
        std::vector<double> myfriends(pop);
        std::fill(myfriends.begin(),myfriends.end(), normalized_net_fill);
        myfriends.at(i) = 0; //sets agents own weight to 0
        A.setFriends(myfriends);
        
        A.updateAgent();
        
        agents.push_back(A);
    }
}

Network::Network(int pop, std::string strat_filepath, float network_learning_speed, float network_discount, float strategy_tremble, float network_tremble, bool network_sym, float imitationRate, int memory, float imitation_tremble, bool memory_sym, bool average_comp, std::string memory_type){
    
    this->pop = pop;
    
    double net_fill = 19.0/(pop-1);
    
    std::ifstream stratstream(strat_filepath);
    std::istream_iterator<double> startstrat(stratstream), endstrat;
    std::vector<double> strat_matrix(startstrat, endstrat);
    
    for(int i = 0; i < pop; i++){
        Agent A(i,strat_matrix,network_learning_speed, network_discount, strategy_tremble, network_tremble, network_sym, imitationRate, memory, imitation_tremble, memory_sym, average_comp, memory_type);
        
        // Create friend weights vector
        std::vector<double> myfriends(pop);
        std::fill(myfriends.begin(),myfriends.end(),net_fill);
        myfriends.at(i) = 0;
        A.setFriends(myfriends);
        
        A.updateAgent();
        
        agents.push_back(A);
        
        // for (auto i: agents.at(i).getStrats(0))
        //    std::cout << i << ' ';
    }
}
	
Agent& Network::GetAgent(std::vector<Agent>::size_type AgentNumber){
    return agents[AgentNumber];
}

void Network::AddAgent(const Agent& NewAgent){
    agents.push_back(NewAgent);
}

int Network::getPop(){
    return pop;
}

int Network::getNumCoop(){
    int numCoop = 0;
	int pop = getPop();
	for(int agent_num = 0; agent_num < pop; agent_num++){
		Agent currentAgent = GetAgent(agent_num);
		if (currentAgent.getStrategyProfile()){
			numCoop++;
		}
	}
	return numCoop;
}

std::vector<Agent> Network::getAgents(){
    return agents;
}
