/* The Network class Header (Network.h) */
#include <stdio.h>
#include <string>   // using string
#include <vector>
#include <bitset>
#include <fstream>
#include <memory>
#include <iostream>
#include <cmath>
#include <boost/range/numeric.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>

typedef boost::mt19937 Engine;
typedef boost::uniform_real<double> UDistribution;
typedef boost::variate_generator< Engine &, UDistribution > UGenerator;

template<typename ... Args>
std::string string_format(const std::string& format, Args ... args){
    size_t size = 1 + std::snprintf(nullptr, 0, format.c_str(), args ...);
    std::unique_ptr<char[]> buf(new char[size]);
    snprintf(buf.get(), size, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size);
}

// Random number generator
struct MersenneRNG {
    MersenneRNG() : dist(0.0, 1.0), rng(eng, dist) {}
    
    Engine eng;
    UDistribution dist;
    UGenerator rng;
};

//Agent Class

class Agent{
    private:
        // Neighbors with weights
        std::vector<double> cur_friends;
        std::vector<double> new_friends;
    
        // Strategies with weights (vector of 2 size 2 vectors)
        //std::vector<std::vector<double>> cur_strategy_profile;
        //std::vector<std::vector<double>> new_strategy_profile;
    
        // Agent id
        int agent_id;
    
        // Track last interaction
        double past_p1_payoff;
        double past_p2_payoff;
		
		// memory of interactions
        std::vector<std::vector<double>> memory_p1_payoff;
		std::vector<std::vector<double>> memory_p2_payoff;
    
        std::vector<double> this_time_payoff;  
    
        double memory_payoff_sum;
        int total_interactions;
    
    
        // Last strategy played
        int currentStrategy;
        // Last payoff earned
        double currentPayoff;
        
		// Last friend visited
        int currentFriend;
        int last_visit;
    
        // Learning Speeds
        float network_learning_speed;
        
        // Past Discounting (Memory)
        float network_discount;
        
        // Tremble (Error rate)
        float strategy_tremble;
        float network_tremble;
		float imitation_tremble;
		float imitationRate;
        
        // Symmetry (For network: are directed network links (in/out) correlated).
        bool network_sym;
		bool memory_sym;
		
		//number of previous interaction payoffs considered
		int memory;
    
        //Average or Total comparison imitation (if 1 average, else total)
        bool average_comp;
		
		//Current strategy
		bool cur_pureCoop;
		bool new_pureCoop;
    
        std::string memory_type;
		
    
    public:																																																																												
        Agent(const int agent_id, float network_learning_speed = 1, float network_discount = 0.01, float strategy_tremble = 0.01, float network_tremble = 0.01, bool network_sym = 0, float imitationRate = 0.25, int memory = 5, float imitation_tremble = 0.01, bool memory_sym = 1, bool average_comp = 1, std::string memory_type = "round"); 
        Agent(int agent_id, std::vector<double> fill_values, float network_learning_speed = 1, float network_discount = 0.01, float strategy_tremble = 0.01, float network_tremble = 0.01, bool network_sym = 0, float imitationRate = 0.01, int memory = 1, float imitation_tremble = 0.01, bool memory_sym = 1, bool average_comp = 1, std::string memory_type = "round");
    
        // Get Agent ID (shouldn't be necessary)
        int getID();
        
        // Get Agent strategy profile for given strategy set (defined by strat_num)
        void discountStrategy(int strat_num);
        void discountNeighbors();
        
        // Set Agent friends from network
        void setFriends(std::vector<double> friends);
		void setCurFriends(std::vector<double> friends);
        std::vector<double> getFriends() const;
        
        void updateAgent();
    
        int chooseFriend(UGenerator rng,std::vector<int> temp_agent_seq);
        int getCurrentFriend();
        void setCurrentFriend(int friend_id);
    
        void chooseStrategy(UGenerator rng, bool pureCoop);
		int getCurrentStrategy();
    
        void setCurrentPayoff(double currentPayoff);
        double getCurrentPayoff();

        void trackPayoffs(double last_payoff);
    
        void addStrategyPayoff(int send_rec);
    
        void addNetworkPayoff();
    
        void recordInteraction(double myPayoff, double oppPayoff);
        
        double getPastmyPayoff();
        double getPastoppPayoff();
        int getPastVisitPartner();
        
        void setNetworkLearning(float network_learning_speed);
        float getNetworkLearning();
        
        void setNetworkDiscount(float network_discount);
        float getNetworkDiscount();
        
        void setStrategyTremble(float strategy_tremble);
        float getStrategyTremble();
        
        void setNetworkTremble(float network_tremble);
        float getNetworkTremble();
        
        void setNetworkSym(bool network_sym);
        bool getNetworkSym();
		
		void chooseToImitate(UGenerator rng, Agent possibleRoleModelAgent);

        void recordMemory(double myPayoff);
        void roundMemory(double myPayoff);
		void interactionMemory(double myPayoff);
    
		int chooseImitationPartner(UGenerator rng, std::vector<int> temp_agent_seq, std::vector<double> norm_adj);
		
		bool getStrategyProfile();
		bool getNextStrategyProfile();
		void setStrategyProfile(bool newPureStrat);
		void setNextStrategyProfile(bool newPureStrat);
		
		double getAveragePayoff();
        double getTotalPayoff();

		bool getMemorySym();
		std::vector<double> getStrats(bool pureCoop);
};

//Network Class
class Network{
    private:
        // Population of network
        int pop;
        
        // Vector of Agents
        std::vector<Agent> agents;
		
		//Initial link weights
		double net_fill;
    
    public:
        Network(int pop = 20, float network_learning_speed = 1, float network_discount = 0.01, float strategy_tremble = 0.01, float network_tremble = 0.01, bool network_sym = 0, int net_fill = 1, float imitationRate = 0.25, int memory = 5, float imitation_tremble = 0.01, bool memory_sym = 1, bool average_comp = 1, std::string memory_type = "round");
    
        Network(int pop, std::string strat_filepath, float network_learning_speed = 1, float network_discount = 0.01, float strategy_tremble = 0.01, float network_tremble = 0.01, bool network_sym = 0, float imitationRate = 0.01, int memory = 1, float imitation_tremble = 0.01, bool memory_sym = 1, bool average_comp = 1, std::string memory_type = "round");
    
        std::vector<int> agent_seq;
    
        int getPop();
        std::vector<Agent> getAgents();
    
        Agent& GetAgent(std::vector<Agent>::size_type ElementNumber);
        void AddAgent(const Agent& NewAgent);
		
		int getNumCoop();

};

class Game{
    private:
		//type performance trackers
		std::vector<double> defect_payoffs;
		std::vector<double> coop_payoffs;
	
        std::string gameName;
        double base_payoff;
        std::vector<std::vector<double>> gamePayoffs;
        
    public:
    
        Game(std::vector<std::vector<double>> gamePayoffs = {{0,1,0.2,0.6},{0,0.2,1,0.6}}, std::string gameName = "HD", double base_payoff = 0.0001);
    
        Game(std::string payoff_filepath, std::string gameName = "IM", double base_payoff = 0.0001);
    
        std::string getName();
        void setPayoffs(std::vector<std::vector<double>> gamePayoffs);
        std::vector<std::vector<double>> getPayoffs();
        double getBasePayoff();
        void playGame(Agent &currentAgent, Agent &friendAgent);
		
		std::vector<double> getCoopPayoffs();
		std::vector<double> getDefectPayoffs();
		void clearPayoffs(); 
};

class Environment{
    private:
        int size;
    
};

struct SimTracking{
    char key[20];
    const char out_folder_complete_path[100] = "/Users/bobloblaw/Dropbox/Research/Evolutionary_Modeling";
    int current_seed;
    
	std::string out_avg_payoff_file;
    std::string out_network_file;
    std::string out_stats_file;
    std::string out_p1strat_file;
    std::string out_p2strat_file;
    std::string out_partners;
    std::string out_p1_payoffs;
    std::string out_p2_payoffs;
	
	
    std::vector<std::vector<double>> player_avg_payoff_t;
    std::vector<std::vector<double>> player_strategies_p1_t;
    std::vector<std::vector<double>> player_strategies_p2_t;
    std::vector<std::vector<double>> network_weights_t;
    std::vector<std::vector<double>> player_p1_payoffs_t;
    std::vector<std::vector<double>> player_p2_payoffs_t;
    std::vector<std::vector<int>> player_partners_t;
    std::vector<double> strategy_correlation_t;
    std::vector<std::vector<double>> prop_interactions_t;
    std::vector<std::vector<double>> strategy_mean_t;
    std::vector<std::vector<double>> strategy_variance_t;
    std::vector<double> instrength_variance_t;
    std::vector<int> times_tracked;
    
    
    int max_time;
    
    std::string out_file_evostats;
    
    void init_Trackers(){
        for(int i = 0; i < 1000; i++){
            times_tracked.push_back(i);
        }        
        for(int i = 1000; i < 10000; i+= 1000){
            times_tracked.push_back(i);
        }
        for(int i = 10000; i < 1000001; i+=10000){
            times_tracked.push_back(i);
        }
        
    }
    
    void updateData(Network &net, int time_t){
        int pop = net.getPop();
        
        std::vector<double> player_strategies_p1;
        player_strategies_p1.reserve(pop*2);
        
        std::vector<double> prop_interactions(5,0.0);
        
        std::vector<double> network_weights;
        network_weights.reserve(pop);
		
        std::vector<double> player_avg_payoff;
        player_avg_payoff.reserve(pop);
        
        std::vector<double> player_p1_payoffs;
        player_p1_payoffs.reserve(pop);
        
        std::vector<double> player_p2_payoffs;
        player_p2_payoffs.reserve(pop);
        
        std::vector<int> player_partners;
        player_partners.reserve(pop);
		
		std::vector<std::vector<double>> pureStrategies;
		std::vector<double> firstrow = {1, 0};
		pureStrategies.push_back(firstrow);
		std::vector<double> secondrow = {0, 1};
		pureStrategies.push_back(secondrow);
        
        for(int agent_num = 0; agent_num < pop; agent_num++){
            
            Agent &curAgent = net.GetAgent(agent_num);
            curAgent.updateAgent();
			
			// Update avg payoff tracker
			player_avg_payoff.push_back(curAgent.getAveragePayoff());
            
            // Update strategy tracker	
            std::vector<double> p1_strats = pureStrategies.at(curAgent.getStrategyProfile());
            double p1_strat_sum = std::accumulate(p1_strats.begin(),p1_strats.end(), 0.0);
            std::transform(p1_strats.begin(), p1_strats.end(), p1_strats.begin(),
                           std::bind2nd(std::multiplies<double>(), 1.0/p1_strat_sum));
            
            player_strategies_p1.insert(player_strategies_p1.end(), p1_strats.begin(), p1_strats.end());
            
            // Update network tracker
            std::vector<double> friends = curAgent.getFriends();
            double net_sum = std::accumulate(friends.begin(),friends.end(), 0.0);
            
            std::transform(friends.begin(), friends.end(), friends.begin(), std::bind2nd(std::multiplies<double>(), 1.0/net_sum));
            
            network_weights.insert(network_weights.end(), friends.begin(), friends.end());
            
            player_p1_payoffs.push_back(curAgent.getPastmyPayoff());
            player_p2_payoffs.push_back(curAgent.getPastoppPayoff());
            player_partners.push_back(curAgent.getPastVisitPartner());
        }
        int strat_1;
		int strat_2;
        
        
        // Get total normalized interaction weights between each pair of agents
        prop_interactions.at(0) = time_t;
        for(int pop_ind_1 = 0; pop_ind_1 < pop; pop_ind_1++){
            for(int pop_ind_2 = 0; pop_ind_2 < pop; pop_ind_2++){
                for(int strat_ind = 0; strat_ind < 4; strat_ind++){
                    strat_1 = strat_ind/2;
					strat_2 = strat_ind%2;
                    
					prop_interactions.at(strat_ind+1) += (network_weights.at(pop_ind_1 * pop + pop_ind_2) * player_strategies_p1.at(pop_ind_1 * 2 + strat_1) * player_strategies_p1.at(pop_ind_2 * 2 + strat_2))/pop;                
				}
            }
        }
        prop_interactions_t.push_back(prop_interactions);
        player_strategies_p1_t.push_back(player_strategies_p1);
		player_avg_payoff_t.push_back(player_avg_payoff);
        network_weights_t.push_back(network_weights);
        player_p1_payoffs_t.push_back(player_p1_payoffs);
        player_p2_payoffs_t.push_back(player_p2_payoffs);
        player_partners_t.push_back(player_partners);
    };
    
    void initAgentList(Network &net){
        int pop = net.getPop();
        for(int agent_num = 0; agent_num < pop; agent_num++){
            net.agent_seq.push_back(agent_num);
        }
    };
};
