// If compiled with -fopenmp, include omp (for multithreading)
#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_thread_num() 0
    #define omp_get_max_threads() 1
#endif

/* A test driver for the Network class (NetworkDriver.cpp) */
#include <stdio.h>
#include <sys/stat.h>
#include <iostream>
#include <memory>
#include <fstream>
#include <iterator>
#include <numeric>
#include <string>
#include <algorithm>
#include <bitset>
#include <iomanip>
#include <vector>
#include <boost/lexical_cast.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>
#include "Network.h"   // using Network class

typedef boost::mt19937 Engine;
typedef boost::uniform_real<double> UDistribution;
typedef boost::variate_generator< Engine &, UDistribution > UGenerator;
typedef boost::normal_distribution<double> NDistribution;   // Normal Distribution
typedef boost::variate_generator<Engine &, NDistribution > NGenerator;    // Variate generator

// This is for the output formatting, prints one line at a time comma separated except for the end of the line
template <typename Stream, typename Iter, typename Infix>
inline Stream& infix (Stream &os, Iter from, Iter to, Infix infix_) {
    if (from == to) return os;
    os << *from;
    for (++from; from!=to; ++from) {
        os << infix_ << *from;
    }
    return os;
}

template <typename Stream, typename Iter>
inline Stream& comma_seperated (Stream &os, Iter from, Iter to) {
    return infix (os, from, to, ", ");
}

void run_model(UGenerator rng, Game &g, SimTracking &tracking_vars, Network &net);
void run_timestep(UGenerator rng, Game &g, SimTracking &tracking_vars, Network &net, int t, std::vector<int> agent_seq);
bool is_number(const std::string& s);
bool file_exists (const std::string& name);
void consider_imitation(UGenerator rng, Network &net);

int main(int argc, char *argv[]){
    
    // Command line arguments at runtime
    char* inputFolder = argv[1]; // Name of input file (decide to include folder here)
    char* inputFileNumber = argv[2];  // Input file number
    int thread_ct = atoi(argv[3]); // Number of threads to run in parallel if fopenmp
    int num_seeds = atoi(argv[4]); // Number of seeds to run
    int base_seed = atoi(argv[5]); // Starting at this seed
    int start_key = atoi(argv[6]); // Starting at this key index
	
	std::cout << "thread_ct: " << thread_ct << std::endl;
    std::cout << "num seeds: " << num_seeds << std::endl;
	std::cout << "base_seed: " << base_seed << std::endl;
	std::cout << "start_key: " << start_key << std::endl;
    // Read in seeds
    //////////////////////////////////////////////////////

    std::ifstream seedfile("ESD_Seeds_All_Ordered.csv"); //../Helpers/ESD_Seeds_All_Ordered.csv
    
    if (seedfile.fail()){
        std::cerr << "Error: " << strerror(errno) << "\n";
    }
    
    std::vector<int> seeds;
    
    int seed_value;
    
    while (seedfile >> seed_value) {
        seeds.push_back(seed_value);
    }
    
    ////////////////////////////////////////////////////// End read


   // Read input file line by line (discard header)
    std::string full_input_folder = string_format("IM_Input/Input_IM_%s",inputFolder);
    std::string inputFilename = string_format("%s/Input_%s.conf",full_input_folder.c_str(),inputFileNumber);//string_format("%s/Input_IM_%s_%s.conf",full_input_folder.c_str(),inputFolder,inputFileNumber);
    std::ifstream in(inputFilename);
    
    std::cout << inputFilename << "\n";

    if (in.fail()){
        std::cerr << "Error: " << strerror(errno) << "\n";
    }

    std::vector<std::vector<std::string>> all_inputs;
    
    std::string header;
    getline(in, header);
    
    if (in) {
        std::string line;
        
        while (std::getline(in, line)) {
            all_inputs.push_back(std::vector<std::string>());
            
            // Break down the row into column values
            std::stringstream split(line); //Creates iterator
            std::string value;
            
			//
            while (split >> value) //Loops while iterator has values to assign
                all_inputs.back().push_back(value);
        }
    }
    ////////////////////////////////////////////////////// End read
	
    
    // Number of keys is number of lines, or size of all inputs first dimension
    int num_keys = (int) all_inputs.size();
    
    #ifdef _OPENMP
    {
        #pragma omp parallel for num_threads(thread_ct) //start thread_ct parallel for loops (each is one simulation)
    #endif
        for(int seeded_run = num_seeds*start_key; seeded_run < num_keys * num_seeds; seeded_run++)
        {
            // Set key index and seed index
            int run_num = seeded_run/num_seeds;
            int seed_ind = seeded_run%num_seeds;
            
            // Locate seed at the relevant index
            int this_seed = seeds.at(base_seed + seed_ind);
            
            // Grab input vector and set simulation parameters
            ////////////////////////////////////////////
            
            std::vector<std::string> these_inputs = all_inputs.at(run_num);
            
            double base_in = std::stod(these_inputs.at(0));
            int net_pop = std::stoi(these_inputs.at(1));    
            int tmax_in = std::stoi(these_inputs.at(2));
            float netdiscount_in = std::stof(these_inputs.at(3));
            float netlearningspeed_in = std::stof(these_inputs.at(4));
            bool memorysymmetric_in = boost::lexical_cast<bool>(these_inputs.at(5));
            bool netsymmetric_in = boost::lexical_cast<bool>(these_inputs.at(6));
            float imitation_tremble = std::stof(these_inputs.at(7));
            float nettremble_in = std::stof(these_inputs.at(8));
            float strattremble_in = std::stof(these_inputs.at(9));
			float imitationRate = std::stof(these_inputs.at(10));
			int memory = std::stoi(these_inputs.at(11));
            int average_comp_in = std::stof(these_inputs.at(12));
            std::string memory_type_in = these_inputs.at(13);
			std::string game_in = these_inputs.at(14);
            std::string outputDesc = these_inputs.at(15);
            std::string key = these_inputs.at(16);
        
            std::string mainOutputFolder = string_format("%s_Output_Data",game_in.c_str());
            
            std::string outputFolder = string_format("%s_Output_Data/Output_%s",game_in.c_str(),outputDesc.c_str());
            
            std::string strat_file = string_format("%s/Strategy/Strategy_%s.csv",full_input_folder.c_str(),key.c_str());
            
            struct stat st = {0};
            
            // If output directory doesn't exist, create it
            if(stat(mainOutputFolder.c_str(), &st) == -1){
                mkdir(mainOutputFolder.c_str(), 0700);
            }
    
            // If output directory doesn't exist, create it
            if(stat(outputFolder.c_str(), &st) == -1){
                mkdir(outputFolder.c_str(), 0700);
            }
            
            ////////////////////////////////////////////
            
            
            // Initialize tracking variables and output files
            SimTracking tracking_vars;
			
            tracking_vars.out_avg_payoff_file = string_format("%s/%s_AvgPayoffs_%s_%d.csv",outputFolder.c_str(),game_in.c_str(),key.c_str(), this_seed);
            tracking_vars.out_network_file = string_format("%s/%s_Weights_%s_%d.csv",outputFolder.c_str(),game_in.c_str(),key.c_str(), this_seed);
            tracking_vars.out_stats_file = string_format("%s/%s_EvoStats_%s_%d.csv",outputFolder.c_str(),game_in.c_str(),key.c_str(), this_seed);
            tracking_vars.out_p1strat_file = string_format("%s/%s_Strategy_%s_%d.csv",outputFolder.c_str(),game_in.c_str(),key.c_str(), this_seed);
            tracking_vars.out_p1_payoffs = string_format("%s/%s_P1Payoffs_%s_%d.csv",outputFolder.c_str(),game_in.c_str(),key.c_str(), this_seed);
            tracking_vars.out_p2_payoffs = string_format("%s/%s_P2Payoffs_%s_%d.csv",outputFolder.c_str(),game_in.c_str(),key.c_str(), this_seed);
            tracking_vars.out_partners = string_format("%s/%s_Partners_%s_%d.csv",outputFolder.c_str(),game_in.c_str(),key.c_str(), this_seed);
            
            // Check if output files exist (if they already exist, why rerun?)
            if(!(file_exists(tracking_vars.out_network_file) && file_exists(tracking_vars.out_stats_file) && file_exists(tracking_vars.out_p1strat_file))){
                Engine eng(this_seed);
                UDistribution udst(0.0, 1.0);
                UGenerator rng(eng, udst);
				
				
                // Construct a Game
                std::string payoff_filename = string_format("%s/Payoffs/Payoffs_%s.csv",full_input_folder.c_str(),key.c_str());
            
				Game g(payoff_filename, game_in.c_str(), base_in);
                
                //Construct a network of Agents
                Network net(net_pop, strat_file, netlearningspeed_in, netdiscount_in, strattremble_in, nettremble_in, netsymmetric_in, imitationRate, memory, imitation_tremble, memorysymmetric_in, average_comp_in, memory_type_in);
                
                // Initialize tracking variables for timestep 0 (initial conditions)
                tracking_vars.init_Trackers();
                tracking_vars.updateData(net, 0);
                tracking_vars.max_time = tmax_in;
                
            
            
                ////////////////////////////////////////////

                
                // Run single simulation
                run_model(rng, g, tracking_vars, net);
                

                // Output tracking data
                /////////////////////////////////////////////
                std::ofstream avg_out(tracking_vars.out_avg_payoff_file.c_str());
                            
                avg_out << std::setprecision(4);

                for(size_t time_i = 0; time_i < tracking_vars.player_avg_payoff_t.size(); time_i++){
                    comma_seperated(avg_out, tracking_vars.player_avg_payoff_t.at(time_i).begin(), tracking_vars.player_avg_payoff_t.at(time_i).end()) << std::endl;
                }
				
				std::ofstream net_out(tracking_vars.out_network_file.c_str());
                            
                net_out << std::setprecision(4);

                for(size_t time_i = 0; time_i < tracking_vars.network_weights_t.size(); time_i++){
                    comma_seperated(net_out, tracking_vars.network_weights_t.at(time_i).begin(), tracking_vars.network_weights_t.at(time_i).end()) << std::endl;
                }
                

                std::ofstream p1_strat_out(tracking_vars.out_p1strat_file.c_str());
                
                p1_strat_out << std::setprecision(3);
                
                for(size_t time_i = 0; time_i < tracking_vars.player_strategies_p1_t.size(); time_i++){
                    comma_seperated(p1_strat_out, tracking_vars.player_strategies_p1_t.at(time_i).begin(), tracking_vars.player_strategies_p1_t.at(time_i).end()) << std::endl;
                }
                
				std::ofstream stats_out(tracking_vars.out_stats_file.c_str());
                
                stats_out << std::setprecision(3);
                
                for(size_t time_i = 0; time_i < tracking_vars.prop_interactions_t.size(); time_i++){
                    comma_seperated(stats_out, tracking_vars.prop_interactions_t.at(time_i).begin(), tracking_vars.prop_interactions_t.at(time_i).end()) << std::endl;
                }
                
                //////////////////////////////////////////// End output
            }

        }
    #ifdef _OPENMP
        }
    #endif
            
    return 0;
}


void consider_imitation(UGenerator rng, Network &net){
	//consider agents in random order
	std::random_shuffle(net.agent_seq.begin(), net.agent_seq.end());
	int agent;
	int pop = net.getPop();
	
	//get vector of a link weights (adj matrix)
	std::vector<double> adj_mat;
	for(int agent_num = 0; agent_num < pop; agent_num++){
		Agent temp_agent = net.GetAgent(agent_num);
		std::vector<double> friends = temp_agent.getFriends();
		adj_mat.insert(adj_mat.end(), friends.begin(), friends.end()); //temp_agent.getFriends().begin(), temp_agent.getFriends().end()
	}
    // Get total normalized interaction weights between each pair of agents
	double row_sum;
	std::vector<double> norm_adj(pop*pop);
    for(int up_adj_i = 0; up_adj_i < pop; up_adj_i++){
        row_sum = std::accumulate(adj_mat.begin()+up_adj_i*pop, adj_mat.begin()+up_adj_i*pop+pop,0.0);
        std::transform(adj_mat.begin()+up_adj_i*pop, adj_mat.begin()+up_adj_i*pop+pop, norm_adj.begin() + up_adj_i*pop, std::bind2nd(std::divides<double>(),row_sum));
    }
	
        
    // Loop through agents, random order
    for(int agent_num = 0; agent_num < pop; agent_num++){
		agent = net.agent_seq.at(agent_num); // Current agent (shuffled order)
		Agent &currentAgent = net.GetAgent(agent);
		std::vector<int> temp_agent_seq(net.agent_seq);
        temp_agent_seq.erase(std::remove(temp_agent_seq.begin(), temp_agent_seq.end(), agent), temp_agent_seq.end());
		int possibleRoleModel = currentAgent.chooseImitationPartner(rng, temp_agent_seq, norm_adj); 
		Agent possibleRoleModelAgent = net.GetAgent(possibleRoleModel);
		currentAgent.chooseToImitate(rng, possibleRoleModelAgent);
	}
	// simultaneous updating of strategy
	for(int agent_num = 0; agent_num < pop; agent_num++){
		agent = net.agent_seq.at(agent_num); // Current agent (shuffled order)
		Agent &currentAgent = net.GetAgent(agent);
		currentAgent.setStrategyProfile(currentAgent.getNextStrategyProfile());
	}
}

void die_replace(UGenerator rng, Network &net, Game &g, float total_coop_this_interval){
	int pop = net.getPop();
	// randomly select agent to die
	int rand_death_index = (int) (rng() * (pop));
    Agent &dead_agent = net.GetAgent(rand_death_index); //need & here??????
	
    // update Dead Agents outgoing weights to initializaiton
	double net_fill = 19.0/(pop-1); //(19.0/(pop-1))*10
	std::vector<double> myfriends(pop);
	std::fill(myfriends.begin(),myfriends.end(),net_fill);
	myfriends.at(rand_death_index) = 0;
	std::cout << "Before: " << dead_agent.getFriends().at(1) << std::endl;
    dead_agent.setCurFriends(myfriends); // ask about this
	std::cout << "After: " << dead_agent.getFriends().at(1) << std::endl;
	
	// update other Agents ingoing weights to initializaiton
	for(int agent_num = 0; agent_num < pop; agent_num++){
		if (rand_death_index != agent_num){
			Agent &currentAgent = net.GetAgent(agent_num);
			std::vector<double> myfriends = currentAgent.getFriends();
			myfriends.at(rand_death_index) = net_fill;
			currentAgent.setCurFriends(myfriends); 
		}
	}
	
	// Select new strategy type 
	std::vector<double> coop_payoffs = g.getCoopPayoffs();
	std::vector<double> defect_payoffs = g.getDefectPayoffs();
	
	float sum_of_coop = std::accumulate(coop_payoffs.begin(), coop_payoffs.end(), 0.0);
	float sum_of_defect = std::accumulate(defect_payoffs.begin(), defect_payoffs.end(), 0.0);
	
	int coop_len = coop_payoffs.size();
	int defect_len = defect_payoffs.size();
	
	int numCoop = net.getNumCoop();
	float pi;
	
	if (true){
		float avg_pay_per = (sum_of_coop + sum_of_defect)/(coop_len + defect_len);
		float avg_pay_per_i = sum_of_coop/coop_len;
		pi = numCoop*(avg_pay_per_i/(pop*avg_pay_per));
	} else {
		float total_payoff = sum_of_coop + sum_of_defect;
		float avg_accum = total_payoff/total_coop_this_interval;
		pi = numCoop*(avg_accum/total_payoff);
	}
	
	double reproduction_tremble = rng();
	double reproduction_tremble_rate = 0.001;
	double strat_draw = rng();
	
	if (reproduction_tremble > reproduction_tremble_rate){
		if(strat_draw < pi){
			dead_agent.setStrategyProfile(true);
		} else {
			dead_agent.setStrategyProfile(false);
		}
	} else {
		if(strat_draw < 0.5){
			dead_agent.setStrategyProfile(true);
		} else {
			dead_agent.setStrategyProfile(false);
		}
	}
	
	// clear payoff trackers
	g.clearPayoffs();
}
	

void run_model(UGenerator rng, Game &g, SimTracking &tracking_vars, Network &net){

    // Initialize agent sequence (to be randomized each round 
    tracking_vars.initAgentList(net);
    
	float cur_num_coop = 0;
	float prev_num_coop = 0;
	float total_coop_this_interval = 0;
	
    // Run simulation for max_time timesteps 
    for (int t = 1; t < tracking_vars.max_time+1; t++)
    {   
		cur_num_coop = net.getNumCoop();
		if (cur_num_coop > prev_num_coop){
			total_coop_this_interval = total_coop_this_interval + (cur_num_coop - prev_num_coop);
		}
		prev_num_coop = cur_num_coop;
		// Run simulation for one time step, loop through all agents once
        run_timestep(rng, g, tracking_vars, net, t, net.agent_seq);
		
		//********Uncomment code below to run model with Imitation
		// Agents consider imitating at end of each round
		consider_imitation(rng, net);
		
		//********Uncomment code below to run model with Moran process
		// Check if agent dies this round
		//if (t % 10 == 0){ // change to variable mod
			//die_replace(rng, net, g, total_coop_this_interval);
			//total_coop_this_interval = 0;
			//prev_num_coop = 0;
		//}
    }
        
}


void run_timestep(UGenerator rng, Game &g, SimTracking &tracking_vars, Network &net, int t, std::vector<int> agent_seq){
    
    // Shuffle agents in random order (updating is synchronous anyways so this only serves as another layer of randomness)

    std::random_shuffle(agent_seq.begin(), agent_seq.end());
        
    //////////////////////////////////////////
    
    int agent;
        
    // Loop through agents, random order
    for(int agent_num = 0; agent_num < net.getPop(); agent_num++)
    {        
        
        agent = agent_seq.at(agent_num); // Current agent (shuffled order)
        
        // Create temp vector for agent to choose random neighbor from
        std::vector<int> temp_agent_seq(agent_seq);

        //Remove current agent from vector
        temp_agent_seq.erase(std::remove(temp_agent_seq.begin(), temp_agent_seq.end(), agent), temp_agent_seq.end());
            
        // Get the current agent
        Agent &currentAgent = net.GetAgent(agent);
                
        // Choose interaction partner according to network weights
        int friend_ind = currentAgent.chooseFriend(rng,temp_agent_seq);
        
        // Set friend agent
        Agent &friendAgent = net.GetAgent(friend_ind);
        friendAgent.setCurrentFriend(agent);
        
        
        //Draw random numbers to determine strategy for host and visitor
        currentAgent.chooseStrategy(rng, currentAgent.getStrategyProfile());
        friendAgent.chooseStrategy(rng, friendAgent.getStrategyProfile());
        
        
        /*
         Interact
         */
        g.playGame(currentAgent, friendAgent);
 
        
        /*
         Update network weights
         */
        currentAgent.discountNeighbors();
        currentAgent.addNetworkPayoff();
        
    
        if(friendAgent.getNetworkSym() == 1) // This mean that agents partner updates their network weights as well
        {
            friendAgent.discountNeighbors();
            friendAgent.addNetworkPayoff();
        }
         
    }
	
    
    if(std::find(tracking_vars.times_tracked.begin(), tracking_vars.times_tracked.end(), t) != tracking_vars.times_tracked.end()) {
        tracking_vars.updateData(net, t);
    }else{
        for(int update_flag = 0; update_flag < net.getPop(); update_flag++){
            Agent &curAgent = net.GetAgent(update_flag);
            curAgent.updateAgent();
        }
    }
}

bool is_number(const std::string& s)
{
    return !s.empty() && std::find_if(s.begin(), s.end(), [](char c) { return !std::isdigit(c); }) == s.end();
}
bool file_exists (const std::string& name) {
    struct stat buffer;   
    return (stat (name.c_str(), &buffer) == 0); 
}
