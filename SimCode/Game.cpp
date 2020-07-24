/* The Network class Implementation (Network).cpp) */
#include "Network.h" // user-defined header in the same directory
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iterator>
#include <math.h>
#include <numeric>

Game::Game(std::vector<std::vector<double>> gamePayoffs,std::string gameName, double base_payoff){
    this->gamePayoffs = gamePayoffs;
    this->gameName = gameName;
    this->base_payoff = base_payoff;
}

Game::Game(std::string payoff_filepath, std::string gameName, double base_payoff){
    this->gameName = gameName;
    this->base_payoff = base_payoff;
    
    std::vector<int> num_strats;
    
    std::ifstream infile(payoff_filepath);
    
    std::string line;
    
    int i = 0;
    
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        
        if(i == 0){
            std::istream_iterator<int> start(iss), end;
            std::vector<int> this_line(start, end);
            num_strats = this_line;
        }else if(i < 3){
            std::istream_iterator<double> start(iss), end;
            std::vector<double> this_line(start, end);
            gamePayoffs.push_back(this_line);
        }
        i++;
    }
    
}


std::string Game::getName(){
    return gameName;
}

void Game::setPayoffs(std::vector<std::vector<double>> gamePayoffs){
    this->gamePayoffs = gamePayoffs;
}

std::vector<std::vector<double>> Game::getPayoffs(){
    return gamePayoffs;
}

std::vector<double> Game::getCoopPayoffs(){
    return coop_payoffs;
}

std::vector<double> Game::getDefectPayoffs(){
    return defect_payoffs;
}

void Game::clearPayoffs(){
    coop_payoffs.clear();
	defect_payoffs.clear();
}

double Game::getBasePayoff(){
    return base_payoff;
};

void Game::playGame(Agent &currentAgent, Agent &friendAgent){
    
    int p1Strategy = currentAgent.getCurrentStrategy();
    int p2Strategy = friendAgent.getCurrentStrategy();
            
    double p1Payoff = gamePayoffs.at(0).at(p1Strategy * 2 + p2Strategy) + base_payoff;
    double p2Payoff = gamePayoffs.at(1).at(p1Strategy * 2 + p2Strategy) + base_payoff;
    
    currentAgent.setCurrentPayoff(p1Payoff);
    friendAgent.setCurrentPayoff(p2Payoff);
    
    //currentAgent.recordInteraction(p1Payoff,p2Payoff);
	currentAgent.recordMemory(p1Payoff);
	friendAgent.recordMemory(p2Payoff);
	
	if (currentAgent.getStrategyProfile()){
		coop_payoffs.push_back(p1Payoff);
    } else {
		defect_payoffs.push_back(p1Payoff);
	}
	
	if (friendAgent.getStrategyProfile()){
		coop_payoffs.push_back(p2Payoff);
    } else {
		defect_payoffs.push_back(p2Payoff);
	}
}
