#ifndef _EF_COMP_AGENDA
#define _EF_COMP_AGENDA

#include "State.h"
#include "../ef/DpOptions.h"

// In fact, the most of the resposibility is on Agenda
// -- This controls the process and know when to stop and back-prop, thus it will remember and book-keeping
class Agenda{
protected:
	// --- states ---
	vector<State*> records;		//for final releasing
	// --- options ---
	bool is_training;	// if training
	DpOptions* opt;
	// --- helpers ---
	void clear(){
		// clear the states
		for(auto* s : records)
			delete s;
	}
public:
	~Agenda(){ clear(); }
	Agenda(bool iftrain, DpOptions* the_opt):is_training(iftrain), opt(the_opt){}
	// to start a search
	decltype(records) init(State* start){
		clear();
		records.push_back(start);
		return vector<State*>{start};
	}
	// the main function -- ranking
	vector<State*> rank_them(vector<StateTemp>& them);
};

#endif