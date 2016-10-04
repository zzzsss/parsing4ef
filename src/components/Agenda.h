#ifndef _EF_COMP_AGENDA
#define _EF_COMP_AGENDA

#include "State.h"
#include "../ef/DpOptions.h"
#include "Scorer.h"

// In fact, the most of the resposibility is on Agenda
// -- This controls the process and know when to stop and back-prop, thus it will remember and book-keeping
class Agenda{
protected:
	// --- states ---
	vector<State*> records;		//for final releasing
	// --- options ---
	bool is_training;	// if training
	DpOptions* opt;
	// --- memories ---
	vector<State*> last_beam;
	vector<State*> max_beam;	//for MAX_VIOLATION update method
	REAL max_viol{0};				//for MAX_VIOLATION update method
	// --- helpers ---
	void clear(){
		// clear the states
		for(auto* s : records)
			delete s;
	}
	// crucial step for updating and finishing when training
	vector<State*> alter_beam(vector<State*>& curr_beam, bool no_gold, bool finished, Scorer& scer);
	void backp_beam(vector<State*>& ubeam, Scorer& scer);
public:
	~Agenda(){ clear(); }
	Agenda(bool iftrain, DpOptions* the_opt):is_training(iftrain), opt(the_opt){}
	// to start a search
	vector<State*> init(State* start){
		clear();
		records.push_back(start);
		if(is_training){
			last_beam = vector<State*>{start};
			max_beam = vector<State*>{start};
		}
		return vector<State*>{start};
	}
	// the main function -- ranking
	vector<State*> rank_them(vector<StateTemp>& them, Scorer& scer);
	State* get_best(){ return last_beam[0]; }
};

#endif