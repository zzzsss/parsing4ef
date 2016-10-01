#ifndef _EF_COMP_AGENDA
#define _EF_COMP_AGENDA

#include <vector>
#include "State.h"
using namespace std;

// In fact, the most of the resposibility is on Agenda
// -- This controls the process and know when to stop and back-prop, thus it will remember and book-keeping
class Agenda{
protected:
	vector<State*> records;		//for final releasing

	void clear(){
		// clear the states
		for(auto* s : records)
			delete s;
	}
public:
	~Agenda(){ clear(); }
	// to start a search
	decltype(records) init(State* start){
		clear();
		records.push_back(start);
		return vector<State*>{start};
	}
};

#endif