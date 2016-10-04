#ifndef _EF_COMP_SCORER
#define _EF_COMP_SCORER

#include "StateTemp.h"

class Scorer{
private:
public:
	void score_them(vector<StateTemp>& them);	//attach scores to them
	void backprop_them(vector<State*>& them, vector<REAL>& grad);
	void clear(){}
};

#endif