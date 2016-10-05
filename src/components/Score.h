#ifndef _EF_COMP_SCORE
#define _EF_COMP_SCORE

#include "../model/Model.h"

// For simplicity and efficiency, Score in fact is a vector of scores for all possible rel-labels
class Score{
private:
	Output* mout;	// outside
	vector<REAL> values;
public:
	Score(Output* x): mout(x){
		values = x->get_vec();
	}
	vector<int> kbest_index(int k);
	REAL get_one(int k);
};
#endif
