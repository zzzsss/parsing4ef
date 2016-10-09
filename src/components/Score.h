#ifndef _EF_COMP_SCORE
#define _EF_COMP_SCORE

#include "../model/Model.h"
#include <algorithm>

// For simplicity and efficiency, Score in fact is a vector of scores for all possible rel-labels
class Score{
private:
	Output* mout;	// outside
	vector<REAL> values;
	vector<REAL> values_sorted;
public:
	Score(Output* x): mout(x){ 
		values = x->get_vec(); 
		values_sorted = values; 
		sort(values_sorted.begin(), values_sorted.end()); 
	}
	Output* get_output(){ return mout; }
	REAL get_one(int k){ return values[k]; }
	vector<int> kbest_index(unsigned k){	//called only when StateTemp::expand_labels
		if(k > values.size())
			k = values.size();
		// simple but inefficient implementation assuming k is small
		vector<int> ret;
		REAL kvalue = values_sorted[k-1];
		for(unsigned i = 0; i < values.size(); i++)
			if(values[i] < kvalue)
				ret.push_back(i);
		return ret;
	}
};
#endif
