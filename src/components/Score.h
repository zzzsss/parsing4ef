#ifndef _EF_COMP_SCORE
#define _EF_COMP_SCORE

#include "../model/Model.h"
#include <algorithm>

// For simplicity and efficiency, Score in fact is a vector of scores for all possible rel-labels
// -- Input is from FeatureManager, Output is from Model, both managed by Score
class Score{
private:
	Input min;
	Output mout;
	vector<REAL> values_sorted;
	const vector<REAL>& get_output(){ return *mout; }
public:
	Score(Input in, Output out): min(in), mout(out){ 
		values_sorted = *mout; 
		sort(values_sorted.begin(), values_sorted.end()); 
		reverse(values_sorted.begin(), values_sorted.end());
	}
	~Score(){ delete min.first; delete min.second; delete mout; }
	Input get_input(){ return min; }
	REAL get_one(int k){ return get_output()[k]; }
	vector<int> kbest_index(unsigned k){	//called only when StateTemp::expand_labels
		if(k > get_output().size())
			k = get_output().size();
		// simple but inefficient implementation assuming k is small
		vector<int> ret;
		REAL kvalue = values_sorted[k-1];
		for(unsigned i = 0; i < get_output().size(); i++)
			if(get_output()[i] >= kvalue)		// approximate
				ret.push_back(i);
		return ret;
	}
};
#endif
