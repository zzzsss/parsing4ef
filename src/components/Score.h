#ifndef _EF_COMP_SCORE
#define _EF_COMP_SCORE

using REAL = float;

// For simplicity and efficiency, Score in fact is a vector of scores for all possible rel-labels
class Score{
protected:

public:
	vector<int> kbest_index(int k);
	REAL get_one(int k);
};
#endif