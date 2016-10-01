#ifndef _EF_COMP_SCORE
#define _EF_COMP_SCORE

// For simplicity and efficiency, Score in fact is a vector of scores for all possible rel-labels
class Score{
protected:

public:
	vector<int> kbest_index(int k);
};
#endif