#ifndef _EF_COMP_SEARCHER
#define _EF_COMP_SEARCHER

#include "../ef/DpSentence.h"
#include "../ef/DpOptions.h"
#include "FeatureManager.h"
#include "../model/Model.h"

// the main class for EF search, both for training and testing
// -- manage the whole searching process (with the help of other class)
class Searcher{
private:
	//from outside: the options, model and fm
	DpOptions* options;		
	bool is_training;
	ModelZ* model;
	FeatureManager* fm;

	int num_sent{0};
	int num_token{0};
	int num_steps{0};	//num of steps explored
public:
	Searcher(DpOptions* op, bool t, ModelZ* m, FeatureManager* f):
		options(op), is_training(t), model(m), fm(f){}
	~Searcher(){}
	void report_stat(ostream& out){
		out << "#sent:" << num_sent << ";#token:" << num_token
			<< ";#steps:" << num_steps << ";%cov:" << num_steps / (num_token + 0.0) << endl;
	}
	void ef_search(DP_PTR one);
	static void report_and_reset_all();
};

#endif
