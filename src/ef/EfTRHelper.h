#ifndef _EF_EFTRHELPER
#define _EF_EFTRHELPER

#include "DpOptions.h"
#include "../tools/DpTools.h"

// helper for the training process 
// -- mainly for control the lrate and iters
class EfTRHelper{
private:
	//
	double lrate_init;		// init lr
	double lrate_current;	// current lr
	double lrate_multiply;	// lr mulply-value
	int iters_all;			// the upper setting for #iters
	int iters_current{0};	// how many iters finished
	vector<double> scores{};
	//cut
	double tr_cut;			// cutting rate for lr
	int tr_cut_times;		// at least cut this times (so real iters maybe more than iter)
	int tr_cut_iters;		// force cut if no cutting for how many iters
	// records
	int total_cut_times{0};
	int last_cut_iter{-1};
	int best_iter{0};
	double best_score{0};
public:
	EfTRHelper(DpOptions* op): lrate_init(op->tr_lrate), lrate_current(op->tr_lrate), iters_all(op->tr_iters),
		tr_cut(op->tr_cut), tr_cut_times(op->tr_cut_times), tr_cut_iters(op->tr_cut_iters){}
	bool keepon(){
		return iters_current < iters_all || total_cut_times < tr_cut_times;
	}
	bool end_iter(double s){	// s is the current dev score, return whether save model
		// first for lrate
		bool this_cut = false;
		bool this_best = false;
		if(scores.empty() || s < scores.back() || (iters_current-last_cut_iter)>=total_cut_times){
			lrate_current *= lrate_multiply;
			last_cut_iter = iters_current;
			this_cut = true;
		}
		if(s > best_score){
			best_iter = iters_current;
			best_score = s;
			this_best = true;
		}
		Logger::get_output() << "- End of Iteration " << iters_current << ", cut/save-best: " << this_cut << "/" << this_best << endl;
		iters_current++;
		return this_best;
	}
	double get_lrate(){ return lrate_current; }
	int get_iter(){ return iters_current; }
};

#endif // !_EF_EFTRHELPER

