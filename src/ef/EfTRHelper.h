#ifndef _EF_EFTRHELPER
#define _EF_EFTRHELPER

#include "DpOptions.h"
#include "../tools/DpTools.h"

// helper for the training process 
// -- mainly for control the lrate and iters
class EfTRHelper{
private:
	//
	DpOptions* opt;
	double lrate_init;		// init lr
	double lrate_current;	// current lr
	int iters_all;			// the upper setting for #iters
	int iters_current{0};	// how many iters finished
	vector<double> scores{};
	//cut
	double tr_cut;			// cutting rate for lr
	int tr_cut_times;		// at least cut this times (so real iters maybe more than iter)
	int tr_cut_iters;		// force cut if no cutting for how many iters
  int tr_nocut_iters; // no cutting for the first several iters
	double tr_cut_sthres;		// if dev has not improved more than this, cut the lr
	// records
	int total_cut_times{0};
	int last_cut_iter{-1};
	int best_iter{0};
	double best_score{0};
public:
	EfTRHelper(DpOptions* op): opt(op), lrate_init(op->tr_lrate), lrate_current(op->tr_lrate), iters_all(op->tr_iters),
		tr_cut(op->tr_cut), tr_cut_times(op->tr_cut_times), tr_cut_iters(op->tr_cut_iters), tr_nocut_iters(op->tr_nocut_iters),
    tr_cut_sthres(op->tr_cut_sthres){}
	bool keepon(){
		return iters_current < iters_all || total_cut_times < tr_cut_times;
	}
	bool end_iter_save(double s){	// s is the current dev score, return whether save model
		// first for lrate
		bool this_cut = false;
		bool this_best = false;
		//<remove the first condition>if( || (iters_current-last_cut_iter)>=tr_cut_iters){
		bool cut_for_score = (!scores.empty() && (s-scores.back()) < tr_cut_sthres);
		bool cut_for_time = ((iters_current - last_cut_iter) >= tr_cut_iters);
		if(cut_for_score || (iters_current >= tr_nocut_iters && cut_for_time)){
			lrate_current *= tr_cut;
			last_cut_iter = iters_current;
			this_cut = true;
			total_cut_times++;
		}
		if(s > best_score){
			best_iter = iters_current;
			best_score = s;
			this_best = true;
		}
    scores.push_back(s);
    iters_current++;
		Logger::get_output() << "- End of Iteration " << iters_current << ", cut/save-best: " << this_cut << "/" << this_best << endl;
		// lower bound for lrate
		if(lrate_current < opt->tr_lrate_lbound){
			Logger::get_output() << "Lrate too small, forced to the lbound." << endl;
			lrate_current = opt->tr_lrate_lbound;
		}
		return this_best;
	}
	double get_lrate(){ return lrate_current; }
	int get_iter(){ return iters_current; }
	void report(){
		ostream& fout = Logger::get_output();
		fout << "-- Finish training: Best/All:" << best_iter << "/" << iters_current << " [";
		for(auto x : scores)
			fout << x << ",";
		fout << "]" << endl;
		// for better grep
		fout << "zzzzz ";
		for(auto x : scores)
			fout << x << " ";
		fout << endl;
	}
};

#endif // !_EF_EFTRHELPER
