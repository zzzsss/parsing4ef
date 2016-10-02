#include "Agenda.h"
#include <queue>
#include <algorithm>

namespace{
	bool TMP_cmp(StateTemp* i, StateTemp* j){ return (i->get_score() > j->get_score()); }
}

// one of the most important ones, deciding the beam at one time
// -- two (beam, training) out of three (..,..,transition) main jobs are here
// TODO -- this may be over-complex and should be devided ?? 
vector<State*> Agenda::rank_them(vector<StateTemp>& them)
{
	// !! for simplicity, just sort them all and extract
	// 1. expand labels
	vector<StateTemp*> them_all;
	for(auto& x : them){
		auto ones = x.expand_labels(opt->beam_flabel, is_training);		// surely add all gold when training
		them_all.insert(them_all.end(), ones.begin(), ones.end());
	}
	// 2. add margin if training
	if(is_training){
		for(auto* x : them_all)
			if(!x->is_correct_cur())
				x->set_score(x->get_score() + opt->margin);		//!! penalties are accumulated
	}
	// 3. sort them by score (high first)
	std::sort(them_all.begin(), them_all.end(), TMP_cmp);
	// 4. extract the highests for beam -- do recombination and gold finding
	vector<State*> beam;
	unordered_map<string, int> beam_index;	// index of ID in beam
	unordered_map<string, int> gold_index;	// index of ID for gold in beam
	unordered_map<string, int> structure_num;	// number in the beam of same structure ignoring labels
	REAL violation_value = 0;	// best - best_gold
	auto iter = them_all.begin();
	while(iter != them_all.end()){

	}
}
