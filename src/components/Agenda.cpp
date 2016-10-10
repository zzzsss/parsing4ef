#include "Agenda.h"
#include <queue>
#include <algorithm>
#include <unordered_set>
#include <cmath>
#include "../tools/DpTools.h"

namespace{
	bool TMP_cmp(const State* i, const State* j){ return (i->get_score() > j->get_score()); }
}

// one of the most important ones, deciding the beam at one time
// -- two (beam, training) out of three (..,..,transition) main jobs are here
// TODO -- this may be over-complex and should be devided ?? 
vector<State*> Agenda::rank_them(vector<StateTemp>& them, Scorer& scer)
{
	// !! for simplicity, just sort them all and extract
	// 1. expand labels
	vector<StateTemp> them_alltemp;
	for(auto& x : them){
		auto ones = StateTemp::expand_labels(x, opt->beam_flabel, is_training);		// surely add all gold when training
		them_alltemp.insert(them_alltemp.end(), ones.begin(), ones.end());
	}
	// 2. stabilize them all (for structured loss) and add margin if training
	vector<State*> them_all;
	for(auto &x: them_alltemp){
		auto one = x.stablize(is_training, opt->margin);
		them_all.push_back(one);
		records.push_back(one);
	}
	// 3. sort them by score (high first)
	std::sort(them_all.begin(), them_all.end(), TMP_cmp);
	// 4. extract the highests for beam -- do recombination and gold finding
	vector<State*> beam;
	unordered_set<string> beam_repr;			// repr of states in beam
	unordered_map<string, unsigned> structure_num;	// number in the beam of same structure ignoring labels
	bool no_gold_yet = true;					// no gold found before insertion
	int first_gold = -1;						// fisrt gold in the beam (could be the later inserted one)
	vector<State*> dropped_golds;				// collect dropped gold for training
	unordered_set<string> gold_repr;			// repr of golds (both dropped or undropped)
	// -- 4.1 first select to the beam size
	auto iter = them_all.begin();
	while(iter != them_all.end()){
		State* one = (*iter);
		bool drop = false;
		string one_repr = "";
		string one_repr_unlabel = "";
		// first check recombination -- with label
		if(opt->recomb_mode != RECOMB_NOPE){
			one_repr = one->get_repr(opt->recomb_mode, true);
			if(beam_repr.find(one_repr) != beam_repr.end())
				drop = true;
		}
		// next check label-beam
		if(!drop){
			one_repr_unlabel = one->get_repr(RECOMB_STRICT, false);
			auto ff = structure_num.find(one_repr_unlabel);
			if(ff != structure_num.end() && ff->second >= opt->beam_flabel)
				drop = true;
		}
		// insert into beam if no-drop
		if(!drop){
			beam.push_back(one);
			beam_repr.insert(one_repr);
			auto ff = structure_num.find(one_repr_unlabel);
			if(ff == structure_num.end())
				structure_num[one_repr_unlabel] = 1;
			else
				ff->second += 1;
		}
		// -- recording sth about gold only in training --
		if(is_training && one->is_correct()){
			if(opt->recomb_mode == RECOMB_NOPE || gold_repr.find(one_repr) == gold_repr.end()){	// also do recombination for golds
				gold_repr.insert(one_repr);
				if(drop)
					dropped_golds.push_back(one);
				else if(first_gold < 0){		// first undropped gold
					first_gold = beam.size() - 1;
					no_gold_yet = false;
				}
			}
		}
		// break if beam is full
		iter++;
		if(beam.size() >= opt->beam_all)
			break;
	}
	// -- 4.2 while training, insert gold if no gold found?
	if(is_training){
		// insert opt->inum or less gold ones
		if(no_gold_yet){
			// find more until gold_inum or end
			// -- but there must be at least one (because StateTemp::expand_labels ensures this)
			while(iter != them_all.end() && dropped_golds.size() < opt->gold_inum){
				State* one = *iter;
				if(one->is_correct()){		// only search for gold ones
					// only check for gold recombination
					string one_repr = one->get_repr(opt->recomb_mode, true);
					if(opt->recomb_mode == RECOMB_NOPE || gold_repr.find(one_repr) == gold_repr.end()){
						gold_repr.insert(one_repr);
						dropped_golds.push_back(one);	// append it to this list for convenience
					}
				}
				iter++;
			}
			// insert them into the beam
			if(opt->gold_inum > 0){	//notice that opt->gold_inum could be zero
				if(dropped_golds.empty()){
					//Logger::Warn("Currently no golds in beam.");
				}
				else{
					first_gold = beam.size();
					beam.insert(beam.end(), dropped_golds.begin(), dropped_golds.begin() + opt->gold_inum);
				}
			}
		}
	}
	// 5. remember some states if training
	last_beam = beam;
	if(is_training){
		REAL violation_value = 0;
		if(first_gold > 0)
			violation_value = beam[0]->get_score() - beam[first_gold]->get_score();	// best - best_gold
		if(violation_value > max_viol){
			max_beam = beam;
			max_viol = violation_value;
		}
	}
	// 6.1 update and possibly alter beam when training
	bool finished = beam[0]->finished(); //here is the simple criterion
	if(is_training)
		beam = alter_beam(beam, no_gold_yet, finished, scer);
	// 6.2 finished for testing --- 
	else if(finished)
		beam.clear();
	return beam;	// the beam is always ordered
}

// crucial step for updating and finishing when training (according to training schemes)
vector<State*> Agenda::alter_beam(vector<State*>& curr_beam, bool no_gold, bool finished, Scorer& scer)
{
	// based on the mode
	switch(opt->update_mode){
	case UPDATE_END:	// only update at the end
		if(finished)
			backp_beam(curr_beam, scer);
		break;
	case UPDATE_MAXV:	// only update at the end, but with the max_violation one
		if(finished)
			backp_beam(max_beam, scer);
		break;
	case UPDATE_EU:		// update and stop both when no_gold or finished
		if(finished || no_gold)
			backp_beam(curr_beam, scer);
		if(no_gold)
			curr_beam.clear();
		break;
	case UPDATE_RESTART:// update and restart with golds (later inserted) when no gold
		if(finished || no_gold)
			backp_beam(curr_beam, scer);
		if(no_gold){	// remain only the gold ones
			auto tmp_beam = curr_beam;
			curr_beam.clear();
			for(auto* one : tmp_beam){
				if(one->is_correct())
					curr_beam.push_back(one);
			}
		}
		break;
	default:
		Logger::Error("Unkonw update(backprop) mode.");
		break;
	}
	if(finished)
		curr_beam.clear();
	return curr_beam;
}

// do the backprop once, here we don't care about lrate, setting as 1.0/div
void Agenda::backp_beam(vector<State*>& ubeam, Scorer& scer)
{
	// should we divide it or not?
	int div;
	switch(opt->updatediv_mode){
	case UPDATEDIV_ONE:
		div = 1;
		break;
	case UPDATEDIV_CUR:
		div = ubeam[0]->get_numarc();
		break;
	case UPDATEDIV_ALL:
		div = ubeam[0]->get_sentence()->size();
		break;
	default:
		Logger::Error("Unkonw update-div mode.");
		break;
	}
	// about the loss
	vector<State*> to_update;
	vector<REAL> to_grads;
	switch(opt->updatediv_mode){	// the first two must need gold ones !!
	case LOSS_PERCEPTRON:	// loss = best - gold
	{
		// TODO: do we need to update for all the golds?
		State* best = ubeam[0];
		State* gold = nullptr;
		for(auto* x : ubeam){
			if(x->is_correct()){
				gold = x;
				break;
			}
		}
		if(!gold){
			Logger::Warn("Update need at least one gold, skip update.");
			return;
		}
		else{
			if(gold != best){
				to_update = vector<State*>{best, gold};
				to_grads = vector<REAL>{1, -1};
			}
		}
		break;
	}
	case LOSS_ACRF:			// loss = - log (exp(gold) - sum(exp(all)))
	{
		to_update = ubeam;
		to_grads = vector<REAL>{};
		REAL exp_all = 0;
		REAL exp_gold = 0;
		for(auto* x : ubeam){
			REAL one_exp = exp(x->get_score());
			to_grads.push_back(one_exp);
			exp_all += one_exp;
			if(x->is_correct())
				exp_gold += one_exp;
		}
		if(!exp_gold){
			Logger::Warn("Update need at least one gold, skip update.");
			return;
		}
		else{
			for(unsigned i = 0; i < ubeam.size(); i++){
				if(ubeam[i]->is_correct())
					to_grads[i] = to_grads[i] / exp_all - to_grads[i] / exp_gold;
				else
					to_grads[i] = to_grads[i] / exp_all;
			}
		}
		break;
	}
	case LOSS_IMRANK:		// calculate move-up/down scores and normalize with exponention, implicit ranking loss
	{
		// -- first distribution sort on real loss (doomed-error)
		vector<int> distribution;
		for(auto* x : ubeam){
			unsigned doom = x->get_doomed();
			while(doom <= distribution.size())
				distribution.push_back(0);
			distribution[doom] ++;
		}
		vector<int> place_up(distribution.size(), 0);
		vector<int> place_down(distribution.size(), 0);
		int accum = 0;	// next start
		for(unsigned i = 0; i < distribution.size(); i++){
			place_up[i] = accum;
			accum += distribution[i];
			place_down[i] = accum - 1;
		}
		// -- next collect the stats
		vector<REAL> distance;
		for(unsigned i = 0; i < ubeam.size(); i++){
			// i is real place, [place_up[doom], place_down[doom]] is the target
			unsigned doom = ubeam[i]->get_doomed();
			unsigned p_r1 = place_up[doom];
			unsigned p_r2 = place_down[doom];
			if(i < p_r1)		// etimated high, it should go down
				distance.push_back(ubeam[p_r1]->get_score() - ubeam[i]->get_score());
			else if(i > p_r2)	// etimated low, it should add up
				distance.push_back(ubeam[p_r2]->get_score() - ubeam[i]->get_score());
			else
				distance.push_back(0.0f);
		}
		// -- finally the gradient
		REAL exp_all_plus = 0, exp_all_minus = 0;
		for(auto r : distance){
			if(r > 0)
				exp_all_plus += exp(r);
			else if(r < 0)
				exp_all_minus += exp(0-r);
		}
		for(unsigned i = 0; i < ubeam.size(); i++){
			REAL one = distance[i];
			if(one > 0){
				to_update.push_back(ubeam[i]);
				to_grads.push_back(0 - exp(one) / exp_all_plus);
			}
			else if(one < 0){
				to_update.push_back(ubeam[i]);
				to_grads.push_back(exp(0-one) / exp_all_minus);
			}
		}
		break;
	}
	default:
		Logger::Error("Unkonw loss mode.");
		break;
	}
	scer.backprop_them(to_update, to_grads, div);
	return;
}
