#include "State.h"
#include "FeatureManager.h"
#include "../ef/DpOptions.h"
#include <sstream>

// 0. basics
State* State::make_empty(DP_PTR s, int opt)
{
	switch(opt){
	case EF_STD:	return new EfstdState(s);
	case EF_EAGER:	return new EfeagerState(s);
	default:	throw runtime_error("Unkonw ef mode.");
	}
}

// 0.1 travels
int State::travel_list(vector<int>* li, int i, int steps)	//static helper
{
	for(int k = 0; k < steps; k++){
		i = (*li)[i];
		if(i == NOPE_YET)
			break;
	}
	return i;
}
int State::travel_upmost(int i)
{
	while(1){	//may forever-loop if wrong data
		int p = travel_up(i ,1);
		if(p == NOPE_YET)
			return i;
		i = p;
	}
}
int State::travel_up(int i, int steps)
{
	return travel_list(&partial_heads, i, steps);
}
int State::travel_downmost(int i, int which)
{
	while(1){	//may forever-loop if wrong data
		int p = travel_down(i, which, 1);
		if(p == NOPE_YET)
			return i;
		i = p;
	}
}
int State::travel_lr(int i, int steps)
{
	vector<int>* current_li = &nb_right;
	if(steps < 0){	// steps<0 means left, else right
		current_li = &nb_left;
		steps = 0 - steps;
	}
	return travel_list(current_li, i, steps);
}
int EfstdState::travel_down(int i, int which, int steps)
{
	vector<int>* current_li;
	switch(which){
	case -1:	current_li = &ch_left;	break;
	case -2:	current_li = &ch_left2;	break;
	case 1:		current_li = &ch_right;	break;
	case 2:		current_li = &ch_right2;break;
	default:	throw runtime_error("Unimplemented travel_down for State.");
	}
	return travel_list(current_li, i, steps);
}

// 1. expand: create the new candidates according to the current state
// EasyFirst-Stdandard
vector<StateTemp> EfstdState::expand()
{
	// In EF-std, only nodes at the top level will be considered as the head
	vector<StateTemp> them;
	// for all the nodes that have no heads
	int cur = nb_right[0];
	while(cur != NOPE_YET){
		// left or right
		if(nb_left[cur] != NOPE_YET)
			them.emplace_back(StateTemp(this, cur, nb_left[cur]));
		if(nb_right[cur] != NOPE_YET)
			them.emplace_back(StateTemp(this, cur, nb_right[cur]));
		cur = nb_right[cur];
	}
	return them;
}

// EasyFirst-Eager
vector<StateTemp> EfeagerState::expand()
{
	// In EF-eager, we need to consider the nodes on the spine as the head
	vector<StateTemp> them;
	// for all the nodes that have no heads
	int cur = nb_right[0];
	int one = NOPE_YET;
	while(cur != NOPE_YET){
		// left or right spine
		one = nb_left[cur];
		while(one != NOPE_YET){
			them.emplace_back(StateTemp(this, cur, one));
			one = ch_right[one];
		}
		one = nb_right[cur];
		while(one != NOPE_YET){
			them.emplace_back(StateTemp(this, cur, one));
			one = ch_left[one];
		}
		cur = nb_right[cur];
	}
	return them;
}

//2. get representation -- for recombination in searching
// TODO: special decoding: assuming the sentence length is smaller than 255, and sometimes '0' as separator
string EfstdState::get_repr(int mode, bool labeled)
{
	string tmp_str;
	bool use_c2 = false;	// use second child?
	int cdepth = -1;		// -1 means whole spine, it can never go down to 0
	switch(mode){
	case RECOMB_STRICT:
	{
		for(unsigned i = 1; i < partial_heads.size(); i++){
			tmp_str += partial_heads[i];
			if(labeled)
				tmp_str += partial_rels[i];
		}
		return tmp_str;		// directly return
	}
	case RECOMB_SPINE:	break;
	case RECOMB_SPINE2:	use_c2 = true; break;
	case RECOMB_TOPC:	cdepth = 1; break;
	case RECOMB_TOPC2:	use_c2 = true; cdepth = 1; break;
	case RECOMB_TOP:	cdepth = 0; break;
	default: throw runtime_error("Unkonw recomb mode."); break;
	}
	// get them
	{
		int cur = 0;
		while(cur != NOPE_YET){
			tmp_str += cur;
			// left or right
			const int step = 2;
			auto them = vector<vector<int>*>{&ch_left, &ch_left2, &ch_right, &ch_right2};
			for(unsigned i = 0; i < them.size(); i+=step){
				vector<int>& ch1 = *(them[i]);
				vector<int>& ch2 = *(them[i+1]);
				int tmp_depth = cdepth;
				int iter = ch1[cur];
				int iter2 = ch2[cur];
				while(tmp_depth != 0 && iter != NOPE_YET){
					tmp_str += iter;
					if(use_c2)
						tmp_str += iter2;
					if(labeled){
						tmp_str += partial_rels[iter];
						if(use_c2)
							tmp_str += partial_rels[iter2];
					}
					iter2 = ch2[iter];
					iter = ch1[iter];
					tmp_depth--;
				}
				tmp_str += '0';		// as separator
			}
			// move right
			cur = nb_right[cur];
		}
	}
	return tmp_str;
}

//3. transform: this is the friend method of StateTemp
void EfstdState::transform(StateTemp* st, bool istraining)
{
	// with the power of friend for StateTemp
	int mod = st->mod;
	int head = st->head;
	int rel_index = st->rel_index;
	Score* scores = st->scores;
	REAL pscore = st->partial_score;
	// add a new edge, currently simple
	// -- records
	num_arc++;
	partial_score = pscore;
	if(istraining)		// only know this when training
		if(!st->is_correct_cur())
			num_wrong++;
	// -- left-right for the mod if one top list, and nb_r/r[mod] has no meaning
	if(travel_up(mod, 1) == NOPE_YET){
		int left = nb_left[mod];
		int right = nb_right[mod];
		if(left != NOPE_YET)
			nb_right[left] = right;
		if(right != NOPE_YET)
			nb_left[right] = left;
	}
	// -- childs
	if(mod < head){
		ch_left2[head] = ch_left[head];
		ch_left[head] = mod;
	}
	else{
		ch_right2[head] = ch_right[head];
		ch_right[head] = mod;
	}
	// -- finally heads
	partial_heads[mod] = head;
	partial_rels[mod] = rel_index;
	partial_sc[mod] = scores;
	return;
}

// others: StateTemp
vector<StateTemp> StateTemp::expand_labels(StateTemp& st, int k, bool iftraining)
{
	vector<StateTemp> ret;
	auto l = st.scores->kbest_index(k);
	auto base = st.base;
	auto head = st.head;
	auto mod = st.mod;
	auto feat = st.feat;
	auto scores = st.scores;
	auto sent = base->get_sentence();
	bool hitflag = false;
	int gold_rel = sent->get_rel(st.mod);
	for(int i : l){
		if(i == gold_rel)
			hitflag = true;
		ret.emplace_back(StateTemp{base, mod, head, i, feat, scores});
	}
	if(iftraining && !hitflag){
		// only add gold for training when gold is possible
		if(base->is_correct() && sent->is_correct(mod, head))
			ret.emplace_back(StateTemp{base, mod, head, gold_rel, feat, scores});
	}
	return ret;
}

State* StateTemp::stablize(bool istraining){	// ..., what a design !=_=
	State* one = base->copy();
	one->transform(this, istraining);
	return one;
}

// StateTemp -- have to be here
StateTemp::StateTemp(State* s, int m, int h, int r, Feature* ff, Score* ss):
	base(s), mod(m), head(h), rel_index(r), feat(ff), scores(ss){
	partial_score = scores->get_one(r) + s->get_score();
}
bool StateTemp::is_correct_all(){ return is_correct_cur() && base->is_correct(); }	// is all correct?
DP_PTR StateTemp::get_sentence(){ return base->get_sentence(); }

Feature* StateTemp::fetch_feature(FeatureManager* fm){	// create if nope (both getter and setter)
	// let the State to add feature for it concerns the details of the structures
	if(feat)
		return feat;
	feat = fm->make_feature(base, mod, head);
	return feat;
}