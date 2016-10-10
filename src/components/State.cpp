#include "State.h"
#include "FeatureManager.h"
#include "../ef/DpOptions.h"
#include "../tools/DpTools.h"
#include <sstream>

const int State::NOPE_YET = -1;

// 0. basics
State* State::make_empty(DP_PTR s, int opt)
{
	switch(opt){
	case EF_STD:	return new EfstdState(s);
	case EF_EAGER:	return new EfeagerState(s);
	default:	Logger::Error("Unkonw ef mode.");
	}
	return nullptr;
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
string State::get_repr(int mode, bool labeled)
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
	default: Logger::Error("Unkonw recomb mode."); break;
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
void State::transform(StateTemp* st, bool istraining, double margin)
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
	partial_score_base += pscore;
	partial_score_all = partial_score_base;
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
	// final updating
	if(istraining){		// only know this when training
		if(!sentence->is_correct(mod, head, rel_index))
			num_wrong_cur++;
		num_wrong_doomed = calculate_destiny();
		partial_score_all += num_wrong_doomed*margin;
	}
	return;
}

int EfstdState::calculate_destiny()
{
	// only need to check whether at the top-list
	int num_doomed = num_wrong_cur;
	int cur = nb_right[0];
	while(cur != NOPE_YET){
		int real_head = sentence->get_head(cur);
		if(partial_heads[real_head] != NOPE_YET)
			num_doomed++;		// not reachable
		cur = nb_right[cur];
	}
	return num_doomed;
}

int EfeagerState::calculate_destiny()
{
	// need to find the ones on the lr-spine
	int num_doomed = num_wrong_cur;
	// find it in the lr-spine
	vector<bool> rspine(sentence->size(), false);
	for(int n = 0; n != NOPE_YET; n = ch_right[n])
		rspine[n] = true;
	vector<bool> lspine(sentence->size(), false);
	int cur = 0;
	int cur2 = nb_right[cur];
	// left to right and find rightmost
	while(cur2 != NOPE_YET){
		int real_head = sentence->get_head(cur2);
		if(real_head < cur2 && !rspine[real_head])
			num_doomed++;		// not reachable
		for(int n = cur2; n != NOPE_YET; n = ch_right[n])
			rspine[n] = true;
		cur = cur2;
		cur2 = nb_right[cur];
	}
	// right to left (cur2 now no-use)
	while(cur != NOPE_YET){
		int real_head = sentence->get_head(cur);
		if(real_head > cur && !lspine[real_head])
			num_doomed++;		// not reachable
		for(int n = cur; n != NOPE_YET; n = ch_left[n])
			lspine[n] = true;
		cur = nb_right[cur];
	}
	return num_doomed;
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

State* StateTemp::stablize(bool istraining, double margin){	// ..., what a design !=_=
	State* one = base->copy();
	one->transform(this, istraining, margin);
	return one;
}

// StateTemp -- have to be here
StateTemp::StateTemp(State* s, int m, int h, int r, Feature* ff, Score* ss):
	base(s), mod(m), head(h), rel_index(r), feat(ff), scores(ss){
	partial_score = scores->get_one(r);
}
DP_PTR StateTemp::get_sentence(){ return base->get_sentence(); }

Feature* StateTemp::fetch_feature(FeatureManager* fm){	// create if nope (both getter and setter)
	// let the State to add feature for it concerns the details of the structures
	if(feat)
		return feat;
	feat = fm->make_feature(base, mod, head);
	return feat;
}