#include "State.h"
#include "FeatureManager.h"
#include "../ef/DpOptions.h"
#include "../tools/DpTools.h"
#include <sstream>
#include <queue>
#include <cstdlib>

const int State::NOPE_YET = -1;
int State::loss_struct;
int State::loss_labels;
int State::loss_future;
int State::loss_span;

// 0. basics
void State::init_loss(int ls, int ll, int lf, int lspan)
{
	loss_struct = ls;
	loss_labels = ll;
	loss_future = lf;
	loss_span = lspan;
}
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
// TODO: special decoding: assuming the sentence length is smaller than 255, and sometimes '|' as separator
string State::get_repr(int mode, bool labeled)
{
	string tmp_str;
	bool use_c2 = false;	// use second child?
	int cdepth = 10000;		// large enough?
	switch(mode){
	case RECOMB_NOPE:	// return random
	{
		const int RANDSTR_NUM = 32;
		string random_str = "";
		for(int i = 0; i < RANDSTR_NUM; i++)
			random_str += (char)(std::rand() % 128);
		return random_str;
	}
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
			// for left and right nodes
			queue<pair<int, int>> qu;	// pair of <index, depth>
			qu.push(make_pair(cur, 0));
			if(ch_left[cur] != NOPE_YET)
				qu.push(make_pair(ch_left[cur], 1));
			if(ch_right[cur] != NOPE_YET)
				qu.push(make_pair(ch_right[cur], 1));
			while(!qu.empty()){
				auto one = qu.front();
				qu.pop();
				// skip the childs
				if(one.second >= cdepth)
					continue;
				// add 2 or 4
				auto them = vector<int>{ch_left[one.first], ch_right[one.first]};
				if(use_c2){
					them.push_back(ch_left2[one.first]);
					them.push_back(ch_right2[one.first]);
				}
				for(auto x : them){
					tmp_str += x;
					if(labeled)
						tmp_str += ((x == NOPE_YET) ? NOPE_YET : partial_rels[x]);
				}
				// next -- not for cur
				if(one.first < cur && ch_left[one.first] != NOPE_YET)
					qu.push(make_pair(ch_left[one.first], one.second+1));
				else if(one.first > cur && ch_right[one.first] != NOPE_YET)
					qu.push(make_pair(ch_right[one.first], one.second + 1));
			}
			tmp_str += '|';		// as separator
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
		if(!sentence->is_correct(mod, head)){	// structure wrong
			num_wrong_struct++;
			num_wrong_span += sentence->get_span(mod);
		}
		num_wrong_future = calculate_destiny();
		partial_score_all += get_loss()*margin;
	}
	return;
}

// calculating future loss
int EfstdState::calculate_destiny()
{
	// only need to check whether at the top-list
	int num_doomed = 0;
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
	int num_doomed = 0;
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