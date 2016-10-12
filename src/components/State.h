#ifndef _EF_COMP_STATE
#define _EF_COMP_STATE

#include "../ef/DpSentence.h"
#include "../tools/DpTools.h"
#include "StateTemp.h"

// State represent a partial tree, which needs further transitions
/*
-- A little explanation of the State and StateTemp.
1. why so complicated?
-> well, mainly for efficiency, most of the StateTemp will not be stablized as State, so only keep the lightweighted one.
2. what is the contract or internal semantic of them?
=> State::init=>EmptyState (the start) 
=> State::expand=>StateTemp (without Feature/Score/rel_index)
=> FeatureManager::feature_them(StateTemp) (adding Feature)
=> Scorer::score_them(StateTemp) (adding Score)
=> Agenda::rank_them(StateTemp)=>State {
	1.extend labels and add gold if training;
	2.calculate score (base->score+new_score+margin)
	3.stablize (add one edge, change structure and recordings and scores)
}
*/
class State{
protected:
	static const int NOPE_YET;	//-1
	// original sentence pointer
	DP_PTR sentence;
	// partial heads and rel-index for the tree, NOPE_YET means nope, also including dummy root node
	vector<int> partial_heads;	
	vector<int> partial_rels;
	vector<Score*> partial_sc;
	// double linked list for the top nodes
	vector<int> nb_left;		// left neibourhood
	vector<int> nb_right;		// right neibourhood
	// present childs --- 
	vector<int> ch_left;		// leftmost child
	vector<int> ch_right;		// rightmost child
	vector<int> ch_left2;		// second leftmost child
	vector<int> ch_right2;		// second rightmost child
	// recordings --- !! all from StateTemp !!
	REAL partial_score_base{0};		// base score
	REAL partial_score_all{0};		// base score + margin*num_wrong_doomed
	int num_arc{0};
	int num_wrong_cur{0};		//only record when training
	int num_wrong_doomed{0};	//doomed wrong arcs
	// special
	virtual int calculate_destiny() = 0;	// for num_wrong_doomed at the last step of transform

	// we only need these two for construction
	State(DP_PTR s): sentence(s), partial_heads(s->size(), NOPE_YET), partial_rels(s->size(), NOPE_YET),
		partial_sc(s->size(), nullptr), nb_left(s->size(), NOPE_YET), nb_right(s->size(), NOPE_YET), 
		ch_left(s->size(), NOPE_YET), ch_right(s->size(), NOPE_YET), ch_left2(s->size(), NOPE_YET), ch_right2(s->size(), NOPE_YET){
		// linear structure at first
		for(int i = 0; i < s->size(); i++){	
			nb_left[i] = i - 1;
			nb_right[i] = i + 1;
		}
		nb_left[0] = NOPE_YET;
		nb_right[s->size()-1] = NOPE_YET;
	}
	State(const State&) = default;
public:
	virtual ~State() = default;
	DP_PTR get_sentence(){ return sentence; }	// mainly for feeding the Feature
	int get_rel_label(int i){ return partial_rels[i]; }
	// init one
	static State* make_empty(DP_PTR s, int opt);
	// later ones are expanded and stabilized
	virtual vector<StateTemp> expand() = 0;
	// travel on the structure -- return NEGATIVE if nope
	static inline int travel_list(vector<int>* li, int i, int steps);
	inline int travel_upmost(int i);
	inline int travel_up(int i, int steps);
	inline int travel_down(int i, int which, int steps);	//(which): left:-1,-2,...; right:1,2,...
	inline int travel_downmost(int i, int which);
	inline int travel_lr(int i, int steps);	//(steps): left:-1,-2,..., right£º1,2,...
	// process in the Agenda part
	REAL get_score() const{ return partial_score_all; }		// return base+penalty score
	int get_doomed() const { return num_wrong_doomed; }
	bool is_correct() const{ return num_wrong_doomed == 0; }	// both structure and labels
	// identifications for recombination
	string get_repr(int mode, bool labeled);
	void transform(StateTemp*, bool, double margin);
	virtual State* copy() = 0;	// copy myself
	// others
	bool finished(){ return num_arc == sentence->size()-1; }
	int get_numarc() { return num_arc; }
	void assignto(DP_PTR x){	// write back to sentence
		if(!finished())
			Logger::Error("State: unfinished assign.");
		x->assign(partial_heads, partial_rels);
	}
	int append_si(vector<Input>& vo, vector<int>& vi){	
		//append scores and index, for backprop; return number
		int n = 0;
		for(unsigned i = 0; i < partial_heads.size(); i++){
			if(partial_heads[i] != NOPE_YET){
				vo.push_back(partial_sc[i]->get_input());
				vi.push_back(partial_rels[i]);
				n++;
			}
		}
		return n;
	}
};

// EasyFirst-Stdandard
class EfstdState: public State{
protected:
	int calculate_destiny() override;
public:
	EfstdState(DP_PTR s): State(s){}
	EfstdState(const EfstdState&) = default;
	~EfstdState() = default;
	State* copy() override { return new EfstdState{*this}; }
	vector<StateTemp> expand() override;
};

// EasyFirst-Eager
class EfeagerState: public EfstdState{
protected:
	int calculate_destiny() override;
public:
	EfeagerState(DP_PTR s): EfstdState(s){}
	EfeagerState(const EfeagerState&) = default;
	~EfeagerState() = default;
	State* copy() override { return new EfeagerState{*this}; }
	vector<StateTemp> expand() override;
};

// 0.1 inline travels
inline int State::travel_list(vector<int>* li, int i, int steps)	//static helper
{
	for(int k = 0; k < steps; k++){
		i = (*li)[i];
		if(i == NOPE_YET)
			break;
	}
	return i;
}
inline int State::travel_upmost(int i)
{
	while(1){	//may forever-loop if wrong data
		int p = travel_up(i, 1);
		if(p == NOPE_YET)
			return i;
		i = p;
	}
}
inline int State::travel_up(int i, int steps)
{
	return travel_list(&partial_heads, i, steps);
}
inline int State::travel_downmost(int i, int which)
{
	while(1){	//may forever-loop if wrong data
		int p = travel_down(i, which, 1);
		if(p == NOPE_YET)
			return i;
		i = p;
	}
}
inline int State::travel_lr(int i, int steps)
{
	vector<int>* current_li = &nb_right;
	if(steps < 0){	// steps<0 means left, else right
		current_li = &nb_left;
		steps = 0 - steps;
	}
	return travel_list(current_li, i, steps);
}
inline int State::travel_down(int i, int which, int steps)
{
	vector<int>* current_li{nullptr};
	switch(which){
	case -1:	current_li = &ch_left;	break;
	case -2:	current_li = &ch_left2;	break;
	case 1:		current_li = &ch_right;	break;
	case 2:		current_li = &ch_right2; break;
	default:	Logger::Error("Unimplemented travel_down for State.");
	}
	return travel_list(current_li, i, steps);
}

#endif
