#ifndef _EF_COMP_STATE
#define _EF_COMP_STATE

#include "../ef/DpSentence.h"
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
	3.stablize (add one edge, change structure and recordings and scores(direct from StateTemp's partial_score))
}
*/
class State{
protected:
	static const int NOPE_YET = -1;
	// original sentence pointer
	DP_PTR sentence;
	// partial heads and rel-index for the tree, NOPE_YET means nope, also including dummy root node
	vector<int> partial_heads;	
	vector<int> partial_rels;
	vector<Score*> partial_sc;
	// double linked list for the top nodes
	vector<int> nb_left;		// left neibourhood
	vector<int> nb_right;		// right neibourhood
	// recordings
	REAL partial_score{0};	//including penalty when training
	int num_arc{0};
	int num_wrong{0};		//only record when training

	// we only need these two for construction
	State(DP_PTR s): sentence(s), partial_heads(s->size(), NOPE_YET), partial_rels(s->size(), NOPE_YET),
		nb_left(s->size(), NOPE_YET), nb_right(s->size(), NOPE_YET), partial_sc(s->size(), nullptr){
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
	static int travel_list(vector<int>* li, int i, int steps);
	int travel_upmost(int i);
	int travel_up(int i, int steps);
	virtual int travel_down(int i, int which, int steps) = 0;	//(which): left:-1,-2,...; right:1,2,...
	int travel_downmost(int i, int which);
	int travel_lr(int i, int steps);	//(steps): left:-1,-2,..., right£º1,2,...
	// process in the Agenda part
	REAL get_score(){ return partial_score; }
	bool is_correct(){ return num_wrong == 0; }	// both structure and labels
	// identifications for recombination
	virtual string get_repr(int mode, bool labeled) = 0;
	virtual State* copy() = 0;	// copy myself
	virtual void transform(StateTemp*, bool) = 0;
	// others
	bool finished(){ return num_arc == sentence->size()-1; }
	int get_numarc() { return num_arc; }
	void assignto(DP_PTR x){	// write back to sentence
		if(!finished())
			throw runtime_error("State: unfinished assign.");
		x->assign(partial_heads, partial_rels);
	}
	int append_si(vector<Output*>& vo, vector<int>& vi){	
		//append scores and index, for backprop; return number
		int n = 0;
		for(unsigned i = 0; i < partial_heads.size(); i++){
			if(partial_heads[i] != NOPE_YET){
				vo.push_back(partial_sc[i]->get_output());
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
	vector<int> ch_left;		// leftmost child
	vector<int> ch_right;		// rightmost child
	vector<int> ch_left2;		// second leftmost child
	vector<int> ch_right2;		// second rightmost child
public:
	EfstdState(DP_PTR s): State(s), ch_left(s->size(), NOPE_YET), ch_right(s->size(), NOPE_YET),
		ch_left2(s->size(), NOPE_YET), ch_right2(s->size(), NOPE_YET){}
	EfstdState(const EfstdState&) = default;
	~EfstdState() = default;
	State* copy() override { return new EfstdState{*this}; }
	vector<StateTemp> expand() override;
	// for both Ef*State
	int travel_down(int i, int which, int steps) override;
	string get_repr(int mode, bool labeled) override;
	void transform(StateTemp*, bool) override;	// this is the friend method of StateTemp
};

// EasyFirst-Eager
class EfeagerState: public EfstdState{
protected:
public:
	EfeagerState(DP_PTR s): EfstdState(s){}
	EfeagerState(const EfeagerState&) = default;
	~EfeagerState() = default;
	State* copy() override { return new EfeagerState{*this}; }
	vector<StateTemp> expand() override;
};

#endif
