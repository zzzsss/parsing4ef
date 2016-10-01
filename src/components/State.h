#ifndef _EF_COMP_STATE
#define _EF_COMP_STATE

#include "../ef/DpSentence.h"
#include "StateTemp.h"
#include "FeatureManager.h"

// the transition-system type
enum{EF_STD, EF_EAGER};

// State represent a partial tree, which needs further transitions
class State{
protected:
	static const int NOPE_YET = -1;
	// original sentence pointer
	DP_PTR sentence;
	// partial heads and rel-index for the tree, NOPE_YET means nope, also including dummy root node
	vector<int> partial_heads;	
	vector<int> partial_rels;
	// double linked list for the top nodes
	vector<int> nb_left;		// left neibourhood
	vector<int> nb_right;		// right neibourhood
	// recordings


	// we only need these two for construction
	State(DP_PTR s): sentence(s), partial_heads(s->size(), NOPE_YET), partial_rels(s->size(), NOPE_YET),
		nb_left(s->size(), NOPE_YET), nb_right(s->size(), NOPE_YET){
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
	virtual vector<StateTemp*> expand() = 0;
	virtual State* stabilize(StateTemp*) = 0;
	// travel on the structure -- return NEGATIVE if nope
	static int travel_list(vector<int>* li, int i, int steps);
	int travel_upmost(int i);
	int travel_up(int i, int steps) = 0;
	virtual int travel_down(int i, int which, int steps) = 0;	//(which): left:-1,-2,...; right:1,2,...
	int travel_downmost(int i, int which);
	int travel_lr(int i, int steps);	//(steps): left:-1,-2,..., right£º1,2,...
};

// EasyFirst-Stdandard
class EfstdState: public State{
protected:
	vector<int> ch_left;		// leftmost child
	vector<int> ch_right;		// rightmost child
	vector<int> ch_left2;		// second leftmost child
	vector<int> ch_right2;		// second rightmost child

	EfstdState(DP_PTR s): State(s), ch_left(s->size(), NOPE_YET), ch_right(s->size(), NOPE_YET),
		ch_left2(s->size(), NOPE_YET), ch_right2(s->size(), NOPE_YET){}
	EfstdState(const EfstdState&) = default;
public:
	virtual ~EfstdState() = default;
	virtual vector<StateTemp*> expand();
	virtual State* stabilize(StateTemp*);
	virtual int travel_down(int i, int which, int steps);
};

// EasyFirst-Eager
class EfeagerState: public EfstdState{
protected:
	EfeagerState(DP_PTR s): EfstdState(s){}
	EfeagerState(const EfeagerState&) = default;
public:
	virtual ~EfeagerState() = default;
	virtual vector<StateTemp*> expand();
	// virtual State* stabilize(StateTemp*);					// same as EfstdState
	// virtual int travel_down(int i, int which, int steps);	// same as EfstdState
};

#endif
