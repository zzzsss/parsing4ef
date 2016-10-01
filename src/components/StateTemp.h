#ifndef _EF_COMP_STATE_TEMP
#define _EF_COMP_STATE_TEMP

#include "State.h"
#include "Score.h"

// This can be seen as a light-weighted State, with one extra transition on the base of a State
// -- when it remains in the beam, it can be later stablized to a new State
// -- in fact, this class should be generalized to a Transition class maybe
class StateTemp{
protected:
	State* base;
	// -- the new edge this time
	int mod;
	int head;
	int rel_index{-1};		//this one is treated in a tricky way, mainly for efficiency
	Feature* feat{nullptr};	// -- the feature
	Score* scores{nullptr};			// -- the scores, in fact it is a vector for all possible rels
public:
	StateTemp(State* s, int m, int h): base(s), mod(m), head(h){}	// only init those
	Feature* fetch_feature(FeatureManager* fm){	// create if nope (both getter and setter)
		// let the State to add feature for it concerns the details of the structures
		if(feat)
			return feat;
		feat = fm->make_feature(base, mod, head);
		return feat;
	}
	void set_score(Score* s){ scores = s; }
	DP_PTR get_sentence(){ return base->get_sentence(); }
	// !! using in Agenda.rank_them() !!

};

#endif

