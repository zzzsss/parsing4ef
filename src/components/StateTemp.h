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
	REAL partial_score{-100000};	// base.partial_score + this_one
public:
	// init1: from State::expand, only init those
	StateTemp(State* s, int m, int h): base(s), mod(m), head(h){}	
	// init2: from StateTemp::expand_labels, full one
	StateTemp(State* s, int m, int h, int r, Feature* ff, Score* ss): 
		base(s), mod(m), head(h), rel_index(r), feat(ff), scores(ss){
		partial_score = scores->get_one(r) + s->get_score();
	}	
	StateTemp(const StateTemp&) = default;	// !! choose to mainly copy this because StateTemp is easy to copy
	StateTemp(StateTemp&&) = default;
	~StateTemp() = default;
	Feature* fetch_feature(FeatureManager* fm){	// create if nope (both getter and setter)
		// let the State to add feature for it concerns the details of the structures
		if(feat)
			return feat;
		feat = fm->make_feature(base, mod, head);
		return feat;
	}
	// setter and getter
	void set_score(REAL a){ partial_score = a; }
	REAL get_score(){ return partial_score; }
	DP_PTR get_sentence(){ return base->get_sentence(); }
	bool is_correct_cur(){ return get_sentence()->is_correct(mod, head, rel_index); }	// is current one correct?
	bool is_correct_all(){ return is_correct_cur() && base->is_correct(); }	// is all correct?
	// !! using in Agenda.rank_them() !!
	// expand the k-best labels and return new StateTemps (this may be bad design!!), ensuring add GOLD when training
	vector<StateTemp*> expand_labels(int k, bool iftraining) 
	{
		vector<StateTemp*> ret;
		auto l = scores->kbest_index(k);
		bool hitflag = false;
		int gold_rel = base->get_sentence()->get_rel(mod);
		for(int i : l){
			if(i == gold_rel)
				hitflag = true;
			ret.push_back(new StateTemp(base, mod, head, i, feat, scores));
		}
		if(iftraining && !hitflag){
			// only add gold for training when gold is possible
			if(base->is_correct() && get_sentence()->is_correct(mod, head))
				ret.push_back(new StateTemp(base, mod, head, gold_rel, feat, scores));
		}
		return ret;
	}
	State* stablize(){	// ..., what a design !=_=
		State* one = base->copy();
		one->stablize(this);
		return one;
	}
};

#endif