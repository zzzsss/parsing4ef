#include "State.h"

// 0. basics
State* State::make_empty(DP_PTR s, int opt)
{
	switch(opt){
	case EF_STD:
		return new EfstdState(s);
	case EF_EAGER:
		return new EfeagerState(s);
	defautl:
		throw runtime_error("Unkonw ef mode.");
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
vector<StateTemp*> EfstdState::expand()
{
	// In EF-std, only nodes at the top level will be considered as the head
	vector<StateTemp*> them;
	// for all the nodes that have no heads
	int cur = nb_right[0];
	while(cur != NOPE_YET){
		// left or right
		if(nb_left[cur] != NOPE_YET)
			them.push_back(new StateTemp(this, cur, nb_left[cur]));
		if(nb_right[cur] != NOPE_YET)
			them.push_back(new StateTemp(this, cur, nb_right[cur]));
		cur = nb_right[cur];
	}
	return them;
}

// EasyFirst-Eager
vector<StateTemp*> EfeagerState::expand()
{
	// In EF-eager, we need to consider the nodes on the spine as the head
	vector<StateTemp*> them;
	// for all the nodes that have no heads
	int cur = nb_right[0];
	int one = NOPE_YET;
	while(cur != NOPE_YET){
		// left or right spine
		one = nb_left[cur];
		while(one != NOPE_YET){
			them.push_back(new StateTemp(this, cur, one));
			one = ch_right[one];
		}
		one = nb_right[cur];
		while(one != NOPE_YET){
			them.push_back(new StateTemp(this, cur, one));
			one = ch_left[one];
		}
		cur = nb_right[cur];
	}
	return them;
}

