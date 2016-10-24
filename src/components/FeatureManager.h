#ifndef _EF_COMP_FEATURE_MANAGER
#define _EF_COMP_FEATURE_MANAGER

#include "Feature.h"
#include "StateTemp.h"
#include <unordered_map>
#include <vector>
using namespace std;
#include "../ef/DpDictionary.h"
#include "../model/Model.h"

class State;

// FeatureManager is the specification for the input (features) of the model.
// -- it also decideds the input size and helps initializing the model, and stores the Feature*
/*
-- Descriptions for the fss option: Feature Specification String (separated by |)
Distance: d-<n1>-<n2>
Word&Pos: <n>-<span>	//default-span:1
Label:    l-<n>
*Node: m/h + [cdptrlnf]<num> (nchild,nchild-backward,parent,top-list,right,left,near,far)
*Node-example: mr1(rightmost child for m-self), ht1n2(second nearest child to m of the top2 node of h), ...
!! Currently only support one-digit number !!

// TODO: NON_EXIST should be more fine-grained??, like -2, -3, ...
*/

// This one is slightly coupled with DpDictionary ... todo ...
class FeatureManager{
private:
	// static for final index of nn
	static const int INDEX_BIAS_W;		// = 1;		// only one kind of non-exist for words and pos
	static const int INDEX_BIAS_L;		// = 2;		// ... for label
	static const int INDEX_DIST_MAX;	// = 50;	// NON_DIS as not-exist
	// 
	static const int NON_EXIST;			// = -1;	//same as State::NOPE_YET
	static const int NON_DIS;			// = 0;		// only 0 is not used for distance
	static const int NON_NOPELABEL;		// = -2;	// label for no nodes
	//
	DpDictionary* dictionary;
	vector<Feature*> records;		//for final releasing
	// feature specifications for the nodes
	unordered_map<string, int> index;	// the place for this feature in the Feature
	vector<int> spans;					// size of the span in the origin surface string (default 1)
	vector<pair<int, int>> distance_pairs;
	vector<int> labels;
public:
	// settle the final indexes
	static int settle_word(int x){ return x + INDEX_BIAS_W; }
	static int settle_distance(int x){		// [0, INDEX_DIST_MAX*2]
		if(x < -INDEX_DIST_MAX)
			x = -INDEX_DIST_MAX;
		else if(x > INDEX_DIST_MAX)
			x = INDEX_DIST_MAX;
		return x + INDEX_DIST_MAX;
	}
	static int settle_label(int x){ return x + INDEX_BIAS_L; }

	void clear(){
		// clear the states
		for(auto* s : records)
			delete s;
		records.clear();
	}
	~FeatureManager(){ clear(); }
	FeatureManager(const string& fss, DpDictionary* d, int ef_mode);		//ef_mode is only for checking
	Feature* make_feature(State* s, int m, int h);

	// helpers for the size
	int num_nodes(){ return spans.size(); }	// plain nodes
	int num_nodes_all(){	// including the context window
		int s = 0;
		for(int i : spans)
			s += i;
		return s;
	}
	int num_distances(){ return distance_pairs.size(); }
	int num_labels(){ return labels.size(); }
	// for mach
	int get_wind(){ return dictionary->num_word() + INDEX_BIAS_W; }	// input dim, vocab's size + bias
	int get_pind(){ return dictionary->num_pos() + INDEX_BIAS_W; }
	int get_dind(){ return INDEX_DIST_MAX * 2 + 1; }
	int get_lind(){ return dictionary->num_rel() + INDEX_BIAS_L; }

	// for pretty-looking
	void feature_them(vector<StateTemp>& them){
		for(auto& i : them)
			i.fetch_feature(this);
	}

	// prepare expanded feature
	Input feature_expand(Feature* ff, DP_PTR sent);
};

#endif
