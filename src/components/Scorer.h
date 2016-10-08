#ifndef _EF_COMP_SCORER
#define _EF_COMP_SCORER

#include "StateTemp.h"
#include "../model/Model.h"
#include <unordered_map>
using std::unordered_map;

class Scorer{
private:
	Model* model;
	vector<Score*> records;		//for final releasing
	unordered_map<Feature, Score*, Feature_hasher> cache;
public:
	Scorer(Model* m): model(m){}
	~Scorer(){ clear(); }
	void score_them(vector<StateTemp>& them, FeatureManager& fm);	//attach scores to them
	void backprop_them(vector<State*>& them, vector<REAL>& grad);
	void clear(){
		// clear the states
		for(auto* s : records)
			delete s;
		records.clear();
		cache.clear();	// bacause feature is related to specified sentence
	}
};

#endif