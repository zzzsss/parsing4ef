#ifndef _EF_COMP_SCORER
#define _EF_COMP_SCORER

#include "StateTemp.h"
#include "../model/Model.h"
#include "../tools/DpTools.h"
#include <unordered_map>
using std::unordered_map;

class Scorer{
private:
	static int num_feature;
	static int num_miss;
	Model* model;
	vector<Score*> records;		//for final releasing
	unordered_map<string, Score*> cache;
public:
	Scorer(Model* m): model(m){}
	~Scorer(){ clear(); }
	void score_them(vector<StateTemp>& them, FeatureManager& fm);	//attach scores to them
	void backprop_them(vector<State*>& them, vector<REAL>& grad, int div);
	void clear(){
		// clear the states
		for(auto* s : records)
			delete s;
		records.clear();
		cache.clear();	// bacause feature is related to specified sentence
		model->clear();	// remember this for Input* and Output*
	}
	static void report_and_reset(){
		Logger::get_output() << "- features calculate/all:" << num_miss << "/" << num_feature << endl;
		num_miss = num_feature = 0;
	}
};

#endif