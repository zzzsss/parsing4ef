#include "Scorer.h"
#include "FeatureManager.h"
#include "State.h"
#include "../tools/DpTools.h"

int Scorer::num_feature = 0;
int Scorer::num_miss = 0;

//attach scores to them, check cache first
// -- analyse this one
void Scorer::score_them(vector<StateTemp>& them, FeatureManager& fm)
{
	vector<int> indexes;
	vector<Feature*> to_score;
	// collect the ones to score
	for(unsigned i = 0; i < them.size(); i++){
		Feature* ff = them[i].get_feature();
		auto iter = cache.find(ff->get_ident());	// will throw if no feature
		if(iter == cache.end()){
			indexes.push_back(i);
			to_score.push_back(ff);
		}
		else	// directly assign score
			them[i].set_score(iter->second);
	}
	// score them by model
	vector<Input> inputs;
	for(unsigned i = 0; i < indexes.size(); i++)
		inputs.push_back(fm.feature_expand(to_score[i], them[indexes[i]].get_sentence()));
	auto outputs = model->forward(inputs);
	// put them back
	for(unsigned i = 0; i < indexes.size(); i++){
		Score* s = new Score(inputs[i], outputs[i]);
		records.push_back(s);
		cache[(to_score[i])->get_ident()] = s;
		them[indexes[i]].set_score(s);
	}
	// stat
	num_feature += them.size();
	num_miss += indexes.size();
	return;
}

void Scorer::backprop_them(vector<State*>& them, vector<REAL>& grad, int div)
{
	vector<Input> vo;
	vector<int> vi;
	vector<REAL> vg;
	for(unsigned i = 0; i < them.size(); i++){
		int n = them[i]->append_si(vo, vi);
		CHECK_EQUAL(n, them[i]->get_numarc());
		for(int j = 0; j < n; j++)
			vg.push_back(grad[i]/div);
	}
	model->backward(vo, vi, vg);
}
