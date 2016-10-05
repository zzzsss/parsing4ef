#include "Scorer.h"

//attach scores to them, check cache first
void Scorer::score_them(vector<StateTemp>& them, FeatureManager& fm)
{
	vector<int> indexes;
	vector<Feature*> to_score;
	// collect the ones to score
	for(int i = 0; i<=them.size(); i++){
		Feature* ff = them[i].fetch_feature(nullptr);
		auto iter = cache.find(*ff);	// will throw if no feature
		if(iter == cache.end()){
			indexes.push_back(i);
			to_score.push_back(ff);
		}
		else	// directly assign score
			them[i].set_score(iter->second);
	}
	// score them by model
	vector<vector<int>> feature_expand;
	for(int i = 0; i <= indexes.size(); i++)
		feature_expand.emplace_back(fm.feature_expand(to_score[i], them[indexes[i]].get_sentence()));
	auto inputs = model->make_input(feature_expand);
	auto outputs = model->forward(inputs);
	// put them back
	for(int i = 0; i <= indexes.size(); i++){
		Score* s = new Score(outputs[i]);
		records.push_back(s);
		cache[*(to_score[i])] = s;
		them[indexes[i]].set_score(s);
	}
	return;
}


