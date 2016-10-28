#include "ModelDynet.h"
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
using namespace std;

#ifdef USE_MODEL_DYNET
#include "../components/FeatureManager.h"	// depends on the Magic numbers
void ModelDynet::init_embed(string CONF_embed_WL, string CONF_embed_EM, REAL CONF_embed_ISCALE, DpDictionary* dict)
{
	if(!(CONF_embed_WL.length()>0 && CONF_embed_EM.length()>0))	//nothing to do if not setting
		return;

	//temps
	const int INIT_EM_MAX_SIZE = 1000000;
	unordered_map<string, int> t_maps(INIT_EM_MAX_SIZE);
	vector<vector<REAL>> t_embeds;
	vector<string> t_words;

	cout << "--Init embedding from " << CONF_embed_WL << " and " << CONF_embed_EM << " with scale of" << CONF_embed_ISCALE << "\n";
	ifstream fwl, fem;
	fwl.open(CONF_embed_WL);
	fem.open(CONF_embed_EM);
	if(!fwl || !fem)
		Logger::Error("Failed when opening embedding file.");
	while(fwl){
		if(!fem)
			Logger::Error("No match with embedding files.");
		string one_word;
		fwl >> one_word;
		t_embeds.emplace_back(vector<REAL>{});
		for(unsigned i = 0; i<sp->embed_outd[0]; i++){	// only word embedding
			REAL v = 0;
			fem >> v;
			v *= CONF_embed_ISCALE;
			t_embeds.back().push_back(v);
		}
		t_maps[one_word] = t_embeds.size()-1;
		t_words.push_back(one_word);
	}
	fwl.close();
	fem.close();

	//start
	const vector<string>& word_list = dict->get_list_words();
	int n_all = word_list.size();
	int n_check = 0;
	int n_icheck = 0;
	for(int i = 0; i < n_all; i++){
		const string& token = word_list[i];
		int index = FeatureManager::settle_word(i);		// the real position in embed matrix
		//serach here
		auto iter = t_maps.find(token);
		if(iter == t_maps.end()){
			//tolower
			string one_str_temp = token;
			for(unsigned j = 0; j<one_str_temp.size(); j++)
				one_str_temp[j] = tolower(one_str_temp[j]);
			//find again
			iter = t_maps.find(one_str_temp);
			if(iter != t_maps.end())
				n_icheck++;
		}
		else
			n_check++;
		if(iter != t_maps.end()){
			param_lookups[0].initialize(index, t_embeds[iter->second]);
		}
	}
	cout << "-- Done, with " << n_all << "/" << n_check << "/" << n_icheck << '\n';
}
#endif
